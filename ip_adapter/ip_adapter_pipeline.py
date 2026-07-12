import os, sys, json, re, requests, gc
import numpy as np
import cv2
import torch
from PIL import Image
from io import BytesIO
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

OUT_DIR = os.path.expanduser('~/mllm_style_transfer/ip_adapter/outputs')
WT_DIR  = os.path.expanduser('~/mllm_style_transfer/ip_adapter/weights')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WT_DIR,  exist_ok=True)


def download_image(url, path=None, size=(512, 512)):
    # some CDNs need a user-agent header
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
    img = Image.open(BytesIO(r.content)).convert('RGB').resize(size, Image.LANCZOS)
    if path:
        img.save(path)
        print(f'  saved {path}')
    return img


# ── images ────────────────────────────────────────────────────────────────────
print('\ndownloading images...')
content = download_image(
    'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512&q=80',
    f'{OUT_DIR}/content.jpg'
)
# Van Gogh Starry Night — style reference for sky region
style_sky = download_image(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png',
    f'{OUT_DIR}/style_sky.jpg'
)
# Monet Water Lilies — style reference for foreground region
style_fore = download_image(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png',
    f'{OUT_DIR}/style_fore.jpg'
)
print('images ready')


# ── helper functions ──────────────────────────────────────────────────────────
def overlay_mask(image, mask, color=(255, 80, 80), alpha=0.4):
    arr = np.array(image.convert('RGB')).astype(float)
    ov = np.zeros_like(arr)
    ov[mask > 127] = color
    return Image.fromarray((arr*(1-alpha)+ov*alpha).astype(np.uint8))


def make_boundary_band(mask, band_px=15):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px*2+1, band_px*2+1))
    return cv2.subtract(cv2.dilate(mask, k), cv2.erode(mask, k))


def boundary_band_lpips(lpips_fn, original, stylised, mask, band_px=15):
    band = make_boundary_band(mask, band_px)
    bt = torch.from_numpy((band > 127).astype(np.float32)).to(DEVICE)
    def to_t(img):
        arr = np.array(img.convert('RGB')).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE) * bt
    with torch.no_grad():
        return lpips_fn(to_t(original), to_t(stylised)).item()


def overlay_band(img, band, color=(255, 255, 0)):
    arr = np.array(img.convert('RGB')).copy()
    arr[band > 127] = color
    return Image.fromarray(arr)


def parse_instruction(instruction):
    pairs = []
    matches = re.findall(
        r'(?:the\s+)?([a-zA-Z ]+?)\s+like\s+([A-Za-z_ ]+?)(?:\s+and\s+|\s*,\s*|$)',
        instruction, re.IGNORECASE
    )
    for region, style in matches:
        region = re.sub(r'^(style|make|render|paint)\s+', '', region.strip(), flags=re.IGNORECASE).strip()
        region = re.sub(r'^the\s+', '', region, flags=re.IGNORECASE).strip()
        style = style.strip().replace('_', ' ')
        if region and style:
            pairs.append({'region': region, 'style': style})
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# PART A: Grounding DINO + SAM
# run first, save masks to disk, then free GPU before loading diffusion models
# ══════════════════════════════════════════════════════════════════════════════
print('\n--- part A: mask generation ---')
print(f'GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

import transformers as tf_module

# transformers >= 4.39 removed get_head_mask from BertModel
if not hasattr(tf_module.models.bert.modeling_bert.BertModel, 'get_head_mask'):
    from transformers.modeling_utils import ModuleUtilsMixin
    tf_module.models.bert.modeling_bert.BertModel.get_head_mask = \
        ModuleUtilsMixin.get_head_mask

gdino_pth = f'{WT_DIR}/gdino.pth'
gdino_cfg = f'{WT_DIR}/gdino_config.py'
sam_pth   = f'{WT_DIR}/sam_vit_h.pth'

if not os.path.exists(gdino_pth):
    os.system(f'wget -q -O {gdino_pth} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
if not os.path.exists(gdino_cfg):
    os.system(f'wget -q -O {gdino_cfg} https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py')
if not os.path.exists(sam_pth):
    os.system(f'wget -q -O {sam_pth} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T_gdino
from segment_anything import sam_model_registry, SamPredictor

gdino_model   = load_model(gdino_cfg, gdino_pth)
sam           = sam_model_registry['vit_h'](checkpoint=sam_pth).to(DEVICE)
sam_predictor = SamPredictor(sam)
print(f'DINO + SAM loaded | GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

gdino_transform = T_gdino.Compose([
    T_gdino.RandomResize([800], max_size=1333),
    T_gdino.ToTensor(),
    T_gdino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def text_to_mask(image_pil, prompt, box_threshold=0.3, text_threshold=0.25):
    W, H = image_pil.size
    img_t, _ = gdino_transform(image_pil, None)
    boxes, scores, _ = predict(
        model=gdino_model, image=img_t, caption=prompt,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )
    if len(boxes) == 0:
        # retry with lower threshold before giving up
        boxes, scores, _ = predict(
            model=gdino_model, image=img_t, caption=prompt,
            box_threshold=0.15, text_threshold=0.15,
        )
    if len(boxes) == 0:
        print(f'  no box found for "{prompt}"')
        return np.zeros((H, W), dtype=np.uint8)

    boxes_px = boxes.clone()
    boxes_px[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_px[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_px[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_px[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
    boxes_px = boxes_px.cpu().numpy()
    print(f'  DINO: {len(boxes)} box(es) for "{prompt}"')

    sam_predictor.set_image(np.array(image_pil.convert('RGB')))
    combined = np.zeros((H, W), dtype=np.uint8)
    for box in boxes_px:
        m, _, _ = sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=np.array(box, dtype=np.float32)[None, :],
            multimask_output=False,
        )
        combined = np.maximum(combined, (m[0] * 255).astype(np.uint8))
    return combined


print('segmenting sky...')
mask_sky  = text_to_mask(content, 'sky')
print('segmenting mountain...')
mask_fore = text_to_mask(content, 'mountain')

mask_sky_path  = f'{OUT_DIR}/mask_sky.png'
mask_fore_path = f'{OUT_DIR}/mask_fore.png'
Image.fromarray(mask_sky).save(mask_sky_path)
Image.fromarray(mask_fore).save(mask_fore_path)
print(f'masks saved')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(content);                                       axes[0].set_title('Content');         axes[0].axis('off')
axes[1].imshow(overlay_mask(content, mask_sky));               axes[1].set_title('sky mask');        axes[1].axis('off')
axes[2].imshow(overlay_mask(content, mask_fore, (80,130,255)));axes[2].set_title('mountain mask');   axes[2].axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/masks_overview.png', dpi=150)
plt.close()
print(f'saved masks_overview.png')

# free Part A models before loading diffusion backbone
del gdino_model, sam, sam_predictor
gc.collect()
torch.cuda.empty_cache()
print(f'GPU free after clearing DINO+SAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')


# ══════════════════════════════════════════════════════════════════════════════
# PART B: SD1.5-Inpainting + IP-Adapter
# load masks from disk, run reference-image conditioned inpainting
# ══════════════════════════════════════════════════════════════════════════════
print('\n--- part B: IP-Adapter inpainting ---')

from diffusers import StableDiffusionInpaintPipeline
from ip_adapter import IPAdapter

# SD1.5 is used here rather than SDXL because IP-Adapter weights are
# pretrained on SD1.5 features and both fit on 8GB in fp16
print('loading SD1.5-Inpainting...')
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    torch_dtype=torch.float16,
    safety_checker=None,
    cache_dir=WT_DIR,
).to(DEVICE)
print(f'SD1.5 loaded | GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

ip_ckpt = f'{WT_DIR}/ip-adapter_sd15.bin'
if not os.path.exists(ip_ckpt):
    os.system(
        f'wget -q -O {ip_ckpt} '
        'https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin'
    )

ip_model = IPAdapter(
    sd_pipe=pipe,
    image_encoder_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    ip_ckpt=ip_ckpt,
    device=DEVICE,
)
print(f'IP-Adapter loaded | GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')


def inpaint_region(content_img, mask_arr, style_ref_img,
                   prompt='artistic style painting',
                   num_steps=30, guidance_scale=7.5,
                   ip_scale=0.7, seed=42):
    gen = torch.Generator(DEVICE).manual_seed(seed)
    mask_pil = Image.fromarray(mask_arr).convert('L')
    result = ip_model.generate(
        pil_image=style_ref_img,
        prompt=prompt,
        negative_prompt='ugly, blurry, distorted, watermark',
        image=content_img,
        mask_image=mask_pil,
        num_samples=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        scale=ip_scale,

    )
    return result[0]


# reload masks from disk
mask_sky  = np.array(Image.open(mask_sky_path).convert('L'))
mask_fore = np.array(Image.open(mask_fore_path).convert('L'))

# single-region test first to confirm mechanism works
print('\nsingle-region: sky -> Van Gogh...')
result_sky = inpaint_region(
    content_img=content,
    mask_arr=mask_sky,
    style_ref_img=style_sky,
    prompt='sky painted in impressionist style, oil painting',
    ip_scale=0.7,
)
result_sky.save(f'{OUT_DIR}/result_sky_only.png')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(content);                              axes[0].set_title('Content');              axes[0].axis('off')
axes[1].imshow(style_sky);                            axes[1].set_title('Style ref (Van Gogh)'); axes[1].axis('off')
axes[2].imshow(Image.fromarray(mask_sky), cmap='gray');axes[2].set_title('Mask: sky');          axes[2].axis('off')
axes[3].imshow(result_sky);                           axes[3].set_title('Result: sky styled');   axes[3].axis('off')
plt.suptitle('Single-region IP-Adapter inpainting — sky region', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/single_region_sky.png', dpi=150)
plt.close()
print('saved single_region_sky.png')

# sequential multi-region: foreground on top of the sky result
print('\nmulti-region: foreground -> Monet (using sky result as base)...')
result_multi = inpaint_region(
    content_img=result_sky,
    mask_arr=mask_fore,
    style_ref_img=style_fore,
    prompt='mountains landscape in impressionist style, oil painting',
    ip_scale=0.7,
)
result_multi.save(f'{OUT_DIR}/result_multi_region.png')

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
axes[0].imshow(content);      axes[0].set_title('Content');             axes[0].axis('off')
axes[1].imshow(style_sky);    axes[1].set_title('Style 1 (Van Gogh)'); axes[1].axis('off')
axes[2].imshow(style_fore);   axes[2].set_title('Style 2 (Monet)');    axes[2].axis('off')
axes[3].imshow(result_sky);   axes[3].set_title('After sky');           axes[3].axis('off')
axes[4].imshow(result_multi); axes[4].set_title('After foreground');    axes[4].axis('off')
plt.suptitle('Sequential multi-region IP-Adapter inpainting', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/multi_region_sequential.png', dpi=150)
plt.close()
print('saved multi_region_sequential.png')


# ── boundary LPIPS evaluation ─────────────────────────────────────────────────
print('\n--- boundary LPIPS evaluation ---')
import lpips
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

lpips_sky_ip  = boundary_band_lpips(lpips_fn, content, result_multi, mask_sky)
lpips_fore_ip = boundary_band_lpips(lpips_fn, content, result_multi, mask_fore)

# NST baseline from Phase 2 notebook — the number to beat
NST_SKY  = 0.0759
NST_FORE = 0.1622

print(f'\n{"="*55}')
print(f'  BOUNDARY LPIPS: IP-Adapter vs NST Baseline')
print(f'{"="*55}')
print(f'  {"region":<20} {"IP-Adapter":>12} {"NST baseline":>14} {"delta"}')
print(f'  {"-"*55}')
print(f'  {"sky":<20} {lpips_sky_ip:>12.4f} {NST_SKY:>14.4f}  {NST_SKY-lpips_sky_ip:+.4f}')
print(f'  {"foreground":<20} {lpips_fore_ip:>12.4f} {NST_FORE:>14.4f}  {NST_FORE-lpips_fore_ip:+.4f}')
print(f'{"="*55}')
print('positive delta = IP-Adapter has less boundary leakage than NST')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(result_multi)
axes[0].set_title('Multi-region result')
axes[0].axis('off')
axes[1].imshow(overlay_band(result_multi, make_boundary_band(mask_sky)))
axes[1].set_title(f'Sky boundary band\nLPIPS={lpips_sky_ip:.4f}')
axes[1].axis('off')
axes[2].imshow(overlay_band(result_multi, make_boundary_band(mask_fore)))
axes[2].set_title(f'Foreground boundary band\nLPIPS={lpips_fore_ip:.4f}')
axes[2].axis('off')
plt.suptitle('Boundary LPIPS evaluation on IP-Adapter outputs', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/boundary_lpips_eval.png', dpi=150)
plt.close()
print('saved boundary_lpips_eval.png')


# ── ip_scale sweep ────────────────────────────────────────────────────────────
# checking what scale gives the best trade-off between style fidelity
# and content preservation for this content/style pair
print('\n--- ip_scale sweep ---')
ip_scales    = [0.3, 0.5, 0.7, 0.9]
scale_images = []
scale_lpips  = []

for scale in ip_scales:
    print(f'  ip_scale={scale}...', end=' ', flush=True)
    res = inpaint_region(
        content_img=content,
        mask_arr=mask_sky,
        style_ref_img=style_sky,
        prompt='sky painted in impressionist style',
        ip_scale=scale,
    )
    bl = boundary_band_lpips(lpips_fn, content, res, mask_sky)
    scale_images.append(res)
    scale_lpips.append(bl)
    print(f'boundary LPIPS={bl:.4f}')

best_idx = scale_lpips.index(min(scale_lpips))
print(f'best ip_scale: {ip_scales[best_idx]} (LPIPS={scale_lpips[best_idx]:.4f})')

fig, axes = plt.subplots(1, len(ip_scales)+1, figsize=(5*(len(ip_scales)+1), 5))
axes[0].imshow(content); axes[0].set_title('Content'); axes[0].axis('off')
for i, (scale, res, bl) in enumerate(zip(ip_scales, scale_images, scale_lpips)):
    axes[i+1].imshow(res)
    axes[i+1].set_title(f'ip_scale={scale}\nLPIPS={bl:.4f}')
    axes[i+1].axis('off')
plt.suptitle('IP-Adapter scale sweep (sky region, Van Gogh reference)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/ip_scale_sweep.png', dpi=150)
plt.close()
print('saved ip_scale_sweep.png')


# ── full end-to-end pipeline ──────────────────────────────────────────────────
print('\n--- full pipeline ---')

STYLE_REFS = {
    'Van Gogh': style_sky,
    'Monet':    style_fore,
}

REGION_PROMPTS = {
    'sky':       'sky painted in impressionist oil painting style',
    'mountain':  'mountains landscape impressionist painting style',
    'mountains': 'mountains landscape impressionist painting style',
    'forest':    'forest trees impressionist painting style',
    'water':     'water reflections impressionist painting style',
}

instruction = 'style the sky like Van Gogh and the mountain like Monet'
print(f'instruction: "{instruction}"')
pairs = parse_instruction(instruction)
print(f'parsed: {pairs}')

region_mask_map = {
    'sky':      mask_sky,
    'mountain': mask_fore,
    'mountains': mask_fore,
}

current = content.copy()
for pair in pairs:
    region     = pair['region']
    style_name = pair['style']
    style_ref  = STYLE_REFS.get(style_name)
    mask       = region_mask_map.get(region)

    if style_ref is None:
        print(f'  style "{style_name}" not in map, skipping')
        continue
    if mask is None or mask.sum() == 0:
        print(f'  no mask for "{region}", skipping')
        continue

    prompt = REGION_PROMPTS.get(region, 'artistic style painting')
    print(f'  inpainting "{region}" -> "{style_name}"')
    current = inpaint_region(
        content_img=current,
        mask_arr=mask,
        style_ref_img=style_ref,
        prompt=prompt,
        ip_scale=ip_scales[best_idx],
    )

current.save(f'{OUT_DIR}/full_pipeline_result.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(content);  axes[0].set_title('Content');       axes[0].axis('off')
axes[1].imshow(current);  axes[1].set_title('Full pipeline'); axes[1].axis('off')
axes[2].imshow(style_sky);axes[2].set_title('Style ref (sky)');axes[2].axis('off')
plt.suptitle('End-to-end: instruction -> DINO+SAM -> IP-Adapter inpainting', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/full_pipeline_overview.png', dpi=150)
plt.close()
print('saved full_pipeline_overview.png')


# ── save summary ──────────────────────────────────────────────────────────────
summary = {
    'model':         'SD1.5-Inpainting + IP-Adapter',
    'backbone':      'runwayml/stable-diffusion-inpainting',
    'ip_adapter':    'h94/IP-Adapter (ip-adapter_sd15.bin)',
    'grounding':     'GroundingDINO + SAM ViT-H',
    'results': {
        'sky_boundary_lpips':  round(lpips_sky_ip,  4),
        'fore_boundary_lpips': round(lpips_fore_ip, 4),
        'nst_sky_baseline':    NST_SKY,
        'nst_fore_baseline':   NST_FORE,
        'best_ip_scale':       ip_scales[best_idx],
    },
    'created_at': datetime.now().isoformat(),
}
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n{"="*55}')
print(f'  FINAL SUMMARY')
print(f'{"="*55}')
print(f'  sky boundary LPIPS:        {lpips_sky_ip:.4f}  (NST was {NST_SKY})')
print(f'  foreground boundary LPIPS: {lpips_fore_ip:.4f}  (NST was {NST_FORE})')
print(f'  best IP-Adapter scale:     {ip_scales[best_idx]}')
print(f'  all outputs saved to: {OUT_DIR}')
print(f'{"="*55}')
