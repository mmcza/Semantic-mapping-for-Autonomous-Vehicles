#pip install transformers torch torchvision Pillow matplotlib numpy accelerate

import os
import random
import time

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation



# --- Configuration ---
image_dir        = '/images'
output_dir       = '/segmentation_hf_b5'
model_checkpoint = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

os.makedirs(output_dir, exist_ok=True)

# Cityscapes palette (19 classes)
cityscapes_id_to_color = {
    0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70), 3: (102, 102, 156),
    4: (190, 153, 153), 5: (153, 153, 153), 6: (250, 170, 30), 7: (220, 220, 0),
    8: (107, 142, 35), 9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
    12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70), 15: (0, 60, 100),
    16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32)
}
palette = [cityscapes_id_to_color[i] for i in range(19)]

# Load model and processor
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
model           = AutoModelForSemanticSegmentation.from_pretrained(model_checkpoint)
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def decode_segmap_to_pil(seg_map, palette_list):
    h, w = seg_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(palette_list):
        rgb[seg_map == idx] = color
    return Image.fromarray(rgb)

# Gather images
all_images = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
if not all_images:
    raise RuntimeError(f"No images in {image_dir}")

# Inference loop
processed, errors = 0, 0
for fname in tqdm(all_images, desc="Segmenting"):
    try:
        img_path = os.path.join(image_dir, fname)
        img_pil  = Image.open(img_path).convert("RGB")
        inputs   = image_processor(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        up = torch.nn.functional.interpolate(
            logits,
            size=img_pil.size[::-1],
            mode='bilinear',
            align_corners=False
        )
        seg_map = up.argmax(1)[0].cpu().numpy()
        mask_pil = decode_segmap_to_pil(seg_map, palette)
        mask_pil.save(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_mask.png"))
        processed += 1
    except Exception as e:
        # create a placeholder error image
        err_img = Image.new('RGB', (200, 100), 'grey')
        d = ImageDraw.Draw(err_img)
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default()
        d.text((10, 10), f"Error: {fname}\n{e}", fill="red", font=font)
        err_img.save(os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_ERROR.png"))
        errors += 1

print(f"Done: {processed} succeeded, {errors} failed. Outputs in {output_dir}")
