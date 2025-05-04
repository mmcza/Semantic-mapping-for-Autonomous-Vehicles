from PIL import Image
from lang_sam import LangSAM
import os
import numpy as np
import json
import cv2
from datetime import datetime
import argparse
import glob
import shutil
from tqdm import tqdm
import torch
from torchvision import transforms

def preprocess_image(image_pil):
    # Apply CLAHE to L channel
    img_np = np.array(image_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_pil = Image.fromarray(enhanced_img)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
    ])
    return normalize(enhanced_pil)

def generate_annotations(input_dir, output_dir,
                         model_name="sam2.1_hiera_small",
                         save_visualizations=False,
                         min_area=100,
                         subset="default"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_models = [
        "sam2.1_hiera_tiny",
        "sam2.1_hiera_small",
        "sam2.1_hiera_base_plus",
        "sam2.1_hiera_large"
    ]
    if model_name not in valid_models:
        raise ValueError(f"Model must be one of: {', '.join(valid_models)}")

    images_output_dir = os.path.join(output_dir, "images", subset)
    os.makedirs(images_output_dir, exist_ok=True)
    if save_visualizations:
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
    # new: prepare manifest list
    manifest = []

    model = LangSAM(sam_type=model_name)

    # gather files
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_files += glob.glob(os.path.join(input_dir, ext))
    image_files.sort(key=lambda x: os.path.basename(x))
    image_files = image_files[::10]

    category_groups = [
        {"id": 1, "name": "Other", "color": [0,0,0]},
        {"id": 2, "name": "Sky", "color": [135,206,235]},
        {"id": 3, "name": "Building", "color": [70,70,70]},
        {"id": 4, "name": "Grass", "color": [0,128,0]},
        {"id": 5, "name": "Sand/Mud", "queries": ["Sand","Mud","Ground"], "color": [210,180,140]},
        {"id": 6, "name": "Road/Pavement","queries":["Road","Asphalt","Cobblestone","Street"],"color":[128,128,128]},
        {"id": 7, "name": "Fence", "color":[139,69,19]},
        {"id": 8, "name": "Tree", "color":[34,139,34]},
        {"id": 9, "name": "Street Furniture","queries":["Road sign","Street lamp","Street light","Traffic sign","Pole","Cone","Bike","Handicapped sign"],"color":[255,215,0]},
        {"id":10,"name":"Vehicle","queries":["Car","Truck"],"color":[255,0,0]},
        {"id":11,"name":"Person","color":[255,192,203]}
    ]

    # build queries ↔ ids
    queries = []
    query_to_id = {}
    for g in category_groups:
        if "queries" in g:
            qstr = " ".join(f"{q.lower()}." for q in g["queries"])
            queries.append(qstr)
            query_to_id[qstr] = g["id"]
        else:
            qstr = g["name"].lower() + "."
            queries.append(qstr)
            query_to_id[qstr] = g["id"]
    if "other." in queries:
        queries.remove("other.")

    id_to_color = {g["id"]: g["color"] for g in category_groups}

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            pil = Image.open(img_path).convert("RGB")
            name = os.path.basename(img_path)
            base = os.path.splitext(name)[0]
            dst_img = os.path.join(images_output_dir, name)
            shutil.copy2(img_path, dst_img)

            pil = preprocess_image(pil)
            h, w = pil.height, pil.width
            id_mask = np.ones((h, w), dtype=np.uint8)

            if save_visualizations:
                vis = np.zeros((h, w, 3), dtype=np.uint8)
                vis[:, :] = id_to_color[1]

            for q in queries:
                cid = query_to_id[q]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # adjust thresholds for furniture group
                bt, tt = (0.125, 0.2) if "street furniture" in q else (0.35, 0.3)
                res = model.predict([pil], [q], box_threshold=bt, text_threshold=tt)[0]
                masks = res.get("masks") if isinstance(res, dict) else None
                if masks is not None and hasattr(masks, "size") and masks.size>0:
                    cm = np.zeros((h, w), dtype=np.uint8)
                    for m in masks:
                        bm = (m>0.5).astype(np.uint8)
                        cm = np.maximum(cm, bm)
                    id_mask[cm>0] = cid
                    if save_visualizations and cm.any():
                        vis[cm>0] = id_to_color[cid]

            # save optional visuals
            if save_visualizations:
                Image.fromarray(id_mask).save(os.path.join(visualizations_dir, f"{base}_id_mask.png"))
                Image.fromarray(vis).save(os.path.join(visualizations_dir, f"{base}_seg.png"))
                blend = Image.fromarray((0.5*np.array(pil)+(0.5*vis)).astype(np.uint8))
                blend.save(os.path.join(visualizations_dir, f"{base}_blend.png"))

            # write mask and record manifest
            masks_dir = os.path.join(output_dir, "robo_masks", subset)
            os.makedirs(masks_dir, exist_ok=True)
            mask_path = os.path.join(masks_dir, f"{base}.png")
            Image.fromarray(id_mask).save(mask_path)

            manifest.append({
                "image": f"images/{subset}/{name}",
                "mask": f"robo_masks/{subset}/{base}.png"
            })

        except Exception as e:
            print(f"Skipped {img_path}: {e}")

    # dump manifest
    with open(os.path.join(output_dir, "robo_manifest.json"), "w") as f:
        json.dump({"annotations": manifest}, f, indent=4)

    print(f"Done — {len(manifest)} entries in robo_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate semantic masks + JSON manifest"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model",
        default="sam2.1_hiera_small",
        choices=[
            "sam2.1_hiera_tiny",
            "sam2.1_hiera_small",
            "sam2.1_hiera_base_plus",
            "sam2.1_hiera_large"
        ],
    )
    parser.add_argument(
        "--subset", default="default",
        help="e.g. train/val"
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true"
    )
    parser.add_argument(
        "--min_area", type=float, default=100.0
    )

    args = parser.parse_args()
    generate_annotations(
        args.input_dir,
        args.output_dir,
        args.model,
        args.save_visualizations,
        args.min_area,
        args.subset
    )
