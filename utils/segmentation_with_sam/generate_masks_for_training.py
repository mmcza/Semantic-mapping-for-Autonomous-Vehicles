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

def generate_annotations(input_dir, output_dir, model_name="sam2.1_hiera_small", save_visualizations=False, min_area=100, subset="default"):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Validate model name
    valid_models = [
        "sam2.1_hiera_tiny",
        "sam2.1_hiera_small", 
        "sam2.1_hiera_base_plus",
        "sam2.1_hiera_large"
    ]
    if model_name not in valid_models:
        raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
    
    # Create output directories structure
    images_output_dir = os.path.join(output_dir, "images", subset)
    
    os.makedirs(images_output_dir, exist_ok=True)
    
    if save_visualizations:
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing LangSAM with model: {model_name}")
    model = LangSAM(sam_type=model_name)
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    image_files.sort(key=lambda x: os.path.basename(x))

    # Keep only every tenth image for processing
    image_files = image_files[::10]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Define category groups with their IDs and visualization colors
    category_groups = [
        {"id": 1, "name": "Other", "color": [0, 0, 0]},  # Black
        {"id": 2, "name": "Sky", "color": [135, 206, 235]},    # Sky Blue
        {"id": 3, "name": "Building", "color": [70, 70, 70]},  # Dark Gray
        {"id": 4, "name": "Grass", "color": [0, 128, 0]},      # Green
        {"id": 5, "name": "Sand/Mud", "queries": ["Sand", "Mud"], "color": [210, 180, 140]},  # Tan
        {"id": 6, "name": "Road/Pavement", "queries": ["Road", "Asphalt", "Cobblestone"], "color": [128, 128, 128]},  # Gray
        {"id": 7, "name": "Fence", "color": [139, 69, 19]},    # Brown
        {"id": 8, "name": "Tree", "color": [34, 139, 34]},     # Forest Green
        {"id": 9, "name": "Street Furniture", "queries": ["Sign", "Lamp", "Pole", "Cone", "Bike"], "color": [255, 215, 0]},  # Gold
        {"id": 10, "name": "Vehicle", "queries": ["Car", "Truck"], "color": [255, 0, 0]},  # Red
        {"id": 11, "name": "Person", "color": [255, 192, 203]}  # Pink
    ]
    
    # Generate query list for model predictions
    queries = []
    for group in category_groups:
        if "queries" in group:
            combined_query = " ".join([f"{q.lower()}." for q in group["queries"]])
            queries.append(combined_query)
        else:
            queries.append(group["name"].lower() + ".")
    
    # Remove "other." query
    if "other." in queries:
        queries.remove("other.")
    
    # Map from combined query to category id
    query_to_id = {}
    for i, group in enumerate(category_groups):
        if "queries" in group:
            combined_query = " ".join([f"{q.lower()}." for q in group["queries"]])
            query_to_id[combined_query] = group["id"]
        else:
            query_to_id[group["name"].lower() + "."] = group["id"]
    
    # Map from category id to color
    id_to_color = {}
    for group in category_groups:
        id_to_color[group["id"]] = group["color"]
    
    # Prepare categories for COCO dataset
    coco_categories = []
    for group in category_groups:
        if "queries" in group:
            for query in group["queries"]:
                coco_categories.append({
                    "id": group["id"],
                    "name": query,
                    "supercategory": group["name"]
                })
        else:
            coco_categories.append({
                "id": group["id"],
                "name": group["name"],
                "supercategory": group["name"]
            })
    
    # Initialize COCO dataset
    coco_dataset = {
        "info": {
            "description": "Generated annotations using LangSAM",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "images": [],
        "annotations": [],
        "categories": coco_categories
    }

    print(queries)
    
    image_id = 1
    annotation_id = 1
    total_images_processed = 0
    total_annotations_created = 0
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            image_pil = Image.open(image_path).convert("RGB")
            image_filename = os.path.basename(image_path)
            image_base = os.path.splitext(image_filename)[0]
            
            # Copy image to the appropriate directory
            image_dest = os.path.join(images_output_dir, image_filename)
            shutil.copy2(image_path, image_dest)

            print(f"\nProcessing {image_filename}")
            
            # Initialize an ID mask with background/other (category 1)
            height, width = image_pil.height, image_pil.width
            id_mask = np.ones((height, width), dtype=np.uint8) 
            
            # Create visualization image (if needed)
            if save_visualizations:
                mask_image = np.zeros((height, width, 3), dtype=np.uint8)
                mask_image[:, :] = id_to_color[1]
            
            # Track masks found for all queries
            masks_found = {group["name"]: 0 for group in category_groups}
            
            # Process each query individually
            for query_idx, query in enumerate(queries):
                try:
                    print(f"  Processing query {query_idx+1}/{len(queries)}: '{query}'")
                    
                    # Get the category_id for this query
                    category_id = query_to_id.get(query, 1)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    result = model.predict([image_pil], [query], box_threshold=0.35, text_threshold=0.3)
                    
                    if result and isinstance(result[0], dict) and 'masks' in result[0] and result[0]['masks'] is not None:
                        masks = result[0]['masks']
                        
                        # Handle case where masks might be a list without size attribute
                        if hasattr(masks, 'size') and masks.size > 0:
                            print(f"    Found {len(masks)} masks")
                            masks_found[next(g["name"] for g in category_groups if g["id"] == category_id)] += len(masks)
                            
                            # Combine all masks for this query
                            combined_mask = np.zeros((height, width), dtype=np.uint8)
                            
                            # Process each mask for this query
                            for mask in masks:
                                binary_mask = (mask > 0.5).astype(np.uint8)
                                combined_mask = np.maximum(combined_mask, binary_mask)
                            
                            # Update ID mask with current category ID
                            id_mask[combined_mask > 0] = category_id
                            
                            if save_visualizations and combined_mask.any():
                                color = id_to_color.get(category_id, [0, 0, 0])
                                mask_image[combined_mask > 0] = color
                        else:
                            print(f"    No valid masks found")
                    else:
                        print(f"    No masks returned")
                
                except Exception as query_error:
                    print(f"    Error processing query: {str(query_error)}")
                    continue
            
            # Print summary of masks found
            print("  Masks found:")
            for group_name, count in masks_found.items():
                if count > 0:
                    print(f"    - {group_name}: {count}")
            
            # Save visualization images (if requested)
            if save_visualizations:
                # Save the ID mask as grayscale image
                id_mask_path = os.path.join(visualizations_dir, f"{image_base}_id_mask.png")
                Image.fromarray(id_mask).save(id_mask_path)
                
                # Save colored segmentation
                mask_path = os.path.join(visualizations_dir, f"{image_base}_segmentation.png")
                Image.fromarray(mask_image).save(mask_path)
                
                # Save blend visualization
                alpha = 0.5
                blend = Image.fromarray((alpha * np.array(image_pil) + (1-alpha) * mask_image).astype(np.uint8))
                blend.save(os.path.join(visualizations_dir, f"{image_base}_visualization.png"))
            
            total_images_processed += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"\nProcessing complete.")
    print(f"- Processed {total_images_processed} images")
    if save_visualizations:
        print(f"- Saved visualizations to {visualizations_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate segmentation masks and COCO annotations")
    parser.add_argument("--input_dir", required=True, help="Directory with input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save annotations and visualizations")
    parser.add_argument("--model", default="sam2.1_hiera_small", choices=[
        "sam2.1_hiera_tiny",
        "sam2.1_hiera_small",
        "sam2.1_hiera_base_plus",
        "sam2.1_hiera_large"
    ], help="SAM model to use")
    parser.add_argument("--subset", default="default", help="Subset name (e.g., 'train', 'val')")
    parser.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    parser.add_argument("--min_area", type=float, default=100.0, help="Minimum contour area to include in annotations")
    
    args = parser.parse_args()
    generate_annotations(args.input_dir, args.output_dir, args.model, 
                         args.save_visualizations, args.min_area, args.subset)
    