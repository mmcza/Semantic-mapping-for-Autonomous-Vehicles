#!/usr/bin/env python3

import os
import glob
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_datasets(input_dir, output_dir):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # Find all dataset directories matching the pattern
    dataset_dirs = sorted(glob.glob(os.path.join(input_dir, 'images_2025_*')))
    
    if not dataset_dirs:
        print(f"No matching directories found in {input_dir}")
        return
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    
    # Track statistics
    total_images = 0
    total_masks = 0
    
    # Process each dataset directory
    for dataset_dir in dataset_dirs:
        dir_name = os.path.basename(dataset_dir)
        print(f"Processing {dir_name}...")
        
        # Find all images
        image_dir = os.path.join(dataset_dir, 'images', 'default')
        if not os.path.exists(image_dir):
            print(f"  Warning: Image directory not found at {image_dir}")
            continue
            
        image_files = glob.glob(os.path.join(image_dir, '*.png'))
        
        # Copy images
        for img_path in tqdm(image_files, desc="Copying images"):
            img_filename = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, 'images', img_filename)
            shutil.copy2(img_path, dest_path)
        
        total_images += len(image_files)
        
        # Find all mask files
        mask_dir = os.path.join(dataset_dir, 'visualizations')
        if not os.path.exists(mask_dir):
            print(f"  Warning: Mask directory not found at {mask_dir}")
            continue
            
        mask_files = glob.glob(os.path.join(mask_dir, '*_id_mask.png'))
        
        # Process and copy masks
        for mask_path in tqdm(mask_files, desc="Processing masks"):
            mask_filename = os.path.basename(mask_path)
            dest_path = os.path.join(output_dir, 'masks', mask_filename)
            
            # Read mask, modify it, and save
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask == 7] = 3
            cv2.imwrite(dest_path, mask)
        
        total_masks += len(mask_files)
    
    print(f"\nProcessing complete:")
    print(f"- Copied {total_images} images to {os.path.join(output_dir, 'images')}")
    print(f"- Processed and copied {total_masks} masks to {os.path.join(output_dir, 'masks')}")

def main():
    parser = argparse.ArgumentParser(description="Copy and process dataset images and masks")
    parser.add_argument("--input_dir", required=True, help="Directory containing dataset folders")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed data")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    process_datasets(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()