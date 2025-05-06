import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from configs.default_config import MASKS_DIR, CLASSES

# Define the class names and their corresponding pixel values
CLASS_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
CLASS_TO_PIXEL = {idx: pixel_val for idx, pixel_val in enumerate(CLASS_IDS)}
PIXEL_TO_CLASS = {pixel_val: idx for idx, pixel_val in enumerate(CLASS_IDS)}

def calculate_class_distribution(masks_dir):
    # Ensure masks directory is provided
    if not masks_dir:
        print("Please provide a valid masks directory path")
        return None
    
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_id_mask.png')]
    if not mask_files:
        print(f"No mask files found in {masks_dir}")
        return None
    
    print(f"Found {len(mask_files)} mask files")
    
    # Initialize counters
    total_pixels = 0
    class_pixels = {class_name: 0 for class_name in CLASSES}
    
    # Process each mask
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(masks_dir, mask_file)
        
        try:
            # Read mask as grayscale
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_file}")
                continue
                
            h, w = mask.shape
            pixels_in_mask = h * w
            total_pixels += pixels_in_mask
            
            # Count pixels for each class
            for class_idx, class_name in enumerate(CLASSES):
                pixel_value = CLASS_IDS[class_idx]
                class_pixels[class_name] += np.sum(mask == pixel_value)
            
        except Exception as e:
            print(f"Error processing {mask_file}: {str(e)}")
    
    # Calculate percentages
    percentages = {}
    for class_name in CLASSES:
        percentages[class_name] = 100.0 * class_pixels[class_name] / total_pixels if total_pixels > 0 else 0
    
    # Calculate inverse weights
    max_percentage = max(percentages.values())
    weights = {}
    for class_name in CLASSES:
        # Avoid division by zero
        if percentages[class_name] > 0:
            weights[class_name] = max_percentage / percentages[class_name]
        else:
            weights[class_name] = 1.0
    
    # Normalize weights so they sum to the number of classes
    weight_sum = sum(weights.values())
    for class_name in weights:
        weights[class_name] = weights[class_name] * len(weights) / weight_sum
    
    return {
        'total_pixels': total_pixels,
        'class_pixels': class_pixels,
        'percentages': percentages,
        'weights': weights
    }

def plot_class_distribution(stats):
    # Create a colormap suitable for more classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    
    plt.figure(figsize=(18, 10))
    
    # Pie chart of percentages
    plt.subplot(1, 2, 1)
    
    # Sort classes by percentage for better visualization
    sorted_classes = sorted(CLASSES, key=lambda c: stats['percentages'][c], reverse=True)
    labels = [f"{cls}\n({stats['percentages'][cls]:.1f}%)" for cls in sorted_classes]
    
    # Only show labels for classes with significant percentage
    threshold = 2.0  # Only show labels for classes with more than 2% representation
    explode = [0.1 if stats['percentages'][cls] < threshold else 0 for cls in sorted_classes]
    
    plt.pie(
        [stats['percentages'][cls] for cls in sorted_classes],
        labels=labels,
        autopct=lambda p: f'{p:.1f}%' if p > threshold else '',
        startangle=90,
        colors=colors,
        explode=explode
    )
    plt.title('Class Distribution in Dataset', fontsize=14)
    
    # Bar chart of weights
    plt.subplot(1, 2, 2)
    
    # Sort classes by weight for better visualization
    sorted_by_weight = sorted(CLASSES, key=lambda c: stats['weights'][c])
    
    y_pos = np.arange(len(CLASSES))
    weights = [stats['weights'][cls] for cls in sorted_by_weight]
    
    bars = plt.barh(y_pos, weights, color=colors)
    plt.yticks(y_pos, sorted_by_weight)
    plt.xlabel('Weight Value')
    plt.title('Class Weights (Inverse Frequency)', fontsize=14)
    
    # Add weight values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()

def main():
    print(f"Analyzing masks in: {MASKS_DIR}")
    
    # If no masks directory is set, ask user to provide one
    masks_dir = MASKS_DIR
    if not masks_dir:
        masks_dir = input("Please enter the path to your masks directory: ")
    
    # Calculate statistics
    stats = calculate_class_distribution(masks_dir)
    
    if stats:
        # Print results in a table format
        print("\n" + "-" * 80)
        print(f"{'Class':<16} | {'Pixels':<12} | {'Percentage':<12} | {'Weight':<12}")
        print("-" * 80)
        
        for cls in CLASSES:
            print(f"{cls:<16} | {stats['class_pixels'][cls]:<12} | {stats['percentages'][cls]:>9.2f}% | {stats['weights'][cls]:<12.4f}")
        
        print("-" * 80)
        print(f"Total Pixels: {stats['total_pixels']}")
        print("-" * 80)
        
        # Print weights in a format ready for config file
        print("\nClass weights for config file:")
        weight_values = [f"{stats['weights'][cls]:.4f}" for cls in CLASSES]
        weights_str = ', '.join(weight_values)
        print(f"FOCAL_LOSS_WEIGHTS = [{weights_str}]")
        
        # Plot results
        plot_class_distribution(stats)
    else:
        print("Failed to calculate class distribution.")

if __name__ == "__main__":
    main()