import os
import sys
import torch
import typer
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamodules.seg_datamodule import SegmentationDataModule
from configs.default_config import *

def visualize_onnx_results(
    onnx_model_path: Path = typer.Argument(..., help="Path to ONNX model"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Path to images directory"),
    masks_dir: Optional[Path] = typer.Option(None, "--masks-dir", "-m", help="Path to masks directory"),
    train_val_test_split: Optional[List[float]] = typer.Option(None, "--split", help="Train/val/test split ratio"),
    random_state: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    img_size: Optional[Tuple[int, int]] = typer.Option(None, "--img-size", help="Image size (height, width)"),
    batch_size: Optional[int] = typer.Option(1, "--batch-size", "-b", help="Batch size for inference"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", "-w", help="Number of workers"),
    device: Optional[str] = typer.Option(None, "--device", help="Device for ONNX runtime (CPU or CUDA)"),
    num_samples: Optional[int] = typer.Option(10, "--num-samples", "-n", help="Number of samples to visualize"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save visualizations"),
    show_plots: Optional[bool] = typer.Option(True, "--show", help="Show plots interactively"),
):
    # Set parameters from command line or config
    _data_dir = data_dir or DATA_DIR
    _masks_dir = masks_dir or MASKS_DIR
    _train_val_test_split = train_val_test_split or TRAIN_VAL_TEST_SPLIT
    _random_state = random_state or RANDOM_STATE
    _img_size = img_size or IMG_SIZE
    _num_workers = num_workers or NUM_WORKERS
    _device = device or DEVICE
    
    # Check if output directory exists and create if not
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_everything(_random_state)
    
    print(f"Loading data from {_data_dir} and {_masks_dir}")
    print(f"Using ONNX model: {onnx_model_path}")
    print(f"Output directory: {output_dir}")

    # Initialize data module
    data_module = SegmentationDataModule(
        image_dir=_data_dir,
        mask_dir=_masks_dir,
        train_transform=AUGMENTATIONS['test'],
        val_transform=AUGMENTATIONS['test'],
        test_transform=AUGMENTATIONS['test'],
        train_val_test_split=_train_val_test_split,
        random_state=_random_state,
        batch_size=batch_size,
        num_workers=_num_workers,
        img_size=_img_size,
        classes=CLASSES
    )
    
    # Setup data splits
    data_module.setup(stage='test')
    
    # Load ONNX model
    print(f"Loading ONNX model from {onnx_model_path}")
    # Set up ONNX Runtime session - use CUDA provider if available and requested
    providers = ['CPUExecutionProvider']
    if 'cuda' in _device.lower() and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
        print("Using CUDA for ONNX Runtime inference")
    else:
        print("Using CPU for ONNX Runtime inference")
        
    session = ort.InferenceSession(str(onnx_model_path), providers=providers)
    
    # Define class IDs and their corresponding colors - normalized to 0-1 for matplotlib
    class_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    class_colors = {
        1: [0, 0, 0],          # Black - Other
        2: [135, 206, 235],    # Sky Blue - Sky
        3: [70, 70, 70],       # Dark Gray - Building/Fence
        4: [0, 128, 0],        # Green - Grass
        5: [210, 180, 140],    # Tan - Sand/Mud
        6: [128, 128, 128],    # Gray - Road/Pavement
        8: [34, 139, 34],      # Forest Green - Tree
        9: [255, 215, 0],      # Gold - Street Furniture
        10: [255, 0, 0],       # Red - Vehicle
        11: [255, 192, 203]    # Pink - Person
    }
    
    # Normalize colors to 0-1 range for matplotlib
    for class_id in class_colors:
        class_colors[class_id] = [c/255.0 for c in class_colors[class_id]]
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Process samples
    samples_processed = 0
    
    for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Generating predictions")):
        # No need to move to device, ONNX Runtime handles this
        images_np = images.numpy()
        
        # Run inference with ONNX
        ort_inputs = {'input': images_np}
        ort_outputs = session.run(['output'], ort_inputs)[0]
        
        # Convert predictions to segmentation masks
        pred_masks = torch.from_numpy(ort_outputs)
        pred_masks = torch.softmax(pred_masks, dim=1)
        pred_masks = torch.argmax(pred_masks, dim=1)
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            if samples_processed >= num_samples and num_samples > 0:
                break
            
            # Get current image
            image = images[i].cpu()
            
            # Extract ground truth mask
            if masks[i].ndim == 3 and masks[i].shape[0] == len(class_ids):
                # If mask is one-hot encoded, convert to class indices
                true_mask = torch.argmax(masks[i], dim=0).cpu().numpy()
                # Map indices back to class IDs
                true_mask_with_ids = np.zeros_like(true_mask)
                for idx, class_id in enumerate(class_ids):
                    true_mask_with_ids[true_mask == idx] = class_id
                true_mask = true_mask_with_ids
            else:
                # If mask is grayscale with class IDs
                true_mask = masks[i].cpu().numpy()
            
            # Get prediction
            pred_mask = pred_masks[i].cpu().numpy()
            
            # Denormalize image for visualization
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot original image
            axes[0].imshow(image_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Create ground truth mask visualization
            gt_mask_vis = np.zeros((true_mask.shape[0], true_mask.shape[1], 3))
            for class_id in class_colors:
                gt_mask_vis[true_mask == class_id] = class_colors[class_id]
            
            # Plot ground truth mask
            axes[1].imshow(gt_mask_vis)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            # Create prediction mask visualization
            pred_mask_vis = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
            
            # Map predicted class indices to class IDs and then to colors
            for idx, class_id in enumerate(class_ids):
                pred_mask_vis[pred_mask == idx] = class_colors[class_id]
            
            # Plot prediction mask
            axes[2].imshow(pred_mask_vis)
            axes[2].set_title("Prediction")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save or display the figure
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, f"onnx_pred_{batch_idx}_{i}.png"), bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
            
            samples_processed += 1
            if samples_processed >= num_samples and num_samples > 0:
                break
        
        if samples_processed >= num_samples and num_samples > 0:
            break
    
    print(f"Processed {samples_processed} samples.")
    if output_dir is not None:
        print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    typer.run(visualize_onnx_results)