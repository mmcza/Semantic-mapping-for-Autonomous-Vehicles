import albumentations as A
import albumentations.pytorch.transforms
import cv2 as cv

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import numpy as np
import torch
from .datasets.seg_dataset import SegmentationDataset
from configs.default_config import (
    IMG_SIZE,
    DATA_DIR,
    MASKS_DIR,
    TRAIN_VAL_TEST_SPLIT,
    RANDOM_STATE,
    BATCH_SIZE,
    NUM_WORKERS,
    AUGMENTATIONS,
)

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir,
        mask_dir,
        train_transform,
        val_transform,
        test_transform,
        classes,
        train_val_test_split=(0.8, 0.1, 0.1),
        random_state=42,
        batch_size=8,
        num_workers=4, 
        img_size=(720, 1280)
    ):
        super().__init__()

        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._train_transform = train_transform
        self._val_transform = val_transform
        self._test_transform = test_transform
    
        self._train_val_test_split = train_val_test_split
        self._random_state = random_state
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._img_size = img_size
        self._classes = classes

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
    def setup(self, stage):
        images = [f[:-4] for f in os.listdir(self._image_dir) if f.endswith('.png')]
        masks = [f[:-4] for f in os.listdir(self._mask_dir) if f.endswith('.png')]
        images.sort()
        masks.sort()

        assert len(images) == len(masks), "Number of images and masks must be the same."

        # Split the dataset into train, validation, and test sets
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            images,
            masks,
            test_size=self._train_val_test_split[2],
            random_state=self._random_state,
            shuffle=True,
        )

        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images,
            train_val_masks,
            test_size=self._train_val_test_split[1] / sum(self._train_val_test_split[:2]),
            random_state=self._random_state,
            shuffle=True,
        )

        self._train_dataset = SegmentationDataset(
            image_dir=self._image_dir,
            mask_dir=self._mask_dir,
            transform=self._train_transform,
            images=train_images,
            classes=self._classes
        )

        self._val_dataset = SegmentationDataset(
            image_dir=self._image_dir,
            mask_dir=self._mask_dir,
            transform=self._val_transform,
            images=val_images,
            classes=self._classes
        )

        self._test_dataset = SegmentationDataset(
            image_dir=self._image_dir,
            mask_dir=self._mask_dir,
            transform=self._test_transform,
            images=test_images,
            classes=self._classes
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    from configs.default_config import (
        IMG_SIZE,
        DATA_DIR,
        MASKS_DIR,
        TRAIN_VAL_TEST_SPLIT,
        RANDOM_STATE,
        BATCH_SIZE,
        NUM_WORKERS,
        AUGMENTATIONS,
    )

    # Check if data paths are set
    if not DATA_DIR or not MASKS_DIR:
        print("Please set DATA_DIR and MASKS_DIR in configs/default_config.py")
        exit(1)

    # Create datamodule using config values
    dm = LBFSegmentationDataModule(
        image_dir=DATA_DIR,
        mask_dir=MASKS_DIR,
        train_transform=AUGMENTATIONS['train'],
        val_transform=AUGMENTATIONS['val'],
        test_transform=AUGMENTATIONS['test'],
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE
    )

    # Setup the datamodule
    dm.setup(stage=None)
    
    # Get the train dataset
    train_dataset = dm._train_dataset
    
    # Select 5 random samples
    sample_indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
    
    # Create a figure to display images and masks
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(15, 5 * len(sample_indices)))
    
    # If only one sample, ensure axes is 2D
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    
    class_colors = [
        [0.5, 0.5, 0.5],  # Background - gray
        [0.0, 0.0, 1.0],  # Branches - blue
        [0.0, 1.0, 0.0],  # Leaves - green  
        [1.0, 0.0, 0.0],  # Fruit - red
    ]
    
    # Display each sample
    for i, idx in enumerate(sample_indices):
        image, mask = train_dataset[idx]
        
        # Convert image from tensor to numpy
        if torch.is_tensor(image):
            # Denormalize image for display
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = image
        
        # Display original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Sample {idx}: Original Image")
        axes[i, 0].axis('off')
        
        # Create colored mask visualization
        mask_vis = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.float32)
        for c in range(4):  # 4 channels: background, branches, leaves, fruit
            mask_vis[mask[c] > 0.5] = class_colors[c]
        
        # Display colored mask
        axes[i, 1].imshow(mask_vis)
        axes[i, 1].set_title("Segmentation Mask")
        axes[i, 1].axis('off')
        
        # Display image with mask overlay
        overlay = image_np * 0.7 + mask_vis * 0.3
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Train dataset: {len(dm._train_dataset)} samples")
    print(f"Validation dataset: {len(dm._val_dataset)} samples")
    print(f"Test dataset: {len(dm._test_dataset)} samples")