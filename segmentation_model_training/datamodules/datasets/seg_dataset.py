from torch.utils.data import Dataset
import cv2 as cv
from albumentations import Compose
import numpy as np
import os
import torch

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, images, classes, transform=None):
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self._images = images
        self._classes = classes

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image, mask = self._get_image_and_mask(idx)

        if self._transform:
            augmented = self._transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, torch.Tensor):
            return image, torch.multiply(mask.float(), 1. / 255.).permute((2, 0, 1))
        else:
            return image, torch.multiply(torch.from_numpy(mask).type(torch.float32), 1. / 255.).permute((2, 0, 1))
    
    def _get_image_and_mask(self, idx):
        image_filename = self._images[idx] + '.png'
        mask_filename = self._images[idx] + '_id_mask.png'
        img_path = os.path.join(self._image_dir, image_filename)
        mask_path = os.path.join(self._mask_dir, mask_filename)
        
        # print(f"Trying to load image: {img_path}")
        # print(f"Trying to load mask: {mask_path}")
        
        # Check if files exist
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        num_classes = len(self._classes)

        one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)

        class_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]

        for idx, class_id in enumerate(class_ids):
            one_hot_mask[:, :, idx] = (mask == class_id) * 255

        return image, one_hot_mask
