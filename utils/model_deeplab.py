import os
from pathlib import Path
import gc

import numpy as np
import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, PadIfNeeded, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class CacheCleanCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Clear memory cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("✓ Cache cleared")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Clear cache after validation too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

num_classes = 12
batch_size = 12
epochs = 30
learning_rate = 1e-3
img_height = 608
img_width = 960
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

value_mapping = {
    0: 0, 70: 1, 75: 2, 76: 3, 84: 4, 95: 5,
    128: 2, 184: 7, 188: 8, 202: 9, 212: 10, 96: 11
}


num_classes = 12
batch_size = 12
epochs = 30
learning_rate = 1e-3
img_height = 608
img_width = 960
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

value_mapping = {
    0: 0, 70: 1, 75: 2, 76: 3, 84: 4, 95: 5,
    128: 2, 184: 7, 188: 8, 202: 9, 212: 10, 96: 11
}


class_colors = [
    [0, 0, 0],        
    [128, 64, 128],  
    [70, 70, 70],     
    [153, 153, 153], 
    [107, 142, 35],  
    [70, 130, 180],   
    [220, 20, 60],   
    [0, 0, 142],     
    [0, 60, 100],    
    [0, 80, 100],   
    [0, 0, 70],      
    [0, 0, 230],     
]


def create_transform():
    return Compose([
        Resize(height=img_height, width=img_width),
        PadIfNeeded(min_height=img_height, min_width=img_width),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def map_mask(mask_np, mapping):
    result = mask_np.copy()
    for old, new in mapping.items():
        result[mask_np == old] = new
    return result


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(Path(image_dir).glob('*_image_raw_*.png'))
        self.masks = sorted(Path(mask_dir).glob('*_segmentation.png'))
        assert len(self.images) == len(self.masks), "images and masks count must match"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        mask = np.array(Image.open(self.masks[idx]).convert('L'))
        mask = map_mask(mask, value_mapping)
        assert mask.min() >= 0 and mask.max() < num_classes, \
            f"mask values out of range: {mask.min()} to {mask.max()}"

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            return aug['image'], aug['mask'].long()
        img_tensor = torch.from_numpy(img).permute(2,0,1).float()
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor


class SegModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name='tu-resnet18',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self(x.to(device))
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(y.to(device), num_classes) \
                    .permute(0,3,1,2).float()
        intersection = (probs * one_hot).sum()
        union = probs.sum() + one_hot.sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        if stage == 'train':
            self.log(f"{stage}_loss", dice_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{stage}_dice", 1 - dice_loss, prog_bar=True, on_epoch=True)
        elif stage == 'val':
            self.log(f"{stage}_loss", dice_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_dice", 1 - dice_loss, prog_bar=True, on_step=False, on_epoch=True)

        return dice_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}


def prepare_datasets(root_path, transform):
    datasets = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    for folder in sorted(Path(root_path).iterdir()):
        if folder.name.startswith("images_2025"):
            img_dir = folder / "images" / "default"
            mask_dir = folder / "visualizations"
            if img_dir.exists() and mask_dir.exists():
                datasets.append(SegDataset(img_dir, mask_dir, transform))
    if not datasets:
        raise RuntimeError(f"no data found in {root_path}")
    return ConcatDataset(datasets)

def visualize_prediction(image, gt_mask, pred_mask, idx=0, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    img = image.permute(1,2,0).cpu().numpy()
    img = (img * np.array([0.229,0.224,0.225]) + \
           np.array([0.485,0.456,0.406])) * 255
    img = img.astype(np.uint8)
    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()
    h, w = gt.shape
    colored_gt = np.zeros((h,w,3), dtype=np.uint8)
    colored_pred = np.zeros_like(colored_gt)
    for i in range(num_classes):
        colored_gt[gt==i] = class_colors[i]
        colored_pred[pred==i] = class_colors[i]
    plt.figure(figsize=(15,5))
    for i, (arr, title) in enumerate([(img,'image'),(colored_gt,'ground truth'),(colored_pred,'prediction')],1):
        plt.subplot(1,3,i)
        plt.imshow(arr)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pred_{idx}.png')
    plt.close()


def main():
    if torch.cuda.is_available():
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()
    
    root_dir = "/root/Shared/annotations2/"
    transform = create_transform()
    full_dataset = prepare_datasets(root_dir, transform)
    
    # Create train-validation split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)  # 20% for validation
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create separate dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle validation data
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    model = SegModel().to(device)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1 if use_cuda else None,
        precision=32,
        log_every_n_steps=10,
        callbacks=[CacheCleanCallback()]
    )
    
    # Pass both loaders to fit
    trainer.fit(model, train_loader, val_loader)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Final cache cleared")

def main_vis():
    if torch.cuda.is_available():
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Path to the checkpoint
    checkpoint_path = "./lightning_logs/version_7/checkpoints/epoch=29-step=6810.ckpt"
    
    # Load data for visualization
    root_dir = "/root/Shared/annotations2/"
    transform = create_transform()
    full_dataset = prepare_datasets(root_dir, transform)
    
    # Create a dataloader for visualization (no need to split)
    vis_loader = DataLoader(
        full_dataset, 
        batch_size=4,  # Smaller batch for visualization
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    
    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}")
    model = SegModel().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Create predictions directory
    os.makedirs('predictions', exist_ok=True)
    
    # Generate and save visualizations
    with torch.no_grad():  # No gradient computation needed
        for i, (images, masks) in enumerate(vis_loader):
            print(f"Processing batch {i+1}/{len(vis_loader)}")
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # Generate visualizations for each image in the batch
            for j in range(images.shape[0]):
                idx = i * vis_loader.batch_size + j
                visualize_prediction(
                    images[j], 
                    masks[j], 
                    preds[j], 
                    idx=idx, 
                    save_dir='predictions'
                )
                
            # Only process first 10 batches to avoid generating too many images
            if i >= 9:
                break
    
    print(f"Visualizations saved to 'predictions/' directory")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Final cache cleared")

if __name__ == "__main__":
    main_vis()
