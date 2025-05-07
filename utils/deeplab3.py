import os
import argparse
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
from pathlib import Path

class CacheCleanCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("✓ Cache cleared")

    def on_validation_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

num_classes = 12
batch_size = 2
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
    [0, 0, 0], [128, 64, 128], [70, 70, 70], [153, 153, 153], [107, 142, 35],
    [70, 130, 180], [220, 20, 60], [0, 0, 142], [0, 60, 100], [0, 80, 100],
    [0, 0, 70], [0, 0, 230]
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

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
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
        one_hot = torch.nn.functional.one_hot(y.to(device), num_classes).permute(0, 3, 1, 2).float()
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

    def configure_optimizers(self, optimizer_type='adamw', scheduler_type='reduce_lr_plateau'):
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        if scheduler_type == 'reduce_lr_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=3
            )
        elif scheduler_type == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=learning_rate, total_steps=epochs
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}

def prepare_datasets(root_path, transform):
    datasets = []

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
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    gt = gt_mask.cpu().numpy()
    pred = pred_mask.cpu().numpy()
    h, w = gt.shape
    colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
    colored_pred = np.zeros_like(colored_gt)
    for i in range(num_classes):
        colored_gt[gt == i] = class_colors[i]
        colored_pred[pred == i] = class_colors[i]
    plt.figure(figsize=(15, 5))
    for i, (arr, title) in enumerate([(img, 'image'), (colored_gt, 'ground truth'), (colored_pred, 'prediction')], 1):
        plt.subplot(1, 3, i)
        plt.imshow(arr)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pred_{idx}.png')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='reduce_lr_plateau', choices=['reduce_lr_plateau', 'one_cycle'])
    parser.add_argument('--root_dir', type=str, default="/root/Shared/annotations2/"")
    return parser.parse_args()

def main():
    args = parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    optimizer_type = args.optimizer
    scheduler_type = args.scheduler
    root_dir = args.root_dir

    print(f"Training for {epochs} epochs with batch size {batch_size}, learning rate {learning_rate}, optimizer {optimizer_type}, scheduler {scheduler_type}")

    transform = create_transform()
    root_dir =  "/root/Shared/annotations2/"
    full_dataset = prepare_datasets(root_dir, transform)

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

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
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SegModel().to(device)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1 if use_cuda else None,
        precision=32,
        log_every_n_steps=10,
        callbacks=[CacheCleanCallback()]
    )

    trainer.fit(model, train_loader, val_loader)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Final cache cleared")

if __name__ == "__main__":
    main()
