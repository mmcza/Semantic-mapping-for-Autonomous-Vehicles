import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU, Fscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

wandb.login(key = 'key')

# --- Configuration ---
IMAGE_DIR = "path/to/images/"  
MASK_DIR = "path/to/masks/" 
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 11
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
TRAIN_SPLIT_RATIO = 0.8
DATASET_SUBSAMPLE_RATIO = 1.0
WANDB_PROJECT = "seg_sem_veh1"
WANDB_RUN_NAME = "Unet-efficientnetb0-10"

# Define color to index mapping
COLOR_MAP = {
    (0, 0, 0): 0,
    (135, 206, 235): 1,
    (70, 70, 70): 2,
    (0, 128, 0): 3,
    (210, 180, 140): 4,
    (128, 128, 128): 5,
    (139, 69, 19): 6,
    (34, 139, 34): 7,
    (255, 215, 0): 8,
    (255, 0, 0): 9,
    (255, 192, 203): 10
}

# Class distribution percentages (for weight computation)
CLASS_PERCENTAGES = {
    1: 0.15,
    2: 0.20,
    3: 0.02,
    4: 0.10,
    5: 0.30,
    6: 0.20,
    7: 0.08,
    8: 0.10,
    9: 0.05,
    10: 0.02
}

# Compute class weights (inverse frequency)
class_weights = torch.zeros(NUM_CLASSES)
for idx, pct in CLASS_PERCENTAGES.items():
    class_weights[idx] = 1.0 / (pct + 1e-6)
class_weights[1:] = class_weights[1:] / class_weights[1:].sum() * (NUM_CLASSES - 1)
class_weights[0] = 0.01  # small non-zero weight for background

# --- Helper Functions ---
def color_to_index_mask(mask_img, color_map):
    arr = np.array(mask_img.convert("RGB"))
    idx_mask = np.zeros(arr.shape[:2], dtype=np.int64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            idx_mask[i, j] = color_map.get(tuple(arr[i, j]), 0)
    return idx_mask


def index_to_color_mask(index_mask, color_map):
    rev = {v: k for k, v in color_map.items()}
    arr = np.zeros((*index_mask.shape, 3), dtype=np.uint8)
    for idx, color in rev.items():
        arr[index_mask == idx] = color
    return Image.fromarray(arr)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return {
            "allocated_MB": torch.cuda.memory_allocated() / 1024**2,
            "reserved_MB": torch.cuda.memory_reserved() / 1024**2
        }
    return {"allocated_MB": 0.0, "reserved_MB": 0.0}

# --- Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size, color_map, transform=None):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        assert len(self.image_paths) == len(self.mask_paths)
        self.image_size = image_size
        self.color_map = color_map
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB").resize(self.image_size)
        msk = Image.open(self.mask_paths[idx]).convert("RGB").resize(self.image_size, Image.NEAREST)
        m_idx = torch.from_numpy(color_to_index_mask(msk, self.color_map)).long()
        if self.transform:
            img = self.transform(img)
        return img, m_idx, os.path.basename(self.image_paths[idx])

# --- Transforms ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# --- Data Loading ---
all_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
all_masks = [os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png')]
if DATASET_SUBSAMPLE_RATIO < 1.0:
    imgs, _, msks, _ = train_test_split(all_images, all_masks,
                                       test_size=1-DATASET_SUBSAMPLE_RATIO,
                                       random_state=42)
else:
    imgs, msks = all_images, all_masks

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    imgs, msks, test_size=1-TRAIN_SPLIT_RATIO, random_state=42
)

train_ds = SegmentationDataset(train_imgs, train_masks, IMAGE_SIZE, COLOR_MAP, transform)
val_ds = SegmentationDataset(val_imgs, val_masks, IMAGE_SIZE, COLOR_MAP, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- Model ---
model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet",
                 in_channels=3, classes=NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Params: {count_parameters(model)}")
if torch.cuda.is_available():
    mem = get_gpu_memory_usage()
    print(f"GPU alloc: {mem['allocated_MB']:.2f} MB, resv: {mem['reserved_MB']:.2f} MB")

# --- Loss & Optimizer ---
ce_loss = CrossEntropyLoss(weight=class_weights.to(device))
dice_loss = DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def combined_loss(pred, true):
    return ce_loss(pred, true) + dice_loss(pred, true)

# --- WandB ---
wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE
})

# --- Training & Validation ---
best_val = float('inf')
num_viz = 3
viz_indices = random.sample(range(len(val_ds)), num_viz)
iou_metric = IoU()
fscore_metric = Fscore()


def calc_metric(metric_fn, pred, target):
    return metric_fn(pred.unsqueeze(0), target.unsqueeze(0)) if pred.numel() else 0.0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for imgs, msks, _ in tqdm(train_loader, desc=f"Train {epoch+1}"):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = combined_loss(out, msks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    wandb.log({"train_loss": train_loss}, step=epoch)

    # Validation
    model.eval()
    val_loss, tot_iou, tot_f = 0, 0, 0
    viz_data = []
    single_logged = False
    with torch.no_grad():
        for i, (imgs, msks, fns) in enumerate(tqdm(val_loader, desc=f"Val {epoch+1}")):
            imgs, msks = imgs.to(device), msks.to(device)

            # Measure single-frame inference time once
            if not single_logged:
                single_img = imgs[0].unsqueeze(0)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                _ = model(single_img)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start_time
                wandb.log({"inference_time_single_frame": elapsed}, step=epoch)
                single_logged = True

            # Batch forward
            out = model(imgs)
            val_loss += combined_loss(out, msks).item()
            preds = out.argmax(1)
            tot_iou += calc_metric(iou_metric, preds, msks).item()
            tot_f += calc_metric(fscore_metric, preds, msks).item()

            # Collect viz samples
            start = i * BATCH_SIZE
            for k in range(imgs.size(0)):
                if start + k in viz_indices:
                    viz_data.append((imgs[k].cpu(), msks[k].cpu(), preds[k].cpu(), fns[k]))

    # Log validation metrics
    val_loss /= len(val_loader)
    wandb.log({"val_loss": val_loss,
               "iou": tot_iou/len(val_loader),
               "fscore": tot_f/len(val_loader)}, step=epoch)

    # Log composite visualization
    for img, gt, pd, fn in viz_data:
        img_denorm = img.permute(1,2,0).numpy() * np.array(STD) + np.array(MEAN)
        img_denorm = np.clip(img_denorm, 0, 1)
        gt_color = np.array(index_to_color_mask(gt.numpy(), COLOR_MAP)) / 255.0
        pd_color = np.array(index_to_color_mask(pd.numpy(), COLOR_MAP)) / 255.0
        composite = np.concatenate([img_denorm, gt_color, pd_color], axis=1)
        wandb.log({
            f"viz/{epoch+1}/{fn}/composite": wandb.Image(
                composite, caption=f"{fn}: input | gt | pred"
            )
        }, step=epoch)

    # Save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")

wandb.finish()
