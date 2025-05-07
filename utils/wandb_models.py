
import os
import random
import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb


wandb.login(key = 'KEY')

# Configuration
IMAGE_DIR = "/images/"
MASK_DIR = "/masks/"
IMAGE_SIZE = (480, 300)
NUM_CLASSES = 11
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
TRAIN_SPLIT_RATIO = 0.8
DATASET_SUBSAMPLE_RATIO = 1.0
WANDB_PROJECT = "seg_sem_veh1"
WANDB_RUN_NAME = "Unet-mobilenet_v2"
SEED = 24
     
# "efficientnet-b0"
# "resnet34"      
# "mobilenet_v2"  
# "swin_tiny"      
ENCODER_NAME = "mobilenet_v2"     
ENCODER_WEIGHTS = "imagenet"    

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Color map for masks
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

# Class distributions (for weight computation)
CLASS_PERCENTAGES = {
    1: 0.15, 2: 0.20, 3: 0.02, 4: 0.10,
    5: 0.30, 6: 0.20, 7: 0.08, 8: 0.10,
    9: 0.05, 10: 0.02
}

# Compute class weights
class_weights = torch.zeros(NUM_CLASSES)
for idx, pct in CLASS_PERCENTAGES.items():
    class_weights[idx] = 1.0 / (pct + 1e-6)
class_weights[1:] = class_weights[1:] / class_weights[1:].sum() * (NUM_CLASSES - 1)
class_weights[0] = 0.01

# Mask conversion helpers
def color_to_index_mask(mask_img, color_map):
    arr = np.array(mask_img.convert("RGB"))
    idx = np.zeros(arr.shape[:2], dtype=np.int64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            idx[i, j] = color_map.get(tuple(arr[i, j]), 0)
    return idx

def index_to_color_mask(idx_mask, color_map):
    rev = {v: k for k, v in color_map.items()}
    h, w = idx_mask.shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in rev.items():
        arr[idx_mask == c] = col
    return Image.fromarray(arr)

# Model utilities
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return {
            "allocated_MB": torch.cuda.memory_allocated() / 1024**2,
            "reserved_MB": torch.cuda.memory_reserved() / 1024**2
        }
    return {"allocated_MB": 0.0, "reserved_MB": 0.0}

# Dataset definition
class SegmentationDataset(Dataset):
    def __init__(self, imgs, msks, size, cmap, tfm=None):
        self.imgs = sorted(imgs)
        self.msks = sorted(msks)
        self.size = size
        self.cmap = cmap
        self.tfm = tfm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB").resize(self.size)
        msk = Image.open(self.msks[i]).convert("RGB").resize(self.size, Image.NEAREST)
        midx = torch.from_numpy(color_to_index_mask(msk, self.cmap)).long()
        img = self.tfm(img) if self.tfm else transforms.ToTensor()(img)
        return img, midx, os.path.basename(self.imgs[i])

# Data transforms
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Load file paths
all_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
all_masks  = [os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.png')]
if DATASET_SUBSAMPLE_RATIO < 1.0:
    imgs, _, msks, _ = train_test_split(
        all_images, all_masks,
        test_size=1 - DATASET_SUBSAMPLE_RATIO,
        random_state=SEED
    )
else:
    imgs, msks = all_images, all_masks

train_imgs, val_imgs, train_msks, val_msks = train_test_split(
    imgs, msks,
    test_size=1 - TRAIN_SPLIT_RATIO,
    random_state=SEED
)

train_ds = SegmentationDataset(train_imgs, train_msks, IMAGE_SIZE, COLOR_MAP, transform)
val_ds   = SegmentationDataset(val_imgs,   val_msks,   IMAGE_SIZE, COLOR_MAP, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model setup with parameterized encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name=ENCODER_NAME,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=NUM_CLASSES
).to(device)

model_size = count_parameters(model)
mem = get_gpu_memory_usage()

# Losses, optimizer, metrics
ce_loss_fn   = CrossEntropyLoss(weight=class_weights.to(device))
dice_loss_fn = DiceLoss(mode='multiclass')
optimizer    = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
iou_metric    = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)
fscore_metric = MulticlassF1Score(num_classes=NUM_CLASSES).to(device)

# WandB init
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_split_ratio": TRAIN_SPLIT_RATIO,
        "dataset_subsample_ratio": DATASET_SUBSAMPLE_RATIO,
        "model_size": model_size,
        "gpu_allocated_MB": mem["allocated_MB"],
        "gpu_reserved_MB": mem["reserved_MB"]
    }
)

# Training & validation loops with full metrics logging
global_step = 0
best_val = float('inf')
single_logged = False

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = train_ce = train_dice_loss = 0.0
    train_iou_total = train_f_total = 0.0

    for imgs, msks, _ in tqdm(train_loader, desc=f"Train {epoch+1}/{NUM_EPOCHS}"):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        ce = ce_loss_fn(out, msks)
        dloss = dice_loss_fn(out, msks)
        loss = ce + dloss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_ce += ce.item()
        train_dice_loss += dloss.item()
        preds = out.argmax(1)
        batch_iou = iou_metric(preds, msks).item()
        batch_f = fscore_metric(preds, msks).item()
        train_iou_total += batch_iou
        train_f_total += batch_f

        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_ce_loss": ce.item(),
            "train/batch_dice_loss": dloss.item(),
            "train/batch_iou": batch_iou,
            "train/batch_fscore": batch_f
        })
        global_step += 1

    train_loss /= len(train_loader)
    train_ce /= len(train_loader)
    train_dice_loss /= len(train_loader)
    train_iou = train_iou_total / len(train_loader)
    train_fscore = train_f_total / len(train_loader)
    wandb.log({
        "train/epoch_loss": train_loss,
        "train/epoch_ce_loss": train_ce,
        "train/epoch_dice_loss": train_dice_loss,
        "train_iou": train_iou,
        "train_fscore": train_fscore
    })

    # Validation
    model.eval()
    val_loss = val_ce = val_dice_loss = 0.0
    val_iou_total = val_f_total = 0.0

    with torch.no_grad():
        for imgs, msks, fns in tqdm(val_loader, desc=f"Val {epoch+1}/{NUM_EPOCHS}"):
            imgs, msks = imgs.to(device), msks.to(device)

            if not single_logged:
                t0 = time.time()
                _ = model(imgs[0].unsqueeze(0))
                if torch.cuda.is_available(): torch.cuda.synchronize()
                wandb.log({"val/single_image_inference_time": time.time() - t0})
                single_logged = True

            out = model(imgs)
            ce = ce_loss_fn(out, msks)
            dloss = dice_loss_fn(out, msks)
            loss = ce + dloss

            val_loss += loss.item()
            val_ce += ce.item()
            val_dice_loss += dloss.item()
            preds = out.argmax(1)
            batch_iou = iou_metric(preds, msks).item()
            batch_f = fscore_metric(preds, msks).item()
            val_iou_total += batch_iou
            val_f_total += batch_f

            wandb.log({
                "val/batch_loss": loss.item(),
                "val/batch_ce_loss": ce.item(),
                "val/batch_dice_loss": dloss.item(),
                "val/batch_iou": batch_iou,
                "val/batch_fscore": batch_f
            })
            global_step += 1

    val_loss /= len(val_loader)
    val_ce /= len(val_loader)
    val_dice_loss /= len(val_loader)
    val_iou = val_iou_total / len(val_loader)
    val_fscore = val_f_total / len(val_loader)
    wandb.log({
        "val/epoch_loss": val_loss,
        "val/epoch_ce_loss": val_ce,
        "val/epoch_dice_loss": val_dice_loss,
        "val_iou": val_iou,
        "val_fscore": val_fscore
    })

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")

# Final summary
wandb.run.summary["best_val_loss"]    = best_val
wandb.run.summary["final_train_loss"] = train_loss
wandb.run.summary["final_val_loss"]   = val_loss
wandb.run.summary["final_val_iou"]    = val_iou
wandb.run.summary["final_val_fscore"]= val_fscore

wandb.finish()

