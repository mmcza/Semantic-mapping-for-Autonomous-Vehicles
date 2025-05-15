import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import wandb
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score


# CONFIG
wandb.login(key = 'key')
IMAGE_DIR = "/images"
MASK_DIR = "/masks"
IMAGE_SIZE = (960, 608)
NUM_CLASSES = 19
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
SEED = 25
model_type_smp = "DL3PLUS"  #  "DL3PLUS" / "FPN" / "UNET"  (defult)
ENCODER_NAME = "efficientnet-b1"
ENCODER_WEIGHTS = "imagenet"
WANDB_PROJECT = "seg_sem_veh_b5"
APPENDIX_COMMENT = "-VAL_34"
WANDB_RUN_NAME = model_type_smp + ENCODER_NAME + APPENDIX_COMMENT
VAL_TS = "images_2025_04_14-17_59_20"
TEST_TS = "images_2025_04_14-18_02_03"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

COLOR_MAP = {
    (128,64,128):0,(244,35,232):1,(70,70,70):2,(102,102,156):3,
    (190,153,153):4,(153,153,153):5,(250,170,30):6,(220,220,0):7,
    (107,142,35):8,(152,251,152):9,(70,130,180):10,(220,20,60):11,
    (255,0,0):12,(0,0,142):13,(0,0,70):14,(0,60,100):15,
    (0,80,100):16,(0,0,230):17,(119,11,32):18
}
CLASS_WEIGHTS = {
    0:0.1,1:0.45,2:0.14,3:1.2,4:0.65,5:1.0,6:11.37,7:2.6,
    8:0.21,9:0.57,10:0.18,11:1.87,12:11.38,13:0.32,14:0.95,
    15:15.26,16:8.34,17:14.28,18:2.98
}

# HELPERS
def color_to_index_mask(mask_img, cmap):
    arr = np.array(mask_img.convert("RGB"))
    idx = np.zeros(arr.shape[:2], dtype=np.int64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            idx[i,j] = cmap.get(tuple(arr[i,j]),0)
    return idx

def index_to_color_mask(idx_mask, cmap):
    rev = {v:k for k,v in cmap.items()}
    h,w = idx_mask.shape
    arr = np.zeros((h,w,3), dtype=np.uint8)
    for c,col in rev.items():
        arr[idx_mask==c] = col
    return Image.fromarray(arr)

# DATALOADERS
class SegmentationDataset(Dataset):
    def __init__(self, imgs, msks, size, cmap, tfm=None):
        self.imgs = imgs
        self.msks = msks
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

all_imgs = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
]

def get_ts(path):
    return os.path.basename(path).split("__")[0]

groups = {}
for img in all_imgs:
    ts = get_ts(img)
    groups.setdefault(ts, []).append(img)


train_imgs, val_imgs, test_imgs = [], [], []
for ts, imgs in groups.items():
    if ts == VAL_TS:
        val_imgs.extend(imgs)
    elif ts == TEST_TS:
        test_imgs.extend(imgs)
    else:
        train_imgs.extend(imgs)

train_msks = [
    os.path.join(MASK_DIR, os.path.basename(img).replace(".png","_segmented_mask.png"))
    for img in train_imgs
]
val_msks = [
    os.path.join(MASK_DIR, os.path.basename(img).replace(".png","_segmented_mask.png"))
    for img in val_imgs
]
test_msks = [
    os.path.join(MASK_DIR, os.path.basename(img).replace(".png","_segmented_mask.png"))
    for img in test_imgs
]

train_ds = SegmentationDataset(train_imgs, train_msks, IMAGE_SIZE, COLOR_MAP, transform)
val_ds   = SegmentationDataset(val_imgs,   val_msks,   IMAGE_SIZE, COLOR_MAP, transform)
test_ds  = SegmentationDataset(test_imgs,  test_msks,  IMAGE_SIZE, COLOR_MAP, transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


#MODELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type_smp == "FPN":
    model = smp.FPN(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES
    ).to(device)
elif model_type_smp == "DL3PLUS":
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES
    ).to(device)
else:
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES
    ).to(device)

# LOSS / METRICS
weights = torch.tensor([CLASS_WEIGHTS[i] for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)
ce_loss_fn    = CrossEntropyLoss(weight=weights)
dice_loss_fn  = DiceLoss(mode="multiclass")
focal_loss_fn = FocalLoss(mode="multiclass", alpha=0.25)
optimizer     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
iou_metric    = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)
fscore_metric = MulticlassF1Score(num_classes=NUM_CLASSES).to(device)


# WANDB SETUP
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "encoder": ENCODER_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "num_classes": NUM_CLASSES,
        "image_size " : IMAGE_SIZE,
        "validation_video": VAL_TS,
        "test_video" : TEST_TS
    }
)

viz_indices = random.sample(range(len(val_ds)), k=5)
global_step = 0
best_val = float("inf")
single_logged = False
# TRAINING LOOP
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = train_ce = train_dice = train_focal = 0.0
    train_iou_total = train_f_total = 0.0

    for imgs_b, msks_b, _ in train_loader:
        imgs_b, msks_b = imgs_b.to(device), msks_b.to(device)
        optimizer.zero_grad()
        out = model(imgs_b)
        ce    = ce_loss_fn(out, msks_b)
        dloss = dice_loss_fn(out, msks_b)
        floss = focal_loss_fn(out, msks_b)
        loss  = ce + dloss + floss
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        biou  = iou_metric(preds, msks_b).item()
        bf    = fscore_metric(preds, msks_b).item()

        train_loss      += loss.item()
        train_ce        += ce.item()
        train_dice      += dloss.item()
        train_focal     += floss.item()
        train_iou_total += biou
        train_f_total   += bf

        wandb.log({
            "train/batch_loss":  loss.item(),
            "train/batch_ce":    ce.item(),
            "train/batch_dice":  dloss.item(),
            "train/batch_focal": floss.item(),
            "train/batch_iou":   biou,
            "train/batch_fscore":bf
        })
        global_step += 1

    wandb.log({
        "train/epoch_loss":  train_loss / len(train_loader),
        "train/epoch_ce":    train_ce   / len(train_loader),
        "train/epoch_dice":  train_dice / len(train_loader),
        "train/epoch_focal": train_focal/ len(train_loader),
        "train/epoch_iou":   train_iou_total / len(train_loader),
        "train/epoch_fscore":train_f_total   / len(train_loader)
    })

    model.eval()
    val_loss = val_ce = val_dice = val_focal = 0.0
    val_iou_total = val_f_total = 0.0
    batch_start = 0

    with torch.no_grad():
        for imgs_b, msks_b, fns in val_loader:
            imgs_b, msks_b = imgs_b.to(device), msks_b.to(device)

            if not single_logged:
                t0 = time.time()
                _ = model(imgs_b[0].unsqueeze(0))
                if torch.cuda.is_available(): torch.cuda.synchronize()
                wandb.log({"val/single_inference_time": time.time() - t0})
                single_logged = True

            out   = model(imgs_b)
            ce    = ce_loss_fn(out, msks_b)
            dloss = dice_loss_fn(out, msks_b)
            floss = focal_loss_fn(out, msks_b)
            loss  = ce + dloss + floss

            preds = out.argmax(1)
            biou  = iou_metric(preds, msks_b).item()
            bf    = fscore_metric(preds, msks_b).item()

            val_loss      += loss.item()
            val_ce        += ce.item()
            val_dice      += dloss.item()
            val_focal     += floss.item()
            val_iou_total += biou
            val_f_total   += bf

            wandb.log({
                "val/batch_loss":  loss.item(),
                "val/batch_ce":    ce.item(),
                "val/batch_dice":  dloss.item(),
                "val/batch_focal": floss.item(),
                "val/batch_iou":   biou,
                "val/batch_fscore":bf
            }, commit=False)

            for i in range(imgs_b.size(0)):
                idx = batch_start + i
                if idx in viz_indices:
                    img_np  = imgs_b[i].cpu().permute(1,2,0).numpy() * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
                    img_np  = np.clip(img_np,0,1)
                    gt_col  = np.array(index_to_color_mask(msks_b[i].cpu().numpy(), COLOR_MAP))/255.0
                    pd_col  = np.array(index_to_color_mask(preds[i].cpu().numpy(), COLOR_MAP))/255.0
                    wandb.log({
                        f"img/{epoch}/{fns[i]}":  wandb.Image(img_np),
                        f"gt/{epoch}/{fns[i]}":   wandb.Image(gt_col),
                        f"pred/{epoch}/{fns[i]}": wandb.Image(pd_col)
                    })
            batch_start += imgs_b.size(0)

    wandb.log({
        "val/epoch_loss":  val_loss      / len(val_loader),
        "val/epoch_ce":    val_ce        / len(val_loader),
        "val/epoch_dice":  val_dice      / len(val_loader),
        "val/epoch_focal": val_focal     / len(val_loader),
        "val/epoch_iou":   val_iou_total / len(val_loader),
        "val/epoch_fscore":val_f_total   / len(val_loader)
    })

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")

# SUMARRY
avg_train = train_loss / len(train_loader)
avg_val   = val_loss   / len(val_loader)
wandb.run.summary["best_val_loss"]     = best_val
wandb.run.summary["final_train_loss"]  = avg_train
wandb.run.summary["final_val_loss"]    = avg_val
wandb.finish()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
