

#based on  https://github.com/XuJiacong/PIDNet

import os
import gc
import numpy as np
from pathlib import Path
from PIL import Image
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from albumentations import Compose, Resize, PadIfNeeded, Normalize
from albumentations.pytorch import ToTensorV2

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

class_names = [
    'road','sidewalk','building','wall','fence','pole',
    'traffic light','traffic sign','vegetation','terrain','sky',
    'person','rider','car','truck','bus','train','motorcycle','bicycle'
]
palette = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),
    (0,80,100),(0,0,230),(119,11,32)
]
value_mapping = {
     8:0, 16:1, 26:2, 46:3, 47:3, 58:4, 70:0, 76:5, 84:6, 90:7,
     95:5, 96:11,108:8,118:9,119:10,120:10,128:2,153:12,164:13,
    178:14,184:7,188:8,195:15,202:9,210:16,212:10
}

IMG_H, IMG_W = 608, 960
BATCH_SIZE    = 2
EPOCHS        = 30
LR            = 1e-3
NUM_CLASSES   = len(class_names)

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc   = False  


def create_transform():
    return Compose([
        Resize(IMG_H, IMG_W),
        PadIfNeeded(IMG_H, IMG_W),
        Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def map_mask(raw, mapping):
    out = np.zeros_like(raw, dtype=np.uint8)
    for v in np.unique(raw):
        out[raw == v] = mapping.get(v, 0)
    return out

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.imgs = sorted(Path(img_dir).glob("*_image_raw_*.png"))
        self.msks = sorted(Path(mask_dir).glob("*_segmented_mask.png"))
        self.tf   = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = np.array(Image.open(self.imgs[i]).convert("RGB"))
        raw = np.array(Image.open(self.msks[i]).convert("L"))
        mask = map_mask(raw, value_mapping)
        if self.tf:
            aug = self.tf(image=img, mask=mask)
            return aug['image'], aug['mask'].long()
        img_t = torch.from_numpy(img).permute(2,0,1).float()
        return img_t, torch.from_numpy(mask).long()

def prepare_datasets(root):
    all_ds = []
    for sub in sorted(Path(root).iterdir()):
        if sub.is_dir() and sub.name.startswith("images_2025"):
            img_dir  = sub/"images"/"default"
            mask_dir = sub/"visualizations"
            if img_dir.exists() and mask_dir.exists():
                all_ds.append(SegDataset(img_dir, mask_dir, create_transform()))
    if not all_ds:
        raise RuntimeError(f"No data found in {root}")
    return ConcatDataset(all_ds)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1   = BatchNorm2d(planes, momentum=bn_mom)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        return out if self.no_relu else self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2   = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False)
        self.bn3   = BatchNorm2d(planes*self.expansion, momentum=bn_mom)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu    = no_relu

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        return out if self.no_relu else self.relu(out)

class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale=None):
        super().__init__()
        self.bn1   = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2   = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, bias=True)
        self.scale = scale

    def forward(self, x):
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.conv1(x)))
        x = self.conv2(x)
        if self.scale:
            h, w = x.shape[-2]*self.scale, x.shape[-1]*self.scale
            x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=algc)
        return x

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BN=nn.BatchNorm2d):
        super().__init__()
        m = bn_mom
        self.scale0 = nn.Sequential(BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale1 = nn.Sequential(nn.AvgPool2d(5,2,2), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(9,4,4), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(17,8,8), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.process1 = nn.Sequential(BN(branch_planes,m), nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes,branch_planes,3,1,1,bias=False))
        self.process2 = nn.Sequential(BN(branch_planes,m), nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes,branch_planes,3,1,1,bias=False))
        self.process3 = nn.Sequential(BN(branch_planes,m), nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes,branch_planes,3,1,1,bias=False))
        self.process4 = nn.Sequential(BN(branch_planes,m), nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_planes,branch_planes,3,1,1,bias=False))
        self.compression = nn.Sequential(BN(branch_planes*5,m), nn.ReLU(inplace=True),
                                         nn.Conv2d(branch_planes*5,outplanes,1,bias=False))
        self.shortcut    = nn.Sequential(BN(inplanes,m), nn.ReLU(inplace=True),
                                         nn.Conv2d(inplanes,outplanes,1,bias=False))

    def forward(self, x):
        h, w = x.shape[-2:]
        s0 = self.scale0(x)
        p1 = self.process1(F.interpolate(self.scale1(x),(h,w),mode='bilinear',align_corners=algc) + s0)
        p2 = self.process2(F.interpolate(self.scale2(x),(h,w),mode='bilinear',align_corners=algc) + p1)
        p3 = self.process3(F.interpolate(self.scale3(x),(h,w),mode='bilinear',align_corners=algc) + p2)
        p4 = self.process4(F.interpolate(self.scale4(x),(h,w),mode='bilinear',align_corners=algc) + p3)
        cat = torch.cat([s0,p1,p2,p3,p4], dim=1)
        return self.compression(cat) + self.shortcut(x)

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BN=nn.BatchNorm2d):
        super().__init__()
        m = bn_mom
        self.scale0 = nn.Sequential(BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale1 = nn.Sequential(nn.AvgPool2d(5,2,2), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(9,4,4), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(17,8,8),BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), BN(inplanes,m), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes,branch_planes,1,bias=False))
        self.scale_proc = nn.Sequential(BN(branch_planes*4,m), nn.ReLU(inplace=True),
                                        nn.Conv2d(branch_planes*4,branch_planes*4,3,padding=1,groups=4,bias=False))
        self.compression = nn.Sequential(BN(branch_planes*5,m), nn.ReLU(inplace=True),
                                         nn.Conv2d(branch_planes*5,outplanes,1,bias=False))
        self.shortcut    = nn.Sequential(BN(inplanes,m), nn.ReLU(inplace=True),
                                         nn.Conv2d(inplanes,outplanes,1,bias=False))

    def forward(self, x):
        h,w = x.shape[-2:]
        s0 = self.scale0(x)
        l = []
        for scale in [self.scale1, self.scale2, self.scale3, self.scale4]:
            l.append(F.interpolate(scale(x),(h,w),mode='bilinear',align_corners=algc) + s0)
        proc = self.scale_proc(torch.cat(l, dim=1))
        cat  = torch.cat([s0, proc], dim=1)
        return self.compression(cat) + self.shortcut(x)

class PagFM(nn.Module):
    def __init__(self, in_c, mid_c, after_relu=False, with_channel=False, BN=nn.BatchNorm2d):
        super().__init__()
        self.after_relu   = after_relu
        self.with_channel = with_channel
        self.f_x = nn.Sequential(nn.Conv2d(in_c,mid_c,1,bias=False), BN(mid_c,momentum=bn_mom))
        self.f_y = nn.Sequential(nn.Conv2d(in_c,mid_c,1,bias=False), BN(mid_c,momentum=bn_mom))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_c,in_c,1,bias=False), BN(in_c,momentum=bn_mom))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        if self.after_relu:
            x, y = self.relu(x), self.relu(y)
        yq = self.f_y(y)
        yq = F.interpolate(yq, size=x.shape[-2:], mode='bilinear', align_corners=False)
        xk = self.f_x(x)
        if self.with_channel:
            sim = torch.sigmoid(self.up(xk * yq))
        else:
            sim = torch.sigmoid((xk * yq).sum(dim=1, keepdim=True))
        y_up = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return (1-sim) * x + sim * y_up

class Light_Bag(nn.Module):
    def __init__(self, in_c, out_c, BN=nn.BatchNorm2d):
        super().__init__()
        self.p = nn.Sequential(nn.Conv2d(in_c,out_c,1,bias=False), BN(out_c,momentum=bn_mom))
        self.i = nn.Sequential(nn.Conv2d(in_c,out_c,1,bias=False), BN(out_c,momentum=bn_mom))

    def forward(self, p, i, d):
        att = torch.sigmoid(d)
        p2 = self.p((1-att)*i + p)
        i2 = self.i(i + att*p)
        return p2 + i2

class Bag(nn.Module):
    def __init__(self, in_c, out_c, BN=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(BN(in_c), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_c,out_c,3,padding=1,bias=False))

    def forward(self, p, i, d):
        att = torch.sigmoid(d)
        return self.conv(att*p + (1-att)*i)

class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm=96, head_planes=128, augment=True):
        super().__init__()
        self.augment = augment
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes,3,2,1), BatchNorm2d(planes,momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes,3,2,1), BatchNorm2d(planes,momentum=bn_mom), nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes*2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes*2, planes*4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes*4, planes*8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes*8, planes*8, 2, stride=2)

        self.c3 = nn.Sequential(nn.Conv2d(planes*4,planes*2,1,bias=False), BatchNorm2d(planes*2,momentum=bn_mom))
        self.c4 = nn.Sequential(nn.Conv2d(planes*8,planes*2,1,bias=False), BatchNorm2d(planes*2,momentum=bn_mom))

        self.pag3 = PagFM(planes*2, planes)
        self.pag4 = PagFM(planes*2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes*2, planes*2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes*2, planes*2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes*2, planes*2,1)

        if m==2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes*2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes,1)
            self.diff3  = nn.Sequential(nn.Conv2d(planes*4,planes,3,1,1,bias=False), BatchNorm2d(planes,momentum=bn_mom))
            self.diff4  = nn.Sequential(nn.Conv2d(planes*8,planes*2,3,1,1,bias=False),BatchNorm2d(planes*2,momentum=bn_mom))
            self.spp    = PAPPM(planes*16, ppm, planes*4)
            self.dfm    = Light_Bag(planes*4, planes*4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes*2, planes*2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes*2, planes*2)
            self.diff3  = nn.Sequential(nn.Conv2d(planes*4,planes*2,3,1,1,bias=False), BatchNorm2d(planes*2,momentum=bn_mom))
            self.diff4  = nn.Sequential(nn.Conv2d(planes*8,planes*2,3,1,1,bias=False), BatchNorm2d(planes*2,momentum=bn_mom))
            self.spp    = DAPPM(planes*16, ppm, planes*4)
            self.dfm    = Bag(planes*4, planes*4)

        self.layer5_d = self._make_layer(Bottleneck, planes*2, planes*2, 1)

        if self.augment:
            self.seg_p = segmenthead(planes*2, head_planes, num_classes)
            self.seg_d = segmenthead(planes*2, planes, 1)

        self.final = segmenthead(planes*4, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inp, planes, blocks, stride=1):
        down = None
        if stride!=1 or inp!=planes*block.expansion:
            down = nn.Sequential(
                nn.Conv2d(inp, planes*block.expansion,1,stride,bias=False),
                BatchNorm2d(planes*block.expansion, momentum=bn_mom)
            )
        layers = [block(inp, planes, stride, down)]
        inp = planes * block.expansion
        for i in range(1, blocks):
            no_relu = (i == blocks-1)
            layers.append(block(inp, planes, 1, None, no_relu))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inp, planes, stride=1):
        down = None
        if stride!=1 or inp!=planes*block.expansion:
            down = nn.Sequential(
                nn.Conv2d(inp, planes*block.expansion,1,stride,bias=False),
                BatchNorm2d(planes*block.expansion, momentum=bn_mom)
            )
        return block(inp, planes, stride, down, True)

    def forward(self, x):
        h8, w8 = x.shape[-2]//8, x.shape[-1]//8
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        x_p = self.layer3_(x)
        x_d = self.layer3_d(x)
        x   = self.relu(self.layer3(x))
        x_p = self.pag3(x_p, self.c3(x))
        x_d = x_d + F.interpolate(self.diff3(x),(h8,w8),mode='bilinear',align_corners=algc)
        if self.augment: tmp_p = x_p

        x = self.relu(self.layer4(x))
        x_p = self.layer4_(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))
        x_p = self.pag4(x_p, self.c4(x))
        x_d = x_d + F.interpolate(self.diff4(x),(h8,w8),mode='bilinear',align_corners=algc)
        if self.augment: tmp_d = x_d

        x_p = self.layer5_(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))
        x   = F.interpolate(self.spp(self.layer5(x)),(h8,w8),mode='bilinear',align_corners=algc)

        out = self.final(self.dfm(x_p, x, x_d))
        if self.augment:
            return [self.seg_p(tmp_p), out, self.seg_d(tmp_d)]
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.s = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets, NUM_CLASSES).permute(0,3,1,2).float()
        p = probs.contiguous().view(-1)
        g = oh.contiguous().view(-1)
        inter = (p * g).sum()
        return 1 - (2*inter + self.s) / (p.sum() + g.sum() + self.s)

def compute_iou(preds, targs):
    p = preds.view(-1)
    t = targs.view(-1)
    ious = []
    for c in range(NUM_CLASSES):
        pi = p == c
        ti = t == c
        inter = (pi & ti).sum().item()
        uni   = pi.sum().item() + ti.sum().item() - inter
        if uni > 0:
            ious.append(inter / uni)
    return torch.tensor(sum(ious) / len(ious)) if ious else torch.tensor(0.0)

def compute_acc(preds, targs):
    return (preds == targs).float().mean()


def visualize_prediction(img_t, gt_t, pr_t, idx=0, out_dir="predictions"):
    os.makedirs(out_dir, exist_ok=True)
    img = img_t.permute(1,2,0).cpu().numpy()
    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])) * 255
    img = img.astype(np.uint8)
    gt = gt_t.cpu().numpy()
    pr = pr_t.cpu().numpy()
    h, w = gt.shape
    gt_c = np.zeros((h,w,3), np.uint8)
    pr_c = np.zeros((h,w,3), np.uint8)
    for c in range(NUM_CLASSES):
        gt_c[gt==c] = palette[c]
        pr_c[pr==c] = palette[c]
    plt.figure(figsize=(12,4))
    for i,(arr,ttl) in enumerate([(img,"Image"),(gt_c,"GT"),(pr_c,"Pred")],1):
        plt.subplot(1,3,i)
        plt.imshow(arr)
        plt.title(ttl)
        plt.axis("off")
    patches = [mpatches.Patch(color=np.array(c)/255.0, label=class_names[i])
               for i,c in enumerate(palette)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pred_{idx}.png")
    plt.close()


class SegModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PIDNet(m=2, n=3, num_classes=NUM_CLASSES, planes=32, ppm=96, head_planes=128, augment=False)
        self.dice  = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        imgs, msks = batch
        logits = self(imgs)
        if logits.shape[-2:] != msks.shape[-2:]:
            logits = F.interpolate(logits, size=msks.shape[-2:], mode='bilinear', align_corners=False)
        ce_loss   = F.cross_entropy(logits, msks)
        dice_loss = self.dice(logits, msks)
        loss = 0.5 * ce_loss + 0.5 * dice_loss
        preds = torch.argmax(logits, dim=1)
        iou   = compute_iou(preds, msks)
        acc   = compute_acc(preds, msks)
        # log metrics
        self.log(f"{stage}_loss", loss,      prog_bar=True)
        self.log(f"{stage}_dice", dice_loss, prog_bar=True)
        self.log(f"{stage}_iou",  iou,       prog_bar=True)
        self.log(f"{stage}_acc",  acc,       prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        return {"optimizer":optimizer, "lr_scheduler":{"scheduler":scheduler, "monitor":"val_loss"}}



if __name__ == "__main__":
    data_root = "/home/daria/Downloads/Semantic-mapping-for-Autonomous-Vehicles-main/annotations2"

    full_ds = prepare_datasets(data_root)
    val_n   = int(0.2 * len(full_ds))
    tr_n    = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [tr_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SegModel()
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[CacheCleanCallback()]
    )


    trainer.fit(model, train_dl, val_dl)

    # save checkpoint
    ckpt = "pidnet_segmentation.ckpt"
    trainer.save_checkpoint(ckpt)
    print(f"Model saved to {ckpt}")

    model.eval()
    imgs, msks = next(iter(val_dl))
    with torch.no_grad():
        preds = torch.argmax(model(imgs), dim=1)
    for i in range(min(4, len(preds))):
        visualize_prediction(imgs[i], msks[i], preds[i], idx=i)

    print("Sample predictions saved in 'predictions/'")
