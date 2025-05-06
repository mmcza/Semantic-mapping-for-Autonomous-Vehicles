import albumentations as A
import albumentations.pytorch.transforms

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda'  # 'cuda' or 'cpu'
NUM_WORKERS = 4
LOSS_FN = 'DiceFocalLoss'  # Options: 'FocalLoss', 'DiceFocalLoss', 'DiceLoss'

# Data parameters
DATA_DIR = '/root/Shared/annotations2/images/'  # Path to the directory with images
MASKS_DIR = '/root/Shared/annotations2/masks/'  # Path to the directory with mask annotations
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]  # Split ratio for train:val:test
RANDOM_STATE = 42  # Random seed for reproducibility
IMG_SIZE = (608, 960) # Height, Width
CLASSES = ['other', 'sky', 'building_fence', 'grass', 'sand_mud', 'road_pavement', 'tree', 'street_furniture', 'vehicle', 'person']
ORIGINAL_IMG_SIZE = (600, 960)

# Loss parameters
FOCAL_LOSS_WEIGHTS = [0.1551, 0.0918, 0.0603, 1.5754, 0.1273, 0.0362, 0.2105, 1.9278, 0.2893, 5.5263]
FOCAL_LOSS_GAMMA = 2.0

# Model parameters
ENCODER_NAME = 'resnet18'  # Encoder backbone for the segmentation model
ENCODER_WEIGHTS = 'imagenet'  # Pre-trained weights for encoder
MODEL_NAME = 'DeepLabV3Plus'  # Model architecture (e.g., 'Unet', 'DeepLabV3Plus')

# Scheduler parameters
SCHEDULER_TYPE = 'CosineAnnealingLR'  # Learning rate scheduler type


# Augmentation parameters
AUGMENTATIONS = {
    'train': A.Compose([
        A.PadIfNeeded(
            min_height=IMG_SIZE[0],
            min_width=IMG_SIZE[1],
            position='bottom_right',
            border_mode=0,
            value=0
        ),
        A.RandomGamma(gamma_limit=(80, 120)),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-5, 5), translate_px=(-10, 10), scale=(0.9, 1.1)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ]),
    'val': A.Compose([
        A.PadIfNeeded(
            min_height=IMG_SIZE[0],
            min_width=IMG_SIZE[1],
            position='bottom_right',
            border_mode=0,
            value=0
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ]),
    'test': A.Compose([
        A.PadIfNeeded(
            min_height=IMG_SIZE[0],
            min_width=IMG_SIZE[1],
            position='bottom_right',
            border_mode=0,
            value=0
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
}

# Other parameters
CHECKPOINT_DIR = './checkpoints/'  # Directory to save model checkpoints
LOG_DIR = './logs/'  # Directory to save logs
MODEL_SAVE_PATH = './model.pth'  # Path to save the final model
LOGGER = True
EXPORT_TO_ONNX = True
PROJECT_NAME = 'car-semantic-segmentation-mapping'
