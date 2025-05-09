import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Patch

def browse_skip(
    image_dir, 
    mask_dir, 
    class_names, 
    palette, 
    step=10
):
    """
    Display one image and mask side by side, with legend on the right.
    Use 'a' / 'd' keys to skip backward/forward by `step` images.
    """
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    total = len(files)
    idx = [0]

    # Enlarged figure
    fig = plt.figure(figsize=(20, 12))
    plt.subplots_adjust(bottom=0.1)

    def show_image():
        fig.clear()
        fname = files[idx[0]]
        base = os.path.splitext(fname)[0]

        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        # find mask by prefix
        cands = [m for m in os.listdir(mask_dir) if m.startswith(base)]
        mask = (Image.open(os.path.join(mask_dir, cands[0])).convert("RGB")
                if cands else Image.new("RGB", img.size, (200,200,200)))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img); ax1.set_title("Original"); ax1.axis("off")
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(mask); ax2.set_title("Mask"); ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        patches = [
            Patch(color=np.array(palette[i]) / 255.0, label=class_names[i])
            for i in range(len(class_names))
        ]
        ax3.legend(handles=patches, loc='center', frameon=False, fontsize='small')
        ax3.axis("off")

        fig.suptitle(f"{idx[0]+1}/{total}: {fname}", fontsize=16)
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'd':
            idx[0] = min(idx[0] + step, total - 1)
            show_image()
        elif event.key == 'a':
            idx[0] = max(idx[0] - step, 0)
            show_image()

    # Connect key handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    show_image()
    plt.show()


# Example usage:
class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
palette = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

browse_skip(
    image_dir="/images",
    mask_dir="/segmentation_hf_b5",
    class_names=class_names,
    palette=palette,
    step=10
)
