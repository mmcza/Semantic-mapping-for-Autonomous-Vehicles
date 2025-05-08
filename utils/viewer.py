import os
from pathlib import Path
from tkinter import Tk, Canvas, StringVar, NW
from tkinter import ttk
from PIL import Image, ImageTk


CLASS_NAMES = [
    "Other", "Sky", "Building", "Grass", "Sand/Mud", "Road/Pavement",
    "Fence", "Tree", "Street Furniture", "Vehicle", "Person"
]

CLASS_COLORS = [
    [0, 0, 0], [135, 206, 235], [70, 70, 70], [0, 128, 0],
    [210, 180, 140], [128, 128, 128], [139, 69, 19], [34, 139, 34], 
    [255, 215, 0], [255, 0, 0], [255, 192, 203]
]

def collect_image_pairs(root_dir):
    annotations_dir = Path(root_dir) / "annotations2"
    pairs = {}

    for subdir in annotations_dir.iterdir():
        vis_dir = subdir / "visualizations"
        img_dir = subdir / "images"/ "default"
        if vis_dir.exists() and img_dir.exists():
            for seg_path in vis_dir.glob("*_segmentation.png"):
            # Extract the base part of the filename (before _segmentation.png)
                base_name = seg_path.stem.replace("_segmentation", "")
            # Look for corresponding raw image
                raw_path = img_dir / f"{base_name}.png"
                if raw_path.exists():
                    pairs[seg_path] = raw_path

        return pairs

def display_images(canvas, seg_path, raw_path, status_var):
    try:
        raw_img = Image.open(raw_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('RGB')
    except Exception as e:
        status_var.set(f"Error loading images: {e}")
        return False

    raw_photo = ImageTk.PhotoImage(raw_img)
    seg_photo = ImageTk.PhotoImage(seg_img)

    canvas.config(width=max(raw_img.width, seg_img.width),
                  height=raw_img.height + seg_img.height)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=NW, image=raw_photo)
    canvas.create_image(0, raw_img.height, anchor=NW, image=seg_photo)

    canvas.raw_photo = raw_photo
    canvas.seg_photo = seg_photo

    return True

def create_legend(frame):
    ttk.Label(frame, text="Class Legend:").pack(pady=5)
    legend_frame = ttk.Frame(frame)
    legend_frame.pack(pady=5)

    for i, (color, name) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        color_box = Canvas(legend_frame, width=20, height=20, bg=hex_color, highlightthickness=1)
        color_box.grid(row=i, column=0, padx=5, pady=2)
        ttk.Label(legend_frame, text=name).grid(row=i, column=1, sticky="w")

def annotation_viewer(root_dir):
    image_pairs = collect_image_pairs(root_dir)
    seg_files = list(image_pairs.keys())
    raw_files = [image_pairs[s] for s in seg_files]

    if not seg_files:
        print("No image pairs found.")
        return

    window = Tk()
    window.title("Segmentation Viewer")
    window.geometry("1600x1000")

    canvas = Canvas(window, bg="black")
    canvas.pack(side="left", fill="both", expand=True)

    control = ttk.Frame(window, width=250)
    control.pack(side="right", fill="y", padx=10, pady=10)

    status_var = StringVar(value="Ready")
    progress_var = StringVar()
    file_var = StringVar()

    ttk.Label(control, textvariable=status_var, wraplength=200).pack(pady=5)
    ttk.Label(control, textvariable=progress_var).pack(pady=5)
    ttk.Label(control, textvariable=file_var, wraplength=200).pack(pady=5)

    create_legend(control)

    idx = [0]

    def load(i):
        if 0 <= i < len(seg_files):
            success = display_images(canvas, seg_files[i], raw_files[i], status_var)
            if success:
                idx[0] = i
                progress_var.set(f"{i + 1} / {len(seg_files)}")
                file_var.set(raw_files[i].name)

    def next_image():
        if idx[0] + 1 < len(seg_files):
            load(idx[0] + 1)

    def prev_image():
        if idx[0] > 0:
            load(idx[0] - 1)

    def delete_current():
        i = idx[0]
        if i >= len(seg_files):
            return

        seg = seg_files[i]
        raw = raw_files[i]

        try:
            os.remove(seg)
            os.remove(raw)
            status_var.set(f"Deleted: {seg.name} and {raw.name}")
        except Exception as e:
            status_var.set(f"Error deleting: {e}")
            return

        del seg_files[i]
        del raw_files[i]

        if i >= len(seg_files):
            idx[0] = max(0, len(seg_files) - 1)

        if seg_files:
            load(idx[0])
        else:
            status_var.set("No images left")

    ttk.Button(control, text="Previous", command=prev_image).pack(fill="x", pady=5)
    ttk.Button(control, text="Next", command=next_image).pack(fill="x", pady=5)
    ttk.Button(control, text="Delete Files", command=delete_current).pack(fill="x", pady=5)
    ttk.Button(control, text="Exit", command=window.destroy).pack(fill="x", pady=20)

    load(0)
    window.mainloop()

annotation_viewer("/home/daria/Downloads/Semantic-mapping-for-Autonomous-Vehicles-main")


