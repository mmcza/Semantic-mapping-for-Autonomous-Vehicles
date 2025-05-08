import os
import shutil
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import argparse

def load_folders(root_dir):
    return sorted([f for f in Path(root_dir).iterdir() if f.name.startswith("images_2025")])

def setup_window():
    window = tk.Tk()
    window.title("Leaf Annotation Cleanup Tool")
    window.geometry("1200x800")
    return window

def setup_frames(window):
    image_frame = ttk.Frame(window)
    image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    control_frame = ttk.Frame(window)
    control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

    return image_frame, control_frame

def setup_canvas(image_frame):
    canvas = tk.Canvas(image_frame, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    return canvas

def setup_labels(control_frame):
    vars = {
        "status": tk.StringVar(value="Ready"),
        "progress": tk.StringVar(value="0/0"),
        "deleted": tk.StringVar(value="Deleted: 0"),
        "path": tk.StringVar()
    }

    for label in ["status", "progress", "deleted", "path"]:
        ttk.Label(control_frame, textvariable=vars[label], wraplength=200).pack(pady=5)

    return vars

def setup_buttons(control_frame, actions):
    ttk.Button(control_frame, text="Keep & Next", command=actions['next']).pack(fill=tk.X, pady=5)
    ttk.Button(control_frame, text="Delete Image", command=actions['delete']).pack(fill=tk.X, pady=5)
    ttk.Button(control_frame, text="Exit", command=actions['exit']).pack(fill=tk.X, pady=20)

def resize_image(img, width):
    ratio = width / img.width
    height = int(img.height * ratio)
    return img.resize((width, height))

def move_to_removed(img_path, mask_path, backup_root):
    img_backup = backup_root / "images" / "default"
    mask_backup = backup_root / "visualizations"

    img_backup.mkdir(parents=True, exist_ok=True)
    mask_backup.mkdir(parents=True, exist_ok=True)

    shutil.move(str(img_path), str(img_backup / img_path.name))
    shutil.move(str(mask_path), str(mask_backup / mask_path.name))

def annotation_cleanup_tool(root_dir):
    folders = load_folders(root_dir)
    if not folders:
        print("No folders found.")
        return

    window = setup_window()
    image_frame, control_frame = setup_frames(window)
    canvas = setup_canvas(image_frame)
    vars = setup_labels(control_frame)

    current = {
        "folder_idx": 0,
        "image_idx": 0,
        "img_paths": [],
        "mask_paths": [],
        "folder": None
    }
    deleted = set()
    processed = set()

    def load_folder():
        if current["folder_idx"] >= len(folders):
            vars["status"].set("All folders processed.")
            return False

        folder = folders[current["folder_idx"]]
        current["folder"] = folder
        img_dir = folder / "images" / "default"
        mask_dir = folder / "visualizations"

        if not img_dir.exists() or not mask_dir.exists():
            current["folder_idx"] += 1
            return load_folder()

        current["img_paths"] = sorted(img_dir.glob("*_image_raw_*.png"))
        current["mask_paths"] = sorted(mask_dir.glob("*_segmentation.png"))
        current["image_idx"] = 0

        if not current["img_paths"]:
            current["folder_idx"] += 1
            return load_folder()

        vars["status"].set(f"Loaded: {folder.name}")
        return True

    def show_image():
        idx = current["image_idx"]
        if idx >= len(current["img_paths"]):
            current["folder_idx"] += 1
            if not load_folder():
                return
            idx = 0

        img_path = current["img_paths"][idx]
        mask_path = current["mask_paths"][idx]

        if str(img_path) in processed:
            current["image_idx"] += 1
            return show_image()

        img = Image.open(img_path).convert("RGB")
        img = resize_image(img, 800)

        photo = ImageTk.PhotoImage(img)
        canvas.config(width=img.width, height=img.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

        vars["progress"].set(f"{idx + 1}/{len(current['img_paths'])}")
        vars["path"].set(f"{img_path.name}")

    def next_image():
        processed.add(str(current["img_paths"][current["image_idx"]]))
        current["image_idx"] += 1
        show_image()

    def delete_image():
        idx = current["image_idx"]
        img_path = current["img_paths"][idx]
        mask_path = current["mask_paths"][idx]

        move_to_removed(img_path, mask_path, Path(root_dir) / "removed" / current["folder"].name)
        deleted.add(str(img_path))

        current["img_paths"].pop(idx)
        current["mask_paths"].pop(idx)

        vars["deleted"].set(f"Deleted: {len(deleted)}")
        vars["status"].set(f"Deleted: {img_path.name}")
        show_image()

    setup_buttons(control_frame, {
        "next": next_image,
        "delete": delete_image,
        "exit": window.destroy
    })

    if load_folder():
        show_image()

    window.mainloop()
    print(f"Finished. Processed: {len(processed)}, Deleted: {len(deleted)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default= "/root/Shared/annotations2/")
    return parser.parse_args()

def main():
    args = parse_args()
    annotation_cleanup_tool(args.root_dir)
