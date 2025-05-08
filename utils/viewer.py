import os
from pathlib import Path
from tkinter import Tk, Canvas, StringVar, NW, Toplevel, Frame, Button
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import shutil
from collections import deque
import argparse

CLASS_NAMES = [
    "Other", "Sky", "Building/Fence", "Grass", "Sand/Mud", "Road/Pavement",
    "Tree", "Street Furniture", "Vehicle", "Person"
]

CLASS_GRAY_COLORS = [
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11
]

CLASS_COLORS = [
    [0, 0, 0], [135, 206, 235], [70, 70, 70], [0, 128, 0],
    [210, 180, 140], [128, 128, 128], [34, 139, 34], 
    [255, 215, 0], [255, 0, 0], [255, 192, 203]
]

def collect_image_pairs(root_dir):
    annotations_dir = Path(root_dir)
    images_dir = annotations_dir / "images"
    masks_dir = annotations_dir / "masks"
    pairs = {}

    if images_dir.exists() and masks_dir.exists():
        for img_path in images_dir.glob("*.png"):
            # Extract the base name without extension
            base_name = img_path.stem
            # Look for corresponding mask
            mask_path = masks_dir / f"{base_name}_id_mask.png"
            if mask_path.exists():
                pairs[mask_path] = img_path

    return pairs

def display_images(canvas, mask_path, raw_path, status_var):
    try:
        raw_img = Image.open(raw_path).convert('RGB')
        # Open mask as grayscale
        mask_img_gray = Image.open(mask_path).convert('L')
        
        # Convert grayscale mask to RGB visualization
        mask_array = np.array(mask_img_gray)
        height, width = mask_array.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Map each grayscale value to its corresponding RGB color
        for i, gray_value in enumerate(CLASS_GRAY_COLORS):
            if i < len(CLASS_COLORS):
                mask = (mask_array == gray_value)
                colored_mask[mask] = CLASS_COLORS[i]
                
        mask_img = Image.fromarray(colored_mask)
        
        target_height = 450
        
        # Resize raw image
        raw_aspect_ratio = raw_img.width / raw_img.height
        raw_new_width = int(target_height * raw_aspect_ratio)
        raw_img_resized = raw_img.resize((raw_new_width, target_height), Image.LANCZOS)
        
        # Resize mask image
        mask_aspect_ratio = mask_img.width / mask_img.height
        mask_new_width = int(target_height * mask_aspect_ratio)
        mask_img_resized = mask_img.resize((mask_new_width, target_height), Image.NEAREST)
        
    except Exception as e:
        status_var.set(f"Error loading images: {e}")
        return False

    raw_photo = ImageTk.PhotoImage(raw_img_resized)
    mask_photo = ImageTk.PhotoImage(mask_img_resized)

    canvas.config(width=max(raw_new_width, mask_new_width),
                  height=target_height * 2)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=NW, image=raw_photo)
    canvas.create_image(0, target_height, anchor=NW, image=mask_photo)

    canvas.raw_photo = raw_photo
    canvas.mask_photo = mask_photo

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
    window.geometry("950x900")

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
            for child in window.winfo_children():
                if isinstance(child, Toplevel) and hasattr(child, 'save_mask'):
                    child.save_mask()
                    child.destroy()
            
            load(idx[0] + 1)

    def prev_image():
        if idx[0] > 0:
            for child in window.winfo_children():
                if isinstance(child, Toplevel) and hasattr(child, 'save_mask'):
                    child.save_mask()
                    child.destroy()
            
            load(idx[0] - 1)

    def delete_current():
        i = idx[0]
        if i >= len(seg_files):
            return

        seg = seg_files[i]
        raw = raw_files[i]
        
        annotations_dir = Path(root_dir) / "annotations2"
        deleted_masks_dir = annotations_dir / "deleted" / "masks"
        deleted_images_dir = annotations_dir / "deleted" / "images"
        
        deleted_masks_dir.mkdir(parents=True, exist_ok=True)
        deleted_images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(seg), str(deleted_masks_dir / seg.name))
            
            shutil.move(str(raw), str(deleted_images_dir / raw.name))
            
            status_var.set(f"Moved to deleted folder: {seg.name} and {raw.name}")
        except Exception as e:
            status_var.set(f"Error moving files: {e}")
            return

        # Remove from the list
        del seg_files[i]
        del raw_files[i]

        if i >= len(seg_files):
            idx[0] = max(0, len(seg_files) - 1)

        if seg_files:
            load(idx[0])
        else:
            status_var.set("No images left")

    def open_blended_view(i):
        if i >= len(seg_files):
            return
            
        try:
            # Keep track of whether the mask was modified
            mask_modified = [False]
            current_class = [0]  # Default class index
            drawing_mode = [None]  # None, 'polygon', or 'change'
            polygon_points = []
            
            # Load the original mask for editing
            original_mask = Image.open(seg_files[i]).convert('L')
            mask_array = np.array(original_mask)
            
            # Create the blended view
            raw_img = Image.open(raw_files[i]).convert('RGB')
            
            # Create colored mask
            height, width = mask_array.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            for j, gray_value in enumerate(CLASS_GRAY_COLORS):
                if j < len(CLASS_COLORS):
                    mask = (mask_array == gray_value)
                    colored_mask[mask] = CLASS_COLORS[j]
            
            mask_img = Image.fromarray(colored_mask)
            
            # Create blended image
            raw_array = np.array(raw_img).astype(float)
            mask_array_color = np.array(mask_img).astype(float)
            alpha = 0.4
            blended_array = (1 - alpha) * raw_array + alpha * mask_array_color
            blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
            blended_img = Image.fromarray(blended_array)
            
            # Create new window
            blended_window = Toplevel(window)
            blended_window.title(f"Edit Segmentation - {raw_files[i].name}")
            
            # Calculate display dimensions
            target_height = 600
            aspect_ratio = blended_img.width / blended_img.height
            new_width = int(target_height * aspect_ratio)
            
            # Create frames for layout
            main_frame = Frame(blended_window)
            main_frame.pack(fill="both", expand=True)
            
            canvas_frame = Frame(main_frame)
            canvas_frame.pack(side="left", fill="both", expand=True)
            
            control_frame = Frame(main_frame)
            control_frame.pack(side="right", fill="y", padx=10, pady=10)
            
            # Create canvas for the image
            canvas = Canvas(canvas_frame, width=new_width, height=target_height, bg="black", cursor="crosshair")
            canvas.pack(fill="both", expand=True)
            
            # Resize for display
            blended_img_resized = blended_img.resize((new_width, target_height), Image.LANCZOS)
            blended_photo = ImageTk.PhotoImage(blended_img_resized)
            
            # Display the image
            image_item = canvas.create_image(0, 0, anchor=NW, image=blended_photo)
            canvas.blended_photo = blended_photo
            scale_x = blended_img.width / new_width
            scale_y = blended_img.height / target_height
            
            def select_class(class_idx):
                current_class[0] = class_idx
                for btn in class_buttons:
                    btn.config(relief="raised")
                class_buttons[class_idx].config(relief="sunken")
                
            def set_mode(mode):
                drawing_mode[0] = mode
                polygon_points.clear()
                # Update UI to reflect mode
                if mode == 'polygon':
                    polygon_btn.config(relief="sunken")
                    change_class_btn.config(relief="raised")
                    canvas.config(cursor="crosshair")
                elif mode == 'change':
                    polygon_btn.config(relief="raised")
                    change_class_btn.config(relief="sunken")
                    canvas.config(cursor="hand2")
                else:
                    polygon_btn.config(relief="raised")
                    change_class_btn.config(relief="raised")
                    canvas.config(cursor="arrow")
                    
            def click_canvas(event):
                if drawing_mode[0] is None:
                    return
                    
                img_x = int(event.x * scale_x)
                img_y = int(event.y * scale_y)
            
                img_x = max(0, min(img_x, mask_array.shape[1] - 1))
                img_y = max(0, min(img_y, mask_array.shape[0] - 1))
                
                if drawing_mode[0] == 'polygon':
                    # Add point to polygon
                    polygon_points.append((img_x, img_y))
                    canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                                    fill="yellow", outline="black", tags="polygon")
                    
                    # If this is at least the second point, draw line to previous point
                    if len(polygon_points) > 1:
                        prev_x, prev_y = polygon_points[-2]
                        prev_x_display = int(prev_x / scale_x)
                        prev_y_display = int(prev_y / scale_y)
                        canvas.create_line(prev_x_display, prev_y_display, event.x, event.y,
                                        fill="yellow", width=2, tags="polygon")
                
                elif drawing_mode[0] == 'change':
                    # Get the gray value at the clicked point
                    gray_value = mask_array[img_y, img_x]
                    flood_fill(mask_array, img_y, img_x, gray_value, CLASS_GRAY_COLORS[current_class[0]])
                    mask_modified[0] = True
                    update_blended_view()
            
            def complete_polygon():
                if drawing_mode[0] != 'polygon' or len(polygon_points) < 3:
                    return
                    
                # Create a mask for the polygon area
                poly_mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(poly_mask)
                draw.polygon(polygon_points, fill=255)
                poly_array = np.array(poly_mask)
                
                # Update the mask array where the polygon is
                mask_array[poly_array > 0] = CLASS_GRAY_COLORS[current_class[0]]
                
                # Clear polygon points and drawing
                polygon_points.clear()
                canvas.delete("polygon")
                
                # Mark as modified and update the display
                mask_modified[0] = True
                update_blended_view()
            
            def update_blended_view():
                nonlocal blended_photo, image_item
                
                # Create updated colored mask
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                for j, gray_value in enumerate(CLASS_GRAY_COLORS):
                    if j < len(CLASS_COLORS):
                        mask = (mask_array == gray_value)
                        colored_mask[mask] = CLASS_COLORS[j]
                
                mask_img = Image.fromarray(colored_mask)
                
                # Create updated blended image
                raw_array = np.array(raw_img).astype(float)
                mask_array_color = np.array(mask_img).astype(float)
                blended_array = (1 - alpha) * raw_array + alpha * mask_array_color
                blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
                blended_img_new = Image.fromarray(blended_array)
                
                # Resize for display
                blended_img_resized = blended_img_new.resize((new_width, target_height), Image.LANCZOS)
                blended_photo = ImageTk.PhotoImage(blended_img_resized)
                canvas.itemconfig(image_item, image=blended_photo)
            
            def flood_fill(array, y, x, old_value, new_value):
                if old_value == new_value:
                    return
                    
                # Get array dimensions
                h, w = array.shape
                
                # Create a queue for BFS
                queue = deque([(y, x)])
                array[y, x] = new_value 
                
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                
                while queue:
                    cy, cx = queue.popleft()
                    
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if array[ny, nx] == old_value:
                                array[ny, nx] = new_value
                                queue.append((ny, nx))
            
            def save_mask():
                if mask_modified[0]:
                    new_mask = Image.fromarray(mask_array.astype(np.uint8))
                    new_mask.save(seg_files[i])
                    mask_modified[0] = False
            
            def on_window_close():
                save_mask()
                blended_window.destroy()
            
            ttk.Label(control_frame, text="Edit Tools:").pack(anchor="w", pady=(0, 5))
            
            tool_frame = Frame(control_frame)
            tool_frame.pack(fill="x", pady=5)
            
            polygon_btn = Button(tool_frame, text="Draw Polygon", 
                            command=lambda: set_mode('polygon'))
            polygon_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
            
            change_class_btn = Button(tool_frame, text="Change Class", 
                                    command=lambda: set_mode('change'))
            change_class_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
            
            complete_btn = Button(control_frame, text="Complete Polygon", 
                                command=complete_polygon)
            complete_btn.pack(fill="x", pady=5)
            
            ttk.Label(control_frame, text="Classes:").pack(anchor="w", pady=5)
            
            # Create class selection buttons
            class_frame = Frame(control_frame)
            class_frame.pack(fill="x", pady=5)
            
            class_buttons = []
            for idx, (color, name) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                btn = Button(class_frame, text=name, bg=hex_color, fg="white" if sum(color) < 380 else "black",
                        command=lambda i=idx: select_class(i))
                btn.pack(fill="x", pady=1)
                class_buttons.append(btn)
            
            select_class(0)
            
            save_btn = Button(control_frame, text="Save Changes", 
                            command=save_mask)
            save_btn.pack(fill="x", pady=10)
            
            canvas.bind("<Button-1>", click_canvas)
            blended_window.protocol("WM_DELETE_WINDOW", on_window_close)
            
        except Exception as e:
            status_var.set(f"Error creating edit view: {e}")

    def on_exit():
        for child in window.winfo_children():
            if isinstance(child, Toplevel) and hasattr(child, 'save_mask'):
                child.save_mask()
        
        window.destroy()

    ttk.Button(control, text="Previous", command=prev_image).pack(fill="x", pady=5)
    ttk.Button(control, text="Next", command=next_image).pack(fill="x", pady=5)
    ttk.Button(control, text="Show Blended View", command=lambda: open_blended_view(idx[0])).pack(fill="x", pady=5)
    ttk.Button(control, text="Delete Files", command=delete_current).pack(fill="x", pady=5)
    ttk.Button(control, text="Exit", command=on_exit).pack(fill="x", pady=20)

    load(0)
    window.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Viewer")
    parser.add_argument("root_dir", type=str, help="Root directory containing annotations2/images and annotations2/masks")
    args = parser.parse_args()

    root_dir = args.root_dir
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        exit(1)

    annotation_viewer(root_dir)