import os
import time
import glob
import typer
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List

def calculate_inference_times(
    models_dir: Path = typer.Option(..., "--models-dir", help="Directory containing ONNX models"),
    images_dir: Path = typer.Option(..., "--images-dir", help="Directory containing images"),
    output_file: Path = typer.Option(..., "--output-file", help="Output CSV file path"),
    img_size: Optional[Tuple[int, int]] = typer.Option(None, "--img-size", help="Image size (height, width)"),
    device: Optional[str] = typer.Option("auto", "--device", help="Device for ONNX runtime (CPU, CUDA, or auto)"),
    num_inferences: int = typer.Option(1000, "--num-inferences", "-n", help="Number of inferences to run (excluding warmup)"),
):
    # Find all ONNX models in the directory
    onnx_models = list(models_dir.glob("*.onnx"))
    if not onnx_models:
        print(f"No ONNX models found in {models_dir}")
        return
    
    print(f"Found {len(onnx_models)} ONNX models in {models_dir}")
    
    # Get all image filenames from the data directory
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    image_files = sorted(image_files)
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    # Limit to 1001 images (1 warmup + 1000 measurements)
    total_images_needed = min(1001, len(image_files))
    image_files = image_files[:total_images_needed]
    
    print(f"Using {len(image_files)} images from {images_dir}")
    
    # Set up ONNX Runtime providers
    available_providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")
    
    if device.lower() == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device.lower() == 'cuda' and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:  # auto
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"Using GPU for inference with providers: {providers}")
        else:
            providers = ['CPUExecutionProvider']
            print(f"Using CPU for inference with providers: {providers}")
    
    # Create a DataFrame to store all results
    all_results = []
    
    # Process each model
    for model_path in onnx_models:
        model_name = model_path.stem
        print(f"\nProcessing model: {model_name}")
        
        # Load ONNX model
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Get model input details
        input_details = session.get_inputs()[0]
        input_name = input_details.name
        
        # Determine input shape
        input_shape = input_details.shape
        if img_size:
            target_height, target_width = img_size
        else:
            # Extract size from model
            if len(input_shape) == 4 and -1 not in input_shape[2:]:
                target_height, target_width = input_shape[2], input_shape[3]
            else:
                print(f"Could not determine image size from model. Using 512x512 as default.")
                target_height, target_width = 512, 512
        
        print(f"Using input size: {target_height}x{target_width}")
        
        # Container for inference times
        inference_times = []
        
        # Run inferences
        for i, img_path in enumerate(tqdm(image_files, desc=f"Running inferences")):
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Error reading image {img_path}")
                    continue
                
                # Convert from BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target dimensions
                img_resized = cv2.resize(img, (target_width, target_height))
                
                # Normalize pixel values to [0,1] and transpose to CHW format
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_chw = img_normalized.transpose(2, 0, 1)
                
                # Add batch dimension
                img_batch = np.expand_dims(img_chw, axis=0)
                
                # Run inference and measure time
                ort_inputs = {input_name: img_batch}
                
                # Time the inference
                start_time = time.perf_counter()
                _ = session.run(None, ort_inputs)
                end_time = time.perf_counter()
                
                # Skip the first inference (warmup)
                if i > 0:
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                
                # Check if we've collected enough measurements
                if len(inference_times) >= num_inferences:
                    break
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        # Calculate statistics
        if inference_times:
            times_array = np.array(inference_times)
            stats = {
                'model_name': model_name,
                'min_time_ms': np.min(times_array),
                'max_time_ms': np.max(times_array),
                'mean_time_ms': np.mean(times_array),
                'std_dev_ms': np.std(times_array),
                'percentile_90_ms': np.percentile(times_array, 90),
                'percentile_99_ms': np.percentile(times_array, 99),
                'num_inferences': len(inference_times)
            }
            
            # Add to results
            all_results.append(stats)
            
            print(f"Model: {model_name}")
            print(f"  Min time: {stats['min_time_ms']:.2f} ms")
            print(f"  Max time: {stats['max_time_ms']:.2f} ms")
            print(f"  Mean time: {stats['mean_time_ms']:.2f} ms")
            print(f"  Std dev: {stats['std_dev_ms']:.2f} ms")
            print(f"  90th percentile: {stats['percentile_90_ms']:.2f} ms")
            print(f"  99th percentile: {stats['percentile_99_ms']:.2f} ms")
            print(f"  Measurements: {stats['num_inferences']}")
        else:
            print(f"No valid measurements for model {model_name}")
    
    if all_results:
        # Create final DataFrame and save to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to {output_file}")
        print(results_df)
    else:
        print("No valid results collected for any model.")

if __name__ == "__main__":
    typer.run(calculate_inference_times)