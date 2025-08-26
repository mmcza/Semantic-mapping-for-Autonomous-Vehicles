import os
import time
import glob
import typer
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
import random
import gc
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List

# Add pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not available. GPU memory monitoring will be disabled.")
    print("To enable GPU monitoring, install pynvml: pip install pynvml")

def init_gpu_monitoring():
    """Initialize GPU monitoring with pynvml"""
    if not PYNVML_AVAILABLE:
        return False
    
    try:
        pynvml.nvmlInit()
        return True
    except Exception as e:
        print(f"Failed to initialize GPU monitoring: {e}")
        return False

def get_gpu_memory_usage(device_id=0):
    """Get current GPU memory usage in MB"""
    if not PYNVML_AVAILABLE:
        return 0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024  # Convert to MB
    except Exception as e:
        print(f"Failed to get GPU memory usage: {e}")
        return 0

def get_gpu_utilization(device_id=0):
    """Get current GPU utilization percentage"""
    if not PYNVML_AVAILABLE:
        return 0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu  # GPU utilization percentage
    except Exception as e:
        print(f"Failed to get GPU utilization: {e}")
        return 0

def measure_model_load_memory(model_paths, providers, num_iterations=100):
    """
    Measure memory usage for loading models multiple times in random order.
    
    Args:
        model_paths: List of paths to ONNX models
        providers: ONNX Runtime providers
        num_iterations: Number of iterations to load each model
        
    Returns:
        Dictionary with model_name as key and memory usage statistics as values
    """
    if not PYNVML_AVAILABLE:
        print("GPU memory monitoring not available. Skipping model load memory measurement.")
        return {}

    print(f"\n--- Measuring model load memory usage ({num_iterations} iterations each) ---")
    
    # Initialize GPU monitoring
    if not init_gpu_monitoring():
        print("Failed to initialize GPU monitoring. Skipping model load memory measurement.")
        return {}
    
    # Get baseline memory
    baseline_memory = get_gpu_memory_usage()
    print(f"Baseline GPU memory: {baseline_memory:.2f} MB")
    
    # Create a list of model indices to load in random order
    model_indices = []
    for i in range(len(model_paths)):
        model_indices.extend([i] * num_iterations)
    
    # Dictionary to store results
    model_load_stats = {}
    for i, path in enumerate(model_paths):
        model_load_stats[path.stem] = {
            'memory_usages': [],
            'load_times': []
        }
    
    # Run the loads in random order
    random.shuffle(model_indices)
    for idx in tqdm(model_indices, desc="Loading models"):
        model_path = model_paths[idx]
        model_name = model_path.stem
        
        try:
            # Clear memory and run garbage collection
            gc.collect()
            
            # Force CUDA to release memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Sleep a bit to let memory settle
            time.sleep(0.2)
            
            # Measure memory before loading
            pre_load_memory = get_gpu_memory_usage()
            
            # Load the model and time it
            start_time = time.perf_counter()
            session = ort.InferenceSession(str(model_path), providers=providers)
            end_time = time.perf_counter()
            
            # Measure memory after loading
            post_load_memory = get_gpu_memory_usage()
            
            # Calculate memory used for this load
            memory_used = post_load_memory - pre_load_memory
            load_time = (end_time - start_time) * 1000  # in ms
            
            # Store the results
            model_load_stats[model_name]['memory_usages'].append(memory_used)
            model_load_stats[model_name]['load_times'].append(load_time)
            
            # Clean up session to free memory
            del session
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    # Calculate statistics for each model
    result_stats = {}
    for model_name, data in model_load_stats.items():
        if data['memory_usages']:
            memory_array = np.array(data['memory_usages'])
            time_array = np.array(data['load_times'])
            
            result_stats[model_name] = {
                'min_load_memory_mb': np.min(memory_array),
                'max_load_memory_mb': np.max(memory_array),
                'mean_load_memory_mb': np.mean(memory_array),
                'median_load_memory_mb': np.median(memory_array),
                'std_load_memory_mb': np.std(memory_array),
                'min_load_time_ms': np.min(time_array),
                'max_load_time_ms': np.max(time_array),
                'mean_load_time_ms': np.mean(time_array)
            }
            
            print(f"\nModel: {model_name}")
            print(f"  Memory usage (min): {result_stats[model_name]['min_load_memory_mb']:.2f} MB")
            print(f"  Memory usage (max): {result_stats[model_name]['max_load_memory_mb']:.2f} MB")
            print(f"  Memory usage (mean): {result_stats[model_name]['mean_load_memory_mb']:.2f} MB")
            print(f"  Memory usage (median): {result_stats[model_name]['median_load_memory_mb']:.2f} MB")
            print(f"  Memory usage (std): {result_stats[model_name]['std_load_memory_mb']:.2f} MB")
            print(f"  Load time (mean): {result_stats[model_name]['mean_load_time_ms']:.2f} ms")
    
    return result_stats

def calculate_inference_times(
    models_dir: Path = typer.Option(..., "--models-dir", help="Directory containing ONNX models"),
    images_dir: Path = typer.Option(..., "--images-dir", help="Directory containing images"),
    output_file: Path = typer.Option(..., "--output-file", help="Output CSV file path"),
    img_size: Optional[Tuple[int, int]] = typer.Option(None, "--img-size", help="Image size (height, width)"),
    device: Optional[str] = typer.Option("auto", "--device", help="Device for ONNX runtime (CPU, CUDA, or auto)"),
    num_inferences: int = typer.Option(1000, "--num-inferences", "-n", help="Number of inferences to run (excluding warmup)"),
    measure_load_memory: bool = typer.Option(True, "--measure-load", help="Measure model load memory multiple times"),
    load_iterations: int = typer.Option(100, "--load-iterations", help="Number of times to load each model for memory measurement"),
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
    
    # Initialize GPU monitoring if using CUDA
    using_gpu = 'CUDAExecutionProvider' in providers
    gpu_monitoring_enabled = False
    
    if using_gpu:
        gpu_monitoring_enabled = init_gpu_monitoring()
        if gpu_monitoring_enabled:
            print("GPU monitoring enabled")
            # Measure baseline GPU memory before loading any models
            baseline_gpu_memory = get_gpu_memory_usage()
            baseline_gpu_util = get_gpu_utilization()
            print(f"Baseline GPU memory usage: {baseline_gpu_memory:.2f} MB")
            print(f"Baseline GPU utilization: {baseline_gpu_util:.2f}%")
    
    # Measure model load memory if requested
    model_load_stats = {}
    if measure_load_memory and gpu_monitoring_enabled:
        model_load_stats = measure_model_load_memory(onnx_models, providers, load_iterations)
    
    # Create a DataFrame to store all results
    all_results = []
    
    # Process each model for inference testing
    for model_path in onnx_models:
        model_name = model_path.stem
        print(f"\nProcessing model for inference: {model_name}")
        
        # Measure GPU memory before loading model
        if gpu_monitoring_enabled:
            pre_model_gpu_memory = get_gpu_memory_usage()
        
        # Load ONNX model
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Measure GPU memory after loading model
        if gpu_monitoring_enabled:
            post_model_gpu_memory = get_gpu_memory_usage()
            single_load_memory = post_model_gpu_memory - pre_model_gpu_memory
            print(f"Single model loading used {single_load_memory:.2f} MB of GPU memory")
        
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
        
        # Container for metrics
        inference_times = []
        gpu_memory_usages = [] if gpu_monitoring_enabled else None
        gpu_utilizations = [] if gpu_monitoring_enabled else None
        
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
                
                # Run inference and measure time and GPU usage
                ort_inputs = {input_name: img_batch}
                
                # Measure GPU metrics before inference
                if gpu_monitoring_enabled:
                    pre_inference_gpu_memory = get_gpu_memory_usage()
                    pre_inference_gpu_util = get_gpu_utilization()
                
                # Time the inference
                start_time = time.perf_counter()
                _ = session.run(None, ort_inputs)
                end_time = time.perf_counter()
                
                # Measure GPU metrics after inference
                if gpu_monitoring_enabled:
                    post_inference_gpu_memory = get_gpu_memory_usage()
                    post_inference_gpu_util = get_gpu_utilization()
                    
                    # Calculate delta memory and utilization (specifically for this inference)
                    inference_memory = post_inference_gpu_memory - pre_inference_gpu_memory
                    inference_util = max(0, post_inference_gpu_util - baseline_gpu_util)  # Remove background util
                
                # Skip the first inference (warmup)
                if i > 0:
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                    
                    if gpu_monitoring_enabled:
                        gpu_memory_usages.append(inference_memory)
                        gpu_utilizations.append(inference_util)
                
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
            
            # Add single load memory measurement
            if gpu_monitoring_enabled:
                stats['single_load_memory_mb'] = single_load_memory
            
            # Add detailed model loading stats if available
            if model_name in model_load_stats:
                load_stats = model_load_stats[model_name]
                for key, value in load_stats.items():
                    stats[key] = value
            
            # Add inference GPU statistics if available
            if gpu_monitoring_enabled and gpu_memory_usages:
                memory_array = np.array(gpu_memory_usages)
                util_array = np.array(gpu_utilizations)
                
                # Add inference memory metrics
                stats['inference_memory_min_mb'] = np.min(memory_array) if len(memory_array) > 0 else 0
                stats['inference_memory_max_mb'] = np.max(memory_array) if len(memory_array) > 0 else 0
                stats['inference_memory_mean_mb'] = np.mean(memory_array) if len(memory_array) > 0 else 0
                
                # Add GPU utilization metrics
                stats['gpu_util_min_percent'] = np.min(util_array) if len(util_array) > 0 else 0
                stats['gpu_util_max_percent'] = np.max(util_array) if len(util_array) > 0 else 0
                stats['gpu_util_mean_percent'] = np.mean(util_array) if len(util_array) > 0 else 0
            
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
            
            # Print GPU memory statistics
            if gpu_monitoring_enabled:
                print(f"  Single model load memory: {stats.get('single_load_memory_mb', 0):.2f} MB")
                
                if model_name in model_load_stats:
                    print(f"  Model load memory (min): {stats.get('min_load_memory_mb', 0):.2f} MB")
                    print(f"  Model load memory (max): {stats.get('max_load_memory_mb', 0):.2f} MB")
                    print(f"  Model load memory (mean): {stats.get('mean_load_memory_mb', 0):.2f} MB")
                    print(f"  Model load memory (median): {stats.get('median_load_memory_mb', 0):.2f} MB")
                
                print(f"  Inference memory (min): {stats.get('inference_memory_min_mb', 0):.2f} MB")
                print(f"  Inference memory (max): {stats.get('inference_memory_max_mb', 0):.2f} MB")
                print(f"  Inference memory (mean): {stats.get('inference_memory_mean_mb', 0):.2f} MB")
                print(f"  GPU utilization (min): {stats.get('gpu_util_min_percent', 0):.2f}%")
                print(f"  GPU utilization (max): {stats.get('gpu_util_max_percent', 0):.2f}%")
                print(f"  GPU utilization (mean): {stats.get('gpu_util_mean_percent', 0):.2f}%")
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