# Automatic image annotations with GroundingDINO and SAM2

> [!NOTE]  
> Based on https://github.com/luca-medeiros/lang-segment-anything

# Build the Docker

For annotations with SAM2 a separate Dockerfile is prepared (it's done because of the size of the image, and majority is not necessary for the ROS2 image). Go into `/utils/segmentation_with_sam` and run:
```
docker build -t sam_annotations .
```
Then you can start the container with the script available inside `/utils/segmentation_with_sam`:
```
bash start_container.sh
```
>[!NOTE]
> You can adjust the paths to directory with images for annotation.

# Use the script to generate masks

To use SAM2 to create masks on the images use:
```
python3 generate_masks_for_training.py --input_dir <path to directory with images> --output_dir <path to save the annotations> --model <name of model> --save_visualizations --min_area <minimal size>
```
Models to choose: `"sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"`

Example:
```
python3 generate_masks_for_training.py --input_dir ~/Shared/rosbag2_2025_04_14-17_54_17/ --output_dir ~/Shared/annotations --model sam2.1_hiera_large --save_visualizations --min_area 600
```

# Use the script to generate masks with batch querying (requires more VRAM)

To use SAM2 to create masks on the images, and query 5 classes at the same time, use:
```
python3 generate_masks_for_training_with_batch_queries.py --input_dir <path to directory with images> --output_dir <path to save the annotations> --model <name of model> --save_visualizations --min_area <minimal size>
```
Models to choose: `"sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"`

Example:
```
python3 generate_masks_for_training_with_batch_queries.py --input_dir ~/Shared/rosbag2_2025_04_14-17_54_17/ --output_dir ~/Shared/annotations --model sam2.1_hiera_large --save_visualizations --min_area 600
```
>[!WARNING]
> Model `sam2.1_hiera_large` requires ~11GB VRAM when running with batches,  and time gain is marginal.

# List of annotated objects:
- `1`: `Other` (doesn't fit any of the other classes listed below);
- `2`: `Sky`;
- `3`: `Building`;
- `4`: `Grass`;
- `5`: `Sand`, `Mud`;
- `6`: `Road`, `Asphalt`, `Cobblestone`;
- `7`: `Fence`;
- `8`: `Tree`;
- `9`: `Sign`, `Lamp`, `Pole`, `Cone`, `Bike`;
- `10`: `Car`, `Truck`;
- `11`: `Person`.