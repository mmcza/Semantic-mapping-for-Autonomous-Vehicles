# Semantic mapping for Autonomous Vehicles

![Semantic map example](/media/Semantic_mapping_example1.png)

- [Semantic mapping for Autonomous Vehicles](#semantic-mapping-for-autonomous-vehicles)
  - [The Docker container](#the-docker-container)
  - [ROS2 Node for mapping with semantic information](#ros2-node-for-mapping-with-semantic-information)
  - [Segmentation models](#segmentation-models)
  - [Tools](#tools)
    - [Get images from rosbags](#get-images-from-rosbags)
    - [Annotate the data with SAM2](#annotate-the-data-with-sam2)
      - [Build the Docker](#build-the-docker)
      - [Use the script to generate masks](#use-the-script-to-generate-masks)
      - [Use the script to generate masks with batch querying (requires more VRAM)](#use-the-script-to-generate-masks-with-batch-querying-requires-more-vram)
  - [List of annotated objects:](#list-of-annotated-objects)

## The Docker container

To build the docker image (requires around 12.6 GB of space), clone the repository, and once you're inside the directory, use the following command:
```
docker build -t semantic-mapping .
```

You can start the container by running:
```
bash start_container.sh
```
>[!NOTE]
>You can adjust the paths to the directory with rosbags and with segmentation model inside the bash script.

>[!WARNING]
>By default the docker container is removed after being closed and all data is lost. To avoid that you can remove `--rm` flag, save the data inside shared folders or copy the data from the container to your computer.

## ROS2 Node for mapping with semantic information

![Semantic map example 2](/media/Semantic_mapping_example2.png)

To start the ROS2 node that runs segmentation on images from a camera and adjusts color of a pointcloud from LIDAR/depth camera based on the segmentation mask, you can run:

```
ros2 launch segmentation_node segmentation_node_launch.py
```

>[!NOTE]
>You can check all available adjustable parameters (topic names, frame ids, not using Octomap, changing number of threads, etc.) by running `ros2 launch segmentation_node segmentation_node_launch.py --show-args`.

The launch file starts the segmentation node that waits for information from topic with camera information, Octomap server and RVIZ for visualition. By default Octomap is not visible in RVIZ due to long time required to update the visualization (the map itself runs fine), but you can enable it at any time. Using default settings, the time required to process a pointcloud with corresponding rgb image is on average around 50 ms (20 fps) on computer with AMD Ryzen 5 5600X with NVidia RTX 4070. Running single-threaded slowed down processing to around 10 fps, while running on all 12 threads led to 25-30 fps.

## Segmentation models

## Tools

### Get images from rosbags

To save all images (and undistort them) from a ROS2 bag you can go into the `utils` directory and run:

```
python3 ros2_bag_to_png.py --bagfile <path to rosbag> --topics <string with topics split with commas> --output <path to save the images (a subdir with name of rosbag will be created there)>
```

Example:
```
python3 ros2_bag_to_png.py --bagfile ~/Shared/rosbags/rosbag2_2025_04_14-17_54_17/ --topics '/sensing/camera/front/image_raw' --output '/root/images_from_rosbags/'
```

>[!NOTE]
>Inside the script the distortion parameters are defined. You can adjust them to work with your camera. You can find the parameters in appropriate ROS2 topic.

### Annotate the data with SAM2

#### Build the Docker

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

> [!NOTE]  
> Based on https://github.com/luca-medeiros/lang-segment-anything

#### Use the script to generate masks

To use SAM2 to create masks on the images use:
```
python3 generate_masks_for_training.py --input_dir <path to directory with images> --output_dir <path to save the annotations> --model <name of model> --save_visualizations --min_area <minimal size>
```
Models to choose: `"sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"`

Example:
```
python3 generate_masks_for_training.py --input_dir ~/Shared/rosbag2_2025_04_14-17_54_17/ --output_dir ~/Shared/annotations --model sam2.1_hiera_large --save_visualizations --min_area 600
```

#### Use the script to generate masks with batch querying (requires more VRAM)

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

## List of annotated objects:
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