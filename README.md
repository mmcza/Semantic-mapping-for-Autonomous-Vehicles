# Semantic mapping for Autonomous Vehicles

![Semantic map example](/media/Semantic_mapping_example1.png)

- [Semantic mapping for Autonomous Vehicles](#semantic-mapping-for-autonomous-vehicles)
  - [The Docker container](#the-docker-container)
  - [ROS2 Node for mapping with semantic information](#ros2-node-for-mapping-with-semantic-information)
    - [Class filtering](#class-filtering)
  - [Segmentation models](#segmentation-models)
  - [Tools](#tools)
    - [Get images from rosbags](#get-images-from-rosbags)
    - [Annotate the data with GroundingDINO and SAM2](#annotate-the-data-with-groundingdino-and-sam2)

![Octomap generation](/media/semantic_octomap_generation.gif)

An example of an Octomap created with the additional semantic information. The recording was sped up 10x to show the complete mapping process.

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

An additional semantic information can be valuable in multiple robotic tasks, particularly in navigation and path planning. For example, it enables intelligent path cost assignment based on terrain type (allowing robots to favor asphalt over slippery or muddy surfaces for safer, faster travel), and distinguishing between static and dynamic obstacles, a knowledge that is useful in planning avoidance strategies.

To start the ROS2 node that runs segmentation on images from a camera and adjusts color of a pointcloud from LIDAR/depth camera based on the segmentation mask, you can run:

```
ros2 launch segmentation_node segmentation_node_launch.py
```

>[!NOTE]
>You can check all available adjustable parameters (topic names, frame ids, not using Octomap, changing number of threads, etc.) by running `ros2 launch segmentation_node segmentation_node_launch.py --show-args`.

![Pointcloud with colors based on semantic segmentation](media/semantic_pointcloud.gif)

The ROS2 node implements a synchronized processing pipeline for semantic mapping. It begins by subscribing to RGB camera images and point cloud data from topics specified as parameters. These images go through preprocessing, including resolution adjustment and normalization using ImageNet weights, before being passed to an ONNX-format segmentation model for inference. Using transformation between the camera and LiDAR frames (obtained from the `/tf` topic), the each 3D point is projected onto the segmentation mask and assigns appropriate semantic colors based on the predicted labels. The resulting point cloud is then published to another topic, where the OctoMap server can subscribe to it and construct a semantic 3D map of the environment.

The launch file starts the segmentation node, Octomap server and RVIZ for visualition. By default Octomap is not visible in RVIZ due to long time required to update the visualization (the map itself runs fine), but you can enable it at any time. Using default settings, the time required to process a pointcloud with corresponding rgb image is on average around 50 ms (20 fps) on computer with AMD Ryzen 5 5600X, 32GB of RAM and NVidia RTX 4070. Running single-threaded slowed down processing to around 10 fps, while running on all 12 threads led to 25-30 fps.

### Class filtering

The segmentation node allows to remove points belonging to selected classes. Users can configure which classes are visible by adjusting the `classes_with_colors` parameter in JSON format, where the boolean value (`true` or `false`) determines whether points from that class are included in the final point cloud visualization. This can be useful to create a general map of the environment, where all movable objects are removed (e.g. cars/people from a road or parking lot). The other usage can be creation of a 3D model of a specific object that is a target of a robot (e.g. a cup on a desk). 

![Example of filtering](/media/filter_no_filter_comp.png)

## Segmentation models
## Inference performance

| Model                                   | Input     | Classes | min (ms)  | max (ms)  | mean (ms) | std (ms)  | p90 (ms)  | p99 (ms)  |
|-----------------------------------------|----------:|--------:|----------:|----------:|----------:|----------:|----------:|----------:|
| FPN (ResNet-18)                         | 960×608   | 19      | 10.0759   | 29.8761   | 11.5372   |  1.2220   | 13.0184   | 14.3673   |
| DeepLabV3+ (EfficientNet-B2)            | 960×608   | 19      | 13.2848   | 35.0960   | 15.1419   |  1.5407   | 16.9568   | 19.3563   |
| FPN (EfficientNet-B0 + SAM2)            | 480×304   | 11      |  2.9341   |  6.2591   |  3.4220   |  0.3605   |  3.8716   |  4.4724   |
| DeepLabV3+ (MobileNet-V2)               | 960×608   | 19      |  9.3042   | 26.9504   | 10.0985   |  0.8786   | 11.0582   | 12.3670   |
| LinkNet (MobileNet-V2)                  | 960×608   | 19      |  9.2175   | 26.0415   | 10.1464   |  0.8092   | 11.0638   | 12.0223   |
| DeepLabV3+ (MobileNet-V2)               | 960×608   | 19      |  9.2910   | 25.4768   | 10.1944   |  0.8288   | 11.1180   | 12.1764   |
| U-Net (EfficientNet-B2)                 | 960×608   | 19      | 13.8027   | 34.7337   | 15.6287   |  1.3539   | 17.3227   | 18.9289   |
| SegFormer (EfficientNet-B2)             | 960×608   | 19      | 14.3441   | 41.8935   | 15.7870   |  1.6151   | 17.6692   | 19.5966   |
| U-Net (MobileNet-V2 + SAM2)             | 480×300   | 11      |  2.9160   |  5.1397   |  3.2360   |  0.1870   |  3.3363   |  4.0436   |
| LinkNet (ResNet-34)                     | 960×608   | 19      | 12.5198   | 29.3958   | 12.7829   |  0.5589   | 12.9578   | 13.5117   |
| DeepLabV3+ (EfficientNet-B0 + SAM2)     | 480×304   | 11      |  2.9897   |  4.2260   |  3.3161   |  0.1276   |  3.4464   |  3.5723   |



## Segmentation models visualisations

| DeepLabV3+ | UNet | FPN |
|------------|------|-----|
| ![DeepLabV3+](https://github.com/user-attachments/assets/75b98bd0-054c-430a-8e98-507324070ed3) | ![UNet](https://github.com/user-attachments/assets/ed6f41f7-f5d6-435c-bbe2-0cf491cf7068) | ![LinkNet](https://github.com/user-attachments/assets/4f29ce41-6288-41e3-a464-64d3218d54a6) |
| *([DeepLabV3+ citation](https://arxiv.org/abs/1802.02611))* | *([UNet citation](https://arxiv.org/abs/1505.04597))* | *([LinkNet citation](https://www.mdpi.com/2072-4292/14/9/2012))* |

| SegFormer | LinkNet |  |
|-----------|---------|--|
| ![SegFormer](https://github.com/user-attachments/assets/89b29b42-db08-4251-b65a-9bde18db4d1e) | ![FPN](https://github.com/user-attachments/assets/abd8ce18-3b97-483d-a396-a918ae449cd0) |  |
| *([SegFormer citation](https://arxiv.org/abs/2105.15203))* | *([FPN citation](https://doi.org/10.1007/s00521-019-04546-6))* |  |










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

### Annotate the data with GroundingDINO and SAM2

A whole pipeline to automatically annotate images is available inside [segmentation_with_sam directory](/utils/segmentation_with_sam/).
