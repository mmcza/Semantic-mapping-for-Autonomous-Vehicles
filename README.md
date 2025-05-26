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

> [!NOTE]
> You can adjust the paths to the directory with rosbags and with segmentation model inside the bash script.

> [!WARNING]
> By default the docker container is removed after being closed and all data is lost. To avoid that you can remove `--rm` flag, save the data inside shared folders or copy the data from the container to your computer.

## ROS2 Node for mapping with semantic information

![Semantic map example 2](/media/Semantic_mapping_example2.png)

An additional semantic information can be valuable in multiple robotic tasks, particularly in navigation and path planning. For example, it enables intelligent path cost assignment based on terrain type (allowing robots to favor asphalt over slippery or muddy surfaces for safer, faster travel), and distinguishing between static and dynamic obstacles, a knowledge that is useful in planning avoidance strategies.

To start the ROS2 node that runs segmentation on images from a camera and adjusts color of a pointcloud from LIDAR/depth camera based on the segmentation mask, you can run:

```
ros2 launch segmentation_node segmentation_node_launch.py
```

> [!NOTE]
> You can check all available adjustable parameters (topic names, frame ids, not using Octomap, changing number of threads, etc.) by running `ros2 launch segmentation_node segmentation_node_launch.py --show-args`.

![Pointcloud with colors based on semantic segmentation](media/semantic_pointcloud.gif)

The ROS2 node implements a synchronized processing pipeline for semantic mapping. It begins by subscribing to RGB camera images and point cloud data from topics specified as parameters. These images go through preprocessing, including resolution adjustment and normalization using ImageNet weights, before being passed to an ONNX-format segmentation model for inference. Using transformation between the camera and LiDAR frames (obtained from the `/tf` topic), the each 3D point is projected onto the segmentation mask and assigns appropriate semantic colors based on the predicted labels. The resulting point cloud is then published to another topic, where the OctoMap server can subscribe to it and construct a semantic 3D map of the environment.

The launch file starts the segmentation node, Octomap server and RVIZ for visualition. By default Octomap is not visible in RVIZ due to long time required to update the visualization (the map itself runs fine), but you can enable it at any time. Using default settings, the time required to process a pointcloud with corresponding rgb image is on average around 50 ms (20 fps) on computer with AMD Ryzen 5 5600X, 32GB of RAM and NVidia RTX 4070. Running single-threaded slowed down processing to around 10 fps, while running on all 12 threads led to 25-30 fps.

### Class filtering

The segmentation node allows to remove points belonging to selected classes. Users can configure which classes are visible by adjusting the `classes_with_colors` parameter in JSON format, where the boolean value (`true` or `false`) determines whether points from that class are included in the final point cloud visualization. This can be useful to create a general map of the environment, where all movable objects are removed (e.g. cars/people from a road or parking lot). The other usage can be creation of a 3D model of a specific object that is a target of a robot (e.g. a cup on a desk).

![Example of filtering](/media/filter_no_filter_comp.png)

## Segmentation models

## Inference performance

| Model                               |    Input | Classes | min (ms) | max (ms) | mean (ms) | std (ms) | p90 (ms) | p99 (ms) |
| ----------------------------------- | -------: | ------: | -------: | -------: | --------: | -------: | -------: | -------: |
| FPN (ResNet-18)                     | 960×608 |      19 |  10.0759 |  29.8761 |   11.5372 |   1.2220 |  13.0184 |  14.3673 |
| DeepLabV3+ (EfficientNet-B2)        | 960×608 |      19 |  13.2848 |  35.0960 |   15.1419 |   1.5407 |  16.9568 |  19.3563 |
| FPN (EfficientNet-B0 + SAM2)        | 480×304 |      11 |   2.9341 |   6.2591 |    3.4220 |   0.3605 |   3.8716 |   4.4724 |
| DeepLabV3+ (MobileNet-V2)           | 960×608 |      19 |   9.3042 |  26.9504 |   10.0985 |   0.8786 |  11.0582 |  12.3670 |
| LinkNet (MobileNet-V2)              | 960×608 |      19 |   9.2175 |  26.0415 |   10.1464 |   0.8092 |  11.0638 |  12.0223 |
| DeepLabV3+ (MobileNet-V2)           | 960×608 |      19 |   9.2910 |  25.4768 |   10.1944 |   0.8288 |  11.1180 |  12.1764 |
| U-Net (EfficientNet-B2)             | 960×608 |      19 |  13.8027 |  34.7337 |   15.6287 |   1.3539 |  17.3227 |  18.9289 |
| SegFormer (EfficientNet-B2)         | 960×608 |      19 |  14.3441 |  41.8935 |   15.7870 |   1.6151 |  17.6692 |  19.5966 |
| U-Net (MobileNet-V2 + SAM2)         | 480×300 |      11 |   2.9160 |   5.1397 |    3.2360 |   0.1870 |   3.3363 |   4.0436 |
| LinkNet (ResNet-34)                 | 960×608 |      19 |  12.5198 |  29.3958 |   12.7829 |   0.5589 |  12.9578 |  13.5117 |
| DeepLabV3+ (EfficientNet-B0 + SAM2) | 480×304 |      11 |   2.9897 |   4.2260 |    3.3161 |   0.1276 |   3.4464 |   3.5723 |

## Segmentation models visualisations

| DeepLabV3+                                                                                   | UNet                                                                                   | LinkNet                                                                                   |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| ![DeepLabV3+](https://github.com/user-attachments/assets/75b98bd0-054c-430a-8e98-507324070ed3) | ![UNet](https://github.com/user-attachments/assets/ed6f41f7-f5d6-435c-bbe2-0cf491cf7068) | ![LinkNet](https://github.com/user-attachments/assets/4f29ce41-6288-41e3-a464-64d3218d54a6) |
| *([DeepLabV3+ citation](https://arxiv.org/abs/1802.02611))*                                   | *([UNet citation](https://arxiv.org/abs/1505.04597))*                                   | *([LinkNet citation](https://www.mdpi.com/2072-4292/14/9/2012))*                           |

| SegFormer                                                                                   | FPN                                                                                   |  |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | - |
| ![SegFormer](https://github.com/user-attachments/assets/89b29b42-db08-4251-b65a-9bde18db4d1e) | ![FPN](https://github.com/user-attachments/assets/abd8ce18-3b97-483d-a396-a918ae449cd0) |  |
| *([SegFormer citation](https://arxiv.org/abs/2105.15203))*                                   | *([FPN citation](https://doi.org/10.1007/s00521-019-04546-6))*                         |  |

## Class Definitions Comparison
| Custom SAM2 Classes (11) | Cityscapes Classes (19) |
|---------------------|-------------------------|
| ![#000000](https://img.shields.io/badge/-000000?style=flat-square&logoColor=white) `0: Other` (0,0,0) | ![#804080](https://img.shields.io/badge/-804080?style=flat-square&logoColor=white) `0: road` (128,64,128) |
| ![#87CEEB](https://img.shields.io/badge/-87CEEB?style=flat-square&logoColor=white) `1: Sky` (135,206,235) | ![#F423E8](https://img.shields.io/badge/-F423E8?style=flat-square&logoColor=white) `1: sidewalk` (244,35,232) |
| ![#464646](https://img.shields.io/badge/-464646?style=flat-square&logoColor=white) `2: Building` (70,70,70) | ![#464646](https://img.shields.io/badge/-464646?style=flat-square&logoColor=white) `2: building` (70,70,70) |
| ![#008000](https://img.shields.io/badge/-008000?style=flat-square&logoColor=white) `3: Grass` (0,128,0) | ![#66669C](https://img.shields.io/badge/-66669C?style=flat-square&logoColor=white) `3: wall` (102,102,156) |
| ![#D2B48C](https://img.shields.io/badge/-D2B48C?style=flat-square&logoColor=white) `4: Sand, Mud` (210,180,140) | ![#BE9999](https://img.shields.io/badge/-BE9999?style=flat-square&logoColor=white) `4: fence` (190,153,153) |
| ![#808080](https://img.shields.io/badge/-808080?style=flat-square&logoColor=white) `5: Road, Asphalt, Cobblestone` (128,128,128) | ![#999999](https://img.shields.io/badge/-999999?style=flat-square&logoColor=white) `5: pole` (153,153,153) |
| ![#8B4513](https://img.shields.io/badge/-8B4513?style=flat-square&logoColor=white) `6: Fence` (139,69,19) | ![#FAAA1E](https://img.shields.io/badge/-FAAA1E?style=flat-square&logoColor=white) `6: traffic light` (250,170,30) |
| ![#228B22](https://img.shields.io/badge/-228B22?style=flat-square&logoColor=white) `7: Tree` (34,139,34) | ![#DCDC00](https://img.shields.io/badge/-DCDC00?style=flat-square&logoColor=white) `7: traffic sign` (220,220,0) |
| ![#FFD700](https://img.shields.io/badge/-FFD700?style=flat-square&logoColor=white) `8: Sign, Lamp, Pole, Cone, Bike` (255,215,0) | ![#6B8E23](https://img.shields.io/badge/-6B8E23?style=flat-square&logoColor=white) `8: vegetation` (107,142,35) |
| ![#FF0000](https://img.shields.io/badge/-FF0000?style=flat-square&logoColor=white) `9: Car, Truck` (255,0,0) | ![#98FB98](https://img.shields.io/badge/-98FB98?style=flat-square&logoColor=white) `9: terrain` (152,251,152) |
| ![#FFC0CB](https://img.shields.io/badge/-FFC0CB?style=flat-square&logoColor=white) `10: Person` (255,192,203) | ![#4682B4](https://img.shields.io/badge/-4682B4?style=flat-square&logoColor=white) `10: sky` (70,130,180) |
| — | ![#DC143C](https://img.shields.io/badge/-DC143C?style=flat-square&logoColor=white) `11: person` (220,20,60) |
| — | ![#FF0000](https://img.shields.io/badge/-FF0000?style=flat-square&logoColor=white) `12: rider` (255,0,0) |
| — | ![#00008E](https://img.shields.io/badge/-00008E?style=flat-square&logoColor=white) `13: car` (0,0,142) |
| — | ![#000046](https://img.shields.io/badge/-000046?style=flat-square&logoColor=white) `14: truck` (0,0,70) |
| — | ![#003C64](https://img.shields.io/badge/-003C64?style=flat-square&logoColor=white) `15: bus` (0,60,100) |
| — | ![#005064](https://img.shields.io/badge/-005064?style=flat-square&logoColor=white) `16: train` (0,80,100) |
| — | ![#0000E6](https://img.shields.io/badge/-0000E6?style=flat-square&logoColor=white) `17: motorcycle` (0,0,230) |
| — | ![#770B20](https://img.shields.io/badge/-770B20?style=flat-square&logoColor=white) `18: bicycle` (119,11,32) |

## Segmentation Annotation Viewer
![media](https://github.com/mmcza/Semantic-mapping-for-Autonomous-Vehicles/blob/main/media/viewer.png)

A tool for interactively browsing raw camera frames and their corresponding `*_segmented_mask.png` overlays. It auto-scans all `images_*` folders under `annotations2/`, pairs masks with their `_image_raw_*.png` counterparts (by suffix after `__`), and displays them vertically. Use “Prev”/“Next” to navigate and “Delete” to move bad pairs into `deleted/`.

## Created datasets links
SAM2 generated masks: https://www.kaggle.com/datasets/mrj111/sem-seg-veh/data

NVIDIA segformer-b5-finetuned-cityscapes-1024-1024 generated masks: https://www.kaggle.com/datasets/mrj111/sem-seg-veh-b5

## Wandb Training Logs
SAM2 custom: https://wandb.ai/qbizm/seg_sem_veh1

City scapes: https://wandb.ai/qbizm/seg_sem_veh_b5

![image](https://github.com/user-attachments/assets/a39bc9d9-4850-4c39-bd99-c96cb2b0d434)



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

> [!NOTE]
> Inside the script the distortion parameters are defined. You can adjust them to work with your camera. You can find the parameters in appropriate ROS2 topic.

### Annotate the data with GroundingDINO and SAM2

A whole pipeline to automatically annotate images is available inside [segmentation_with_sam directory](/utils/segmentation_with_sam/).
