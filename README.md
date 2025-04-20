# Semantic mapping for Autonomous Vehicles

- [Semantic mapping for Autonomous Vehicles](#semantic-mapping-for-autonomous-vehicles)
  - [The Docker container](#the-docker-container)
  - [Tools](#tools)
    - [Get images from rosbags](#get-images-from-rosbags)
    - [Annotate the data with SAM2](#annotate-the-data-with-sam2)
  - [Info about the rosbags](#info-about-the-rosbags)

## The Docker container

To build the docker image, clone the repository and when inside the directory use the following command:
```
docker build -t semantic-mapping .
```

You can start the container by running:
```
bash start_container.sh
```
>[!NOTE]
>You can adjust the paths to this package and to the directory with rosbags inside the bash script.

>[!WARNING]
>By default the docker container is removed after being closed and all data is lost. To avoid that you can remove `--rm` flag, save the data inside shared folders or copy the data from the container to your computer.

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

TODO

## Info about the rosbags

- As global frame use `viewer`
- For RGB image use topic `/sensing/camera/front/image_raw`. To display in RViz use the following settings:
  - History Policy: `Keep Last`
  - Reliability Policy: `Best Effort`
  - Durability Policy: `Volatile`
- 