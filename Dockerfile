FROM osrf/ros:humble-desktop

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    nano \
    git \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-py \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rosbag2-compression \
    ros-humble-rosbag2-compression-zstd \
    python3-opencv \
    libasio-dev \
    libgtest-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create a workspace directory, clone the required repositories and build them
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws/src

RUN git clone -b ros2 https://github.com/KumarRobotics/ublox.git && \
    git clone https://github.com/tier4/tier4_autoware_msgs.git && \
    git clone https://github.com/autowarefoundation/autoware_msgs.git

WORKDIR /root/ros2_ws

RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select ublox_serialization && \
    colcon build --symlink-install --packages-select ublox_msgs && \
    colcon build --symlink-install --packages-select tier4_debug_msgs && \
    colcon build --symlink-install --packages-select autoware_sensing_msgs && \
    source install/setup.bash

RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

WORKDIR /root/Shared/