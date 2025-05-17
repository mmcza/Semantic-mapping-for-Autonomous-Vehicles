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

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select ublox_serialization && \
    colcon build --symlink-install --packages-select ublox_msgs && \
    colcon build --symlink-install --packages-select tier4_debug_msgs && \
    colcon build --symlink-install --packages-select autoware_sensing_msgs && \
    source install/setup.bash"

RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Install CUDA 12.4
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN 9 for CUDA 12
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add CUDA paths to environment
ENV PATH /usr/local/cuda-12.4/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}

# Install ONNX Runtime 1.19 with GPU support for C++
WORKDIR /tmp
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz \
    && mkdir -p /usr/local/onnxruntime \
    && tar -xzf onnxruntime-linux-x64-gpu-1.19.0.tgz -C /usr/local/onnxruntime --strip-components=1 \
    && rm onnxruntime-linux-x64-gpu-1.19.0.tgz

# Set ONNX Runtime environment variables
ENV ONNXRUNTIME_ROOT_DIR=/usr/local/onnxruntime
ENV PATH=${ONNXRUNTIME_ROOT_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT_DIR}/lib:${LD_LIBRARY_PATH}
ENV CMAKE_PREFIX_PATH=${ONNXRUNTIME_ROOT_DIR}:${CMAKE_PREFIX_PATH}

# Install PCL and related packages
RUN apt-get update && apt-get install -y \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    ros-humble-ros-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/ros2_ws