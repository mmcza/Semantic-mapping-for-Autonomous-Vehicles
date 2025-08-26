cmake_minimum_required(VERSION 3.8)
project(pointcloud_merger)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)

# C++17 for std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Create executable
add_executable(pointcloud_merger_node src/pointcloud_merger_node.cpp)

# Specify target libraries
target_link_libraries(pointcloud_merger_node
  ${PCL_LIBRARIES}
)

# Add dependencies
ament_target_dependencies(pointcloud_merger_node
  rclcpp
  sensor_msgs
  tf2_ros
  tf2_eigen
  pcl_conversions
)

# Install executable
install(TARGETS
  pointcloud_merger_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()