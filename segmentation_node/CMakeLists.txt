cmake_minimum_required(VERSION 3.8)
project(segmentation_node)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIRS
  NAMES onnxruntime_cxx_api.h
  PATHS 
    ${ONNXRUNTIME_ROOT_DIR}/include
    /usr/local/onnxruntime/include
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS 
    ${ONNXRUNTIME_ROOT_DIR}/lib
    /usr/local/onnxruntime/lib
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT_DIR to point to the ONNX Runtime installation.")
else()
  message(STATUS "ONNX Runtime found at ${ONNXRUNTIME_INCLUDE_DIRS}")
  message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

# Define the executable
add_executable(segmentation_node 
  src/segmentation_node.cpp
)

ament_target_dependencies(segmentation_node
  rclcpp
  sensor_msgs
  message_filters
  tf2
  tf2_ros
  tf2_eigen
  cv_bridge
  pcl_conversions
  pcl_ros
)

# Add include directories
target_include_directories(segmentation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(segmentation_node
  ${ONNXRUNTIME_LIBRARY}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

# Install the executable
install(TARGETS segmentation_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # The following line skips the linter which checks for copyrights.
  # Comment the line when a copyright and license is added to all source files.
  set(ament_cmake_copyright_FOUND TRUE)
  # The following line skips cpplint (only works in a git repo).
  # Comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files.
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
