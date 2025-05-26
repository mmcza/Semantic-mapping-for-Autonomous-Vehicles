from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
import os

def generate_launch_description():
    package_dir = get_package_share_directory('segmentation_node')
    rviz_config_path = os.path.join(package_dir, 'rviz', 'segmentation_node.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
    )

    declare_use_octomap_arg = DeclareLaunchArgument(
        'use_octomap',
        default_value='true',
        description='Use Octomap server for 3D Mapping [true/false]'
    )

    declare_octomap_resolution_arg = DeclareLaunchArgument(
        'octomap_resolution',
        default_value='0.25',
        description='Resolution for the Octomap server'
    )
    
    octomap_server_node = Node(
        package='octomap_server',
        executable='color_octomap_server_node',
        name='octomap_server',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_octomap')),
        remappings=[
            ('cloud_in', '/sensing/lidar/segmented_pointcloud'),
        ],
        parameters=[{
            'frame_id': 'viewer',
            'base_frame_id': 'lidar_laser_top_link',
            'resolution': LaunchConfiguration('octomap_resolution'),
            'height_map': False,
            'filter_ground': False,
            'latch': False,
            'color_factor': 1.0,
        }]
    )

    declare_camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_front_optical_link',
        description='Camera optical frame ID'
    )
    
    declare_lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='lidar_laser_top_link',
        description='LiDAR frame ID'
    )
    
    declare_camera_image_topic_arg = DeclareLaunchArgument(
        'camera_image_topic',
        default_value='/sensing/camera/front/image_raw',
        description='Camera image topic'
    )
    
    declare_camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/sensing/camera/front/camera_info',
        description='Camera info topic'
    )
    
    declare_lidar_pointcloud_topic_arg = DeclareLaunchArgument(
        'lidar_pointcloud_topic',
        default_value='/sensing/lidar/top/pointcloud',
        description='LiDAR point cloud topic'
    )
    
    declare_segmented_pointcloud_topic_arg = DeclareLaunchArgument(
        'segmented_pointcloud_topic',
        default_value='/sensing/lidar/segmented_pointcloud',
        description='Output segmented point cloud topic'
    )
    
    declare_model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/root/Shared/saved_models/unet_mobilenet_v2_480_300_11C_SAM2.onnx',
        description='Path to the ONNX segmentation model'
    )

    declare_classes_with_colors_arg = DeclareLaunchArgument(
        'classes_with_colors',
        default_value='{' \
            ' "0": ["other", [0, 0, 0], true],' \
            ' "1": ["sky", [135, 206, 235], true],' \
            ' "2": ["building", [70, 70, 70], true],' \
            ' "3": ["grass", [0, 128, 0], true],' \
            ' "4": ["sand_mud", [210, 180, 140], true],' \
            ' "5": ["road", [128, 128, 128], true],' \
            ' "6": ["fence", [139, 69, 19], true],' \
            ' "7": ["tree", [34, 139, 34], true],' \
            ' "8": ["sign_lamp_pole_cone_bike", [255, 215, 0], true],' \
            ' "9": ["car_truck", [255, 0, 0], true],' \
            ' "10": ["person", [255, 192, 203], true]' \
        '}',
        description='Custom class-color mapping in JSON format with RGB values and visibility flags'
    )


    declare_thread_count_arg = DeclareLaunchArgument(
        'thread_count',
        default_value='4',
        description='Number of threads to use for processing'
    )
    
    segmentation_node = Node(
        package='segmentation_node',
        executable='segmentation_node',
        name='segmentation_node',
        output='screen',
        parameters=[{
            'camera_frame': LaunchConfiguration('camera_frame'),
            'lidar_frame': LaunchConfiguration('lidar_frame'),
            'camera_image_topic': LaunchConfiguration('camera_image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'lidar_pointcloud_topic': LaunchConfiguration('lidar_pointcloud_topic'),
            'segmented_pointcloud_topic': LaunchConfiguration('segmented_pointcloud_topic'),
            'model_path': LaunchConfiguration('model_path'),
            'classes_with_colors': ParameterValue(
                LaunchConfiguration('classes_with_colors'),
                value_type=str
            ),
            'thread_count': LaunchConfiguration('thread_count')
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        declare_camera_frame_arg,
        declare_lidar_frame_arg,
        declare_camera_image_topic_arg,
        declare_camera_info_topic_arg,
        declare_lidar_pointcloud_topic_arg,
        declare_segmented_pointcloud_topic_arg,
        declare_model_path_arg,
        declare_classes_with_colors_arg,
        declare_thread_count_arg,
        declare_octomap_resolution_arg,
        declare_use_octomap_arg,
        segmentation_node,
        octomap_server_node,
        rviz_node
    ])