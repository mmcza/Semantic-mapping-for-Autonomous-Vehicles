from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
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
        default_value='/root/Shared/saved_models/model.onnx',
        description='Path to the ONNX segmentation model'
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
            'model_path': LaunchConfiguration('model_path')
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
        segmentation_node
    ])