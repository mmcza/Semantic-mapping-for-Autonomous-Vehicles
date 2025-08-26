from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Define launch arguments
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/sensing/lidar/top/pointcloud',
        description='Topic with point cloud data'
    )
    
    declare_pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/localization/pose_twist_fusion_filter/biased_pose_with_covariance',
        description='Topic with robot pose data'
    )
    
    declare_lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='lidar_laser_top_link',
        description='LiDAR frame ID'
    )
    
    declare_base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Base frame ID'
    )
    
    declare_target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='viewer',
        description='Target frame for transforming point clouds'
    )
    
    declare_processing_interval_arg = DeclareLaunchArgument(
        'processing_interval',
        default_value='10',
        description='Process only every Nth message'
    )
    
    declare_voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='0.1',
        description='Voxel grid filter size in meters'
    )
    
    declare_min_points_per_voxel_arg = DeclareLaunchArgument(
        'min_points_per_voxel',
        default_value='8',
        description='Minimum points per voxel to keep in final filtering'
    )
    
    declare_min_distance_arg = DeclareLaunchArgument(
        'min_distance',
        default_value='1.35',
        description='Minimum distance from LiDAR to keep points (in meters)'
    )   
    
    declare_output_directory_arg = DeclareLaunchArgument(
        'output_directory',
        default_value=os.path.expanduser('~/pointcloud_maps'),
        description='Directory to save merged point cloud'
    )
    
    declare_output_filename_arg = DeclareLaunchArgument(
        'output_filename',
        default_value='merged_pointcloud',
        description='Base filename for the merged point cloud'
    )
    
    declare_check_publisher_timeout_arg = DeclareLaunchArgument(
        'check_publisher_timeout',
        default_value='5.0',
        description='Time to wait (in seconds) to check if publisher is still active'
    )
    
    declare_save_format_arg = DeclareLaunchArgument(
        'save_format',
        default_value='pcd',
        description='Format for saving the point cloud (pcd, ply, or both)'
    )
    
    declare_filter_outliers_arg = DeclareLaunchArgument(
        'filter_outliers',
        default_value='true',
        description='Whether to filter outlier points'
    )
    
    declare_outlier_radius_arg = DeclareLaunchArgument(
        'outlier_radius',
        default_value='2.5',
        description='Radius for outlier removal in meters'
    )
    
    declare_min_neighbors_arg = DeclareLaunchArgument(
        'min_neighbors',
        default_value='1',
        description='Minimum neighbors for outlier removal'
    )

    declare_max_distance_arg = DeclareLaunchArgument(
        'max_distance',
        default_value='60.0',
        description='Maximum distance from LiDAR to keep points (in meters)'
    )
    
    # Create the node
    pointcloud_merger_node = Node(
        package='pointcloud_merger',
        executable='pointcloud_merger_node',
        name='pointcloud_merger_node',
        parameters=[{
            'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            'lidar_frame': LaunchConfiguration('lidar_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
            'target_frame': LaunchConfiguration('target_frame'),
            'processing_interval': LaunchConfiguration('processing_interval'),
            'voxel_size': LaunchConfiguration('voxel_size'),
            'min_points_per_voxel': LaunchConfiguration('min_points_per_voxel'),
            'min_distance': LaunchConfiguration('min_distance'),
            'output_directory': LaunchConfiguration('output_directory'),
            'output_filename': LaunchConfiguration('output_filename'),
            'check_publisher_timeout': LaunchConfiguration('check_publisher_timeout'),
            'save_format': LaunchConfiguration('save_format'),
            'filter_outliers': LaunchConfiguration('filter_outliers'),
            'outlier_radius': LaunchConfiguration('outlier_radius'),
            'max_distance': LaunchConfiguration('max_distance'),
            'min_neighbors': LaunchConfiguration('min_neighbors')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        declare_pointcloud_topic_arg,
        declare_pose_topic_arg,
        declare_lidar_frame_arg,
        declare_base_frame_arg,
        declare_target_frame_arg,
        declare_processing_interval_arg,
        declare_voxel_size_arg,
        declare_min_points_per_voxel_arg,
        declare_min_distance_arg,
        declare_output_directory_arg,
        declare_output_filename_arg,
        declare_check_publisher_timeout_arg,
        declare_save_format_arg,
        declare_filter_outliers_arg,
        declare_outlier_radius_arg,
        declare_min_neighbors_arg,
        declare_max_distance_arg,
        pointcloud_merger_node
    ])