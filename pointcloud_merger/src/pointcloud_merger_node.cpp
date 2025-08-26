#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <memory>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <limits>
#include <Eigen/Geometry>

class PointCloudMergerNode : public rclcpp::Node 
{
public:
    PointCloudMergerNode() : Node("pointcloud_merger_node") 
    {
        // Declare parameters
        declare_parameter("pointcloud_topic", "/sensing/lidar/top/pointcloud");
        declare_parameter("pose_topic", "/localization/pose_twist_fusion_filter/biased_pose_with_covariance");
        declare_parameter("lidar_frame", "lidar_laser_top_link");
        declare_parameter("base_frame", "base_link");
        declare_parameter("target_frame", "viewer");
        declare_parameter("processing_interval", 50);
        declare_parameter("voxel_size", 0.01);  // 1 cm resolution
        declare_parameter("min_points_per_voxel", 100);  // Minimum points to keep a voxel
        declare_parameter("min_distance", 0.75);  // Minimum distance from LiDAR in meters
        declare_parameter("max_distance", 25.0);  // Maximum distance from LiDAR in meters
        declare_parameter("output_directory", std::string(getenv("HOME")) + "/pointcloud_maps");
        declare_parameter("output_filename", "merged_pointcloud");
        declare_parameter("check_publisher_timeout", 5.0);  // Time to wait before checking if publisher is gone
        declare_parameter("save_format", "pcd");  // Options: "pcd", "ply", or "both"
        declare_parameter("filter_outliers", true);  // Remove outliers
        declare_parameter("outlier_radius", 0.5);  // Radius for outlier removal in meters
        declare_parameter("min_neighbors", 2);  // Minimum neighbors for outlier removal
        
        // Get parameters
        pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
        pose_topic_ = get_parameter("pose_topic").as_string();
        lidar_frame_ = get_parameter("lidar_frame").as_string();
        base_frame_ = get_parameter("base_frame").as_string();
        target_frame_ = get_parameter("target_frame").as_string();
        processing_interval_ = get_parameter("processing_interval").as_int();
        voxel_size_ = get_parameter("voxel_size").as_double();
        min_points_per_voxel_ = get_parameter("min_points_per_voxel").as_int();
        min_distance_ = get_parameter("min_distance").as_double();
        max_distance_ = get_parameter("max_distance").as_double();
        output_directory_ = get_parameter("output_directory").as_string();
        output_filename_ = get_parameter("output_filename").as_string();
        check_publisher_timeout_ = get_parameter("check_publisher_timeout").as_double();
        save_format_ = get_parameter("save_format").as_string();
        filter_outliers_ = get_parameter("filter_outliers").as_bool();
        outlier_radius_ = get_parameter("outlier_radius").as_double();
        min_neighbors_ = get_parameter("min_neighbors").as_int();
        
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_directory_);
        
        // Initialize TF buffer and listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Initialize the combined point cloud
        raw_combined_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        
        // Flag to ensure we have received at least one pose message
        have_pose_ = false;
        
        // Create subscriber for pose with a QoS profile suitable for sensor data
        auto qos = rclcpp::QoS(rclcpp::SensorDataQoS());
        
        // Create point cloud subscription
        pointcloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic_, qos,
            std::bind(&PointCloudMergerNode::pointcloud_callback, this, std::placeholders::_1));
            
        // Create pose subscription
        pose_subscription_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            pose_topic_, qos,
            std::bind(&PointCloudMergerNode::pose_callback, this, std::placeholders::_1));
        
        // Create timer to check if publisher has stopped
        publisher_check_timer_ = create_wall_timer(
            std::chrono::duration<double>(check_publisher_timeout_),
            std::bind(&PointCloudMergerNode::check_publisher, this));
        
        // Track when we last received a message
        last_message_time_ = get_clock()->now();
        
        RCLCPP_INFO(get_logger(), "PointCloud Merger Node initialized with:");
        RCLCPP_INFO(get_logger(), "- Subscribing to point cloud: %s", pointcloud_topic_.c_str());
        RCLCPP_INFO(get_logger(), "- Subscribing to pose: %s", pose_topic_.c_str());
        RCLCPP_INFO(get_logger(), "- LiDAR frame: %s", lidar_frame_.c_str());
        RCLCPP_INFO(get_logger(), "- Base frame: %s", base_frame_.c_str());
        RCLCPP_INFO(get_logger(), "- Target frame: %s", target_frame_.c_str());
        RCLCPP_INFO(get_logger(), "- Processing every %d-th message", processing_interval_);
        RCLCPP_INFO(get_logger(), "- Voxel grid size: %.3f meters", voxel_size_);
        RCLCPP_INFO(get_logger(), "- Min points per voxel: %d", min_points_per_voxel_);
        RCLCPP_INFO(get_logger(), "- Min distance from LiDAR: %.2f meters", min_distance_);
        RCLCPP_INFO(get_logger(), "- Max distance from LiDAR: %.2f meters", max_distance_);
        RCLCPP_INFO(get_logger(), "- Output directory: %s", output_directory_.c_str());
        RCLCPP_INFO(get_logger(), "- Save format: %s", save_format_.c_str());
        RCLCPP_INFO(get_logger(), "- Will merge and save final point cloud when publisher stops");
        RCLCPP_INFO(get_logger(), "- Filter outliers: %s", filter_outliers_ ? "true" : "false");
        if (filter_outliers_) {
            RCLCPP_INFO(get_logger(), "  - Outlier radius: %.2f m, min neighbors: %d", outlier_radius_, min_neighbors_);
        }
    }
    
    ~PointCloudMergerNode() 
    {
        // Process and save the final point cloud
        if (!final_processing_done_) {
            process_and_save_merged_cloud();
        }
    }

private:
    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) 
    {
        // Store the latest pose
        latest_pose_ = *msg;
        have_pose_ = true;
    }
    
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        // Update the last message time
        last_message_time_ = get_clock()->now();
        
        // Increment counter and process only every Nth message
        message_counter_++;
        if (message_counter_ % processing_interval_ != 0) {
            return;
        }
        
        // Make sure we have received at least one pose message
        if (!have_pose_) {
            RCLCPP_WARN(get_logger(), "Waiting for pose message before processing point cloud");
            return;
        }
        
        RCLCPP_INFO(get_logger(), "Processing point cloud message %lu", message_counter_);
        
        try {
            geometry_msgs::msg::TransformStamped tf_lidar_to_base;
            geometry_msgs::msg::TransformStamped tf_map_to_viewer;
            
            // Look up the transform from lidar to base_link using the point cloud timestamp
            try {
                tf_lidar_to_base = tf_buffer_->lookupTransform(
                    base_frame_, lidar_frame_, msg->header.stamp, tf2::durationFromSec(0.1));
            } catch (tf2::TransformException &ex) {
                RCLCPP_ERROR(get_logger(), "Failed to lookup transform from %s to %s: %s", 
                           lidar_frame_.c_str(), base_frame_.c_str(), ex.what());
                return;
            }
            
            // Look up the transform from map to viewer using the point cloud timestamp
            if (target_frame_ != "map") {
                try {
                    tf_map_to_viewer = tf_buffer_->lookupTransform(
                        target_frame_, "map", msg->header.stamp, tf2::durationFromSec(0.1));
                } catch (tf2::TransformException &ex) {
                    RCLCPP_WARN(get_logger(), "Failed to lookup transform from map to %s: %s", 
                               target_frame_.c_str(), ex.what());
                    RCLCPP_WARN(get_logger(), "Will use map frame for this point cloud");
                    std::string original_target_frame = target_frame_;
                    target_frame_ = "map";
                    defer_target_frame_restore_ = true;
                    deferred_target_frame_ = original_target_frame;
                }
            }
            
            // Convert ROS PointCloud2 to PCL point cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);
            
            if (cloud->empty()) {
                RCLCPP_WARN(get_logger(), "Received empty point cloud");
                if (defer_target_frame_restore_) {
                    target_frame_ = deferred_target_frame_;
                    defer_target_frame_restore_ = false;
                }
                return;
            }
            
            // Remove NaN points from the cloud
            std::vector<int> nan_indices;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, nan_indices);
            
            if (!nan_indices.empty()) {
                RCLCPP_INFO(get_logger(), "Removed %zu NaN points from cloud", nan_indices.size());
            }
            
            // Filter points with infinite values
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_inf(new pcl::PointCloud<pcl::PointXYZ>);
            size_t inf_points = 0;
            
            for (const auto& point : cloud_filtered->points) {
                if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                    cloud_no_inf->points.push_back(point);
                } else {
                    inf_points++;
                }
            }
            
            if (inf_points > 0) {
                RCLCPP_INFO(get_logger(), "Removed %zu points with infinite values", inf_points);
                cloud_no_inf->width = cloud_no_inf->points.size();
                cloud_no_inf->height = 1;
                cloud_no_inf->is_dense = true;
            } else {
                cloud_no_inf = cloud_filtered;
            }
            
            // Filter points that are too close or too far from the LiDAR
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_distance_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            float min_distance_squared = min_distance_ * min_distance_;
            float max_distance_squared = max_distance_ * max_distance_;
            size_t close_points = 0;
            size_t far_points = 0;
            
            for (const auto& point : cloud_no_inf->points) {
                // Calculate squared distance from origin (LiDAR position)
                float distance_squared = point.x * point.x + point.y * point.y + point.z * point.z;
                
                // Keep points that are between min and max distance
                if (distance_squared > min_distance_squared && distance_squared < max_distance_squared) {
                    cloud_distance_filtered->points.push_back(point);
                } else if (distance_squared <= min_distance_squared) {
                    close_points++;
                } else {
                    far_points++;
                }
            }
            
            cloud_distance_filtered->width = cloud_distance_filtered->points.size();
            cloud_distance_filtered->height = 1;
            cloud_distance_filtered->is_dense = true;
            
            if (close_points > 0 || far_points > 0) {
                if (close_points > 0) {
                    RCLCPP_INFO(get_logger(), "Removed %zu points closer than %.2f meters to LiDAR", 
                               close_points, min_distance_);
                }
                if (far_points > 0) {
                    RCLCPP_INFO(get_logger(), "Removed %zu points farther than %.2f meters from LiDAR", 
                               far_points, max_distance_);
                }
                RCLCPP_INFO(get_logger(), "Kept %zu points within range", cloud_distance_filtered->size());
            } else {
                RCLCPP_INFO(get_logger(), "All %zu points are within distance range", cloud_distance_filtered->size());
            }
            
            // STEP 1: Transform from LiDAR frame to base_link using the fresh transform
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_base(new pcl::PointCloud<pcl::PointXYZ>);
            Eigen::Isometry3d lidar_to_base = tf2::transformToEigen(tf_lidar_to_base.transform);
            pcl::transformPointCloud(*cloud_distance_filtered, *cloud_in_base, lidar_to_base.matrix().cast<float>());
            
            // STEP 2: Get the transform from base_link to map from the pose message
            Eigen::Isometry3d base_to_map = Eigen::Isometry3d::Identity();
            
            Eigen::Vector3d position(
                latest_pose_.pose.pose.position.x,
                latest_pose_.pose.pose.position.y,
                latest_pose_.pose.pose.position.z);
                
            Eigen::Quaterniond orientation(
                latest_pose_.pose.pose.orientation.w,
                latest_pose_.pose.pose.orientation.x,
                latest_pose_.pose.pose.orientation.y,
                latest_pose_.pose.pose.orientation.z);
                
            orientation.normalize();
                
            base_to_map.translation() = position;
            base_to_map.linear() = orientation.toRotationMatrix();
            
            // STEP 3: Transform from base_link to map
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_map(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*cloud_in_base, *cloud_in_map, base_to_map.matrix().cast<float>());
            
            // STEP 4: If target frame is different from map, apply the map-to-viewer transform
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud;
            if (target_frame_ != "map") {
                transformed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                Eigen::Isometry3d map_to_viewer = tf2::transformToEigen(tf_map_to_viewer.transform);
                pcl::transformPointCloud(*cloud_in_map, *transformed_cloud, map_to_viewer.matrix().cast<float>());
            } else {
                transformed_cloud = cloud_in_map;
            }
            
            // Perform radius outlier removal
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_outliers;
            if (filter_outliers_) {
                cloud_no_outliers = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_filter;
                outlier_filter.setInputCloud(transformed_cloud);
                outlier_filter.setRadiusSearch(outlier_radius_);
                outlier_filter.setMinNeighborsInRadius(min_neighbors_);
                outlier_filter.filter(*cloud_no_outliers);
                
                size_t removed_outliers = transformed_cloud->size() - cloud_no_outliers->size();
                if (removed_outliers > 0) {
                    RCLCPP_INFO(get_logger(), "Removed %zu outlier points", removed_outliers);
                }
            } else {
                cloud_no_outliers = transformed_cloud;
            }
            
            // Add to the raw combined cloud 
            *raw_combined_cloud_ += *cloud_no_outliers;
            
            // Count points
            total_points_processed_ += cloud->size();
            RCLCPP_INFO(get_logger(), "Raw combined point cloud now has %lu points", raw_combined_cloud_->size());
            
            // Restore target_frame_ if it was temporarily changed
            if (defer_target_frame_restore_) {
                target_frame_ = deferred_target_frame_;
                defer_target_frame_restore_ = false;
            }
            
        } catch (const std::exception &e) {
            RCLCPP_ERROR(get_logger(), "Error processing point cloud: %s", e.what());
            
            // Restore target_frame_ if it was temporarily changed
            if (defer_target_frame_restore_) {
                target_frame_ = deferred_target_frame_;
                defer_target_frame_restore_ = false;
            }
        }
    }
    
    void check_publisher() 
    {
        // If we haven't received a message for a while, check if publisher is still active
        auto now = get_clock()->now();
        auto time_since_last_message = now - last_message_time_;
        
        if (time_since_last_message.seconds() > check_publisher_timeout_ && message_counter_ > 0) {
            // Check if there are any publishers
            auto topic_names_and_types = get_topic_names_and_types();
            bool found_publisher = false;
            
            for (const auto& topic_and_types : topic_names_and_types) {
                if (topic_and_types.first == pointcloud_topic_) {
                    // Now check if there are publishers
                    auto count_publishers_result = count_publishers(pointcloud_topic_);
                    if (count_publishers_result > 0) {
                        found_publisher = true;
                        break;
                    }
                }
            }
            
            if (!found_publisher) {
                RCLCPP_INFO(get_logger(), "No publishers found for topic %s. Processing final point cloud.", 
                          pointcloud_topic_.c_str());
                process_and_save_merged_cloud();
                
                // rclcpp::shutdown();
            }
        }
    }
    
    void process_and_save_merged_cloud() 
    {
        if (raw_combined_cloud_->empty()) {
            RCLCPP_WARN(get_logger(), "No points to process and save");
            return;
        }
        
        // Mark as done to avoid processing again in destructor
        final_processing_done_ = true;
        
        RCLCPP_INFO(get_logger(), "Processing final combined point cloud with %lu points", raw_combined_cloud_->size());
        
        try {
            // Apply voxel grid filter with minimum points per voxel
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(raw_combined_cloud_);
            voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
            voxel_filter.setMinimumPointsNumberPerVoxel(min_points_per_voxel_);
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            voxel_filter.filter(*filtered_cloud);
            
            RCLCPP_INFO(get_logger(), "Final filtered point cloud has %lu points", filtered_cloud->size());
            
            // Create filename
            std::string base_filename = output_directory_ + "/" + output_filename_;
            
            bool saved = false;
            
            // Save based on the selected format
            if (save_format_ == "pcd" || save_format_ == "both") {
                std::string pcd_filename = base_filename + ".pcd";
                pcl::io::savePCDFileBinary(pcd_filename, *filtered_cloud);
                RCLCPP_INFO(get_logger(), "- Saved to %s", pcd_filename.c_str());
                saved = true;
            }
            
            if (save_format_ == "ply" || save_format_ == "both") {
                std::string ply_filename = base_filename + ".ply";
                pcl::io::savePLYFileBinary(ply_filename, *filtered_cloud);
                RCLCPP_INFO(get_logger(), "- Saved to %s", ply_filename.c_str());
                saved = true;
            }
            
            if (saved) {
                RCLCPP_INFO(get_logger(), "Saved final point cloud with %lu points", filtered_cloud->size());
                RCLCPP_INFO(get_logger(), "Total points processed: %lu", total_points_processed_);
            }
            
        } catch (const std::exception &e) {
            RCLCPP_ERROR(get_logger(), "Error processing and saving point cloud: %s", e.what());
        }
    }
    
    // Class members
    std::string pointcloud_topic_;
    std::string pose_topic_;
    std::string lidar_frame_;
    std::string base_frame_;
    std::string target_frame_;
    int processing_interval_;
    double voxel_size_;
    int min_points_per_voxel_;
    double min_distance_;
    double max_distance_;
    std::string output_directory_;
    std::string output_filename_;
    double check_publisher_timeout_;
    std::string save_format_;
    bool filter_outliers_;
    double outlier_radius_;
    int min_neighbors_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_subscription_;
    rclcpp::TimerBase::SharedPtr publisher_check_timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_combined_cloud_;
    uint64_t message_counter_ = 0;
    uint64_t total_points_processed_ = 0;
    rclcpp::Time last_message_time_;
    bool final_processing_done_ = false;
    
    geometry_msgs::msg::PoseWithCovarianceStamped latest_pose_;
    bool have_pose_;
    
    bool defer_target_frame_restore_ = false;
    std::string deferred_target_frame_;
};

int main(int argc, char *argv[]) 
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudMergerNode>();
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}