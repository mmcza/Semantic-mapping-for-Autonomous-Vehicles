#include "segmentation_node/segmentation_node.hpp"

SegmentationNode::SegmentationNode() 
    : Node("segmentation_node"), 
      env_(ORT_LOGGING_LEVEL_WARNING, "segmentation_session"),
      qos_(10)
{
    // Declare and get parameters
    declare_parameter("camera_frame", "camera_front_optical_link");
    declare_parameter("lidar_frame", "lidar_laser_top_link");
    declare_parameter("camera_image_topic", "/sensing/camera/front/image_raw");
    declare_parameter("camera_info_topic", "/sensing/camera/front/camera_info");
    declare_parameter("lidar_pointcloud_topic", "/sensing/lidar/top/pointcloud");
    declare_parameter("segmented_pointcloud_topic", "/sensing/lidar/segmented_pointcloud");
    declare_parameter("model_path", "/root/Shared/saved_model/model.onnx");
    
    // Get parameters
    camera_frame_ = get_parameter("camera_frame").as_string();
    lidar_frame_ = get_parameter("lidar_frame").as_string();
    camera_image_topic_ = get_parameter("camera_image_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    lidar_pointcloud_topic_ = get_parameter("lidar_pointcloud_topic").as_string();
    output_topic_ = get_parameter("segmented_pointcloud_topic").as_string();
    model_path_ = get_parameter("model_path").as_string();

    RCLCPP_INFO(get_logger(), "Model path: %s", model_path_.c_str());
    RCLCPP_INFO(get_logger(), "Camera frame: %s", camera_frame_.c_str());
    RCLCPP_INFO(get_logger(), "LiDAR frame: %s", lidar_frame_.c_str());
    
    // Initialize TF2 components
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Initialize proper QoS settings for the camera info subscriber
    qos_.best_effort();
    qos_.durability_volatile();

    // Get the camera info
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, qos_,
        std::bind(&SegmentationNode::camera_info_callback, this, std::placeholders::_1));
    
    RCLCPP_INFO(get_logger(), "Waiting for camera info...");
}

SegmentationNode::~SegmentationNode() 
{
}

void SegmentationNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    if (camera_info_received_) {
        return;  // We only need to get the camera info once
    }
    
    // Store camera info to the struct
    camera_info_.fromMsg(msg);
    camera_info_received_ = true;
    
    RCLCPP_INFO(get_logger(), "Camera info received: resolution %dx%d, fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                camera_info_.width, camera_info_.height, 
                camera_info_.fx, camera_info_.fy,
                camera_info_.cx, camera_info_.cy);
    
    // Set up synchronization for image and point cloud
    image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        this, camera_image_topic_, qos_.get_rmw_qos_profile());
    point_cloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        this, lidar_pointcloud_topic_, qos_.get_rmw_qos_profile());
    
    // Initialize synchronizer
    synchronizer_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *image_sub_, *point_cloud_sub_);
    
    // Register callback
    synchronizer_->registerCallback(
        std::bind(&SegmentationNode::synchronized_callback, this, 
                 std::placeholders::_1, std::placeholders::_2));
    
    // Free the camera info subscriber
    camera_info_sub_.reset();
    
    RCLCPP_INFO(get_logger(), "Synchronization setup complete.");
}

void SegmentationNode::synchronized_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
{
    // TODO: Implement synchronized callback
    RCLCPP_INFO(get_logger(), "Received synchronized messages");
}

bool SegmentationNode::get_transform(
    const std::string& target_frame, 
    const std::string& source_frame,
    Eigen::Isometry3d& transform)
{
    try {
        geometry_msgs::msg::TransformStamped transform_stamped = 
            tf_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
        
        transform = tf2::transformToEigen(transform_stamped);
        return true;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(get_logger(), "Could not transform %s to %s: %s", 
                    source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }
}

void SegmentationNode::preprocess_image(
    const sensor_msgs::msg::Image::SharedPtr &image_msg)
{
    // TODO: Implement image preprocessing
}

void SegmentationNode::postprocess_output()
{
    // TODO: Implement output postprocessing
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegmentationNode>());
    rclcpp::shutdown();
    return 0;
}