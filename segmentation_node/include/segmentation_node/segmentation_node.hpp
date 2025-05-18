#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>

#include <onnxruntime_cxx_api.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <nlohmann/json.hpp>
#include <omp.h>

struct CameraInfo {
    // Original camera info parameters
    int height;
    int width;
    int undistorted_width;
    int undistorted_height;
    std::string distortion_model;
    std::vector<double> d;
    
    // Camera intrinsics in Eigen matrix format
    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Matrix<double, 3, 4> P;
    Eigen::Matrix3d undistorted_K;
    Eigen::Matrix3d undistorted_R;
    Eigen::Matrix<double, 3, 4> undistorted_P;
    
    // Common camera parameters for quick access
    double fx, fy;
    double cx, cy;
    double undistorted_fx, undistorted_fy;
    double undistorted_cx, undistorted_cy;
    
    // Get camera info from ROS message
    void fromMsg(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg);
    
    // Check if the camera info is valid
    bool isValid() const;

    // Calculate undistorted camera parameters
    void calculate_undistorted_params(int resized_width = 0, int resized_height = 0);

    // Project 3D point to 2D pixel coordinates
    bool point2pixel_undistorted(const Eigen::Vector3d& point, int& u, int& v) const;
};


struct ClassesWithColors {
    std::vector<std::vector<int>> colors;
    std::vector<std::string> names;

    void fromJSON(const std::string& json_string);
};

class SegmentationNode : public rclcpp::Node
{
public:
    SegmentationNode();
    ~SegmentationNode();

private:
    // Node parameters
    std::string camera_frame_;
    std::string lidar_frame_;
    std::string camera_image_topic_;
    std::string camera_info_topic_;
    std::string lidar_pointcloud_topic_;
    std::string output_topic_;
    std::string model_path_;

    // Camera info and status
    CameraInfo camera_info_;
    bool camera_info_received_ = false;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_point_cloud_pub_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> point_cloud_sub_;

    // QoS settings
    rclcpp::QoS qos_;

    // Synchronization
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::PointCloud2> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
    
    // TF2 buffer and listener
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ONNX Runtime variables
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};

    // Store model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;

    // Transform variables
    Eigen::Isometry3d lidar_to_camera_tf_;

    // Image and point cloud data
    cv::Mat image_;
    using InputPointType = pcl::PointXYZ; 
    using OutputPointType = pcl::PointXYZRGB;
    pcl::PointCloud<InputPointType>::Ptr point_cloud_;
    pcl::PointCloud<OutputPointType>::Ptr segmented_point_cloud_;
    sensor_msgs::msg::PointCloud2::SharedPtr segmented_point_cloud_msg_;

    // Classes and colors
    ClassesWithColors classes_with_colors_;

    // Functions
    void initialize_onnx_session();
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void synchronized_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg);
    void run_segmentaion();
    bool get_transform(const std::string& target_frame, const std::string& source_frame, 
                      Eigen::Isometry3d& transform);
    void preprocess_image(const sensor_msgs::msg::Image::ConstSharedPtr &image_msg);
    void postprocess_output(const std::vector<Ort::Value>& output_tensors);
};