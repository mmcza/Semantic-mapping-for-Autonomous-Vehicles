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

struct CameraInfo {
    // Original camera info parameters
    int height;
    int width;
    std::string distortion_model;
    std::vector<double> d;
    
    // Camera intrinsics in Eigen matrix format
    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Matrix<double, 3, 4> P;
    
    // Common camera parameters for quick access
    double fx, fy;
    double cx, cy;
    
    // Utility methods
    void fromMsg(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
        // Store basic info
        height = msg->height;
        width = msg->width;
        distortion_model = msg->distortion_model;
        d = msg->d;
        
        // Convert K (intrinsic) to Eigen format
        K = Eigen::Matrix3d::Zero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                K(i, j) = msg->k[i*3 + j];
            }
        }
        
        // Convert R (rectification) to Eigen format
        R = Eigen::Matrix3d::Zero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R(i, j) = msg->r[i*3 + j];
            }
        }
        
        // Convert P (projection) to Eigen format
        P = Eigen::Matrix<double, 3, 4>::Zero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                P(i, j) = msg->p[i*4 + j];
            }
        }
        
        fx = K(0, 0);
        fy = K(1, 1);
        cx = K(0, 2);
        cy = K(1, 2);
    }
    
    // Check if the camera info is valid
    bool isValid() const {
        return fx > 0 && fy > 0 && width > 0 && height > 0;
    }
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

    // Functions
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void synchronized_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg);
        
    bool get_transform(const std::string& target_frame, const std::string& source_frame, 
                      Eigen::Isometry3d& transform);
    void preprocess_image(const sensor_msgs::msg::Image::SharedPtr &image_msg);
    void postprocess_output();
};