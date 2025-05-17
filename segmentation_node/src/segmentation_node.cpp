#include "segmentation_node/segmentation_node.hpp"

const char* GREEN = "\033[32m";
const char* RESET = "\033[0m";

SegmentationNode::SegmentationNode() 
    :   Node("segmentation_node"), 
        qos_(10),
        env_(ORT_LOGGING_LEVEL_WARNING, "segmentation_session")
{
    // Declare and get parameters
    declare_parameter("camera_frame", "camera_front_optical_link");
    declare_parameter("lidar_frame", "lidar_laser_top_link");
    declare_parameter("camera_image_topic", "/sensing/camera/front/image_raw");
    declare_parameter("camera_info_topic", "/sensing/camera/front/camera_info");
    declare_parameter("lidar_pointcloud_topic", "/sensing/lidar/top/pointcloud");
    declare_parameter("segmented_pointcloud_topic", "/sensing/lidar/segmented_pointcloud");
    declare_parameter("model_path", "/root/Shared/saved_models/model.onnx");
    
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

    // Initialize the publisher for segmented point cloud
    segmented_point_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        output_topic_, qos_);

    // Initialize the ONNX Runtime session
    initialize_onnx_session();

    // Get the camera info
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, qos_,
        std::bind(&SegmentationNode::camera_info_callback, this, std::placeholders::_1));
    
    // Initialize the point clouds
    point_cloud_ = std::make_shared<pcl::PointCloud<InputPointType>>();
    segmented_point_cloud_ = std::make_shared<pcl::PointCloud<OutputPointType>>();

    RCLCPP_INFO(get_logger(), "Waiting for information about camera...");
}

SegmentationNode::~SegmentationNode() 
{
}

void SegmentationNode::initialize_onnx_session()
{
    // Initialize the ONNX Runtime session with optimized settings
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    session_options_.SetLogId("SegmentationNode");
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Check for CUDA availability
    auto available_providers = Ort::GetAvailableProviders();
    bool cuda_available = false;
    for (const auto& provider : available_providers) {
        if (provider.find("CUDA") != std::string::npos) {
            cuda_available = true;
            break;
        }
    }

    if (cuda_available) {
        RCLCPP_INFO(get_logger(), "CUDA available for ONNX Runtime - using GPU");
        
        // Configure CUDA provider with optimized settings
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.do_copy_in_default_stream = 1;
        
        // Add CUDA provider to session options
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        RCLCPP_INFO(get_logger(), "CUDA provider added to ONNX Runtime session");
    } else {
        RCLCPP_INFO(get_logger(), "CUDA not available, using CPU for inference");
    }
    // Create the ONNX Runtime session with more error checking
    try {
        RCLCPP_INFO(get_logger(), "Attempting to load ONNX model from: %s", model_path_.c_str());
        session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);
        
        // Check if session was created successfully
        const Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        // Reserve space for the names
        input_names_.resize(num_input_nodes);
        output_names_.resize(num_output_nodes);

        RCLCPP_INFO(get_logger(), "%sONNX model loaded successfully from path: %s %s", GREEN, model_path_.c_str(), RESET);
        RCLCPP_INFO(get_logger(), "Number of inputs: %zu, Number of outputs: %zu", 
                    num_input_nodes, num_output_nodes);
        
        // Get input names and shapes
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_[i] = input_name.get(); // Store as std::string
            
            // Get input shape
            Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();
            
            RCLCPP_INFO(get_logger(), "Input %zu: %s, shape: [%s]", 
                        i, input_names_[i].c_str(),
                        [&]() {
                            std::stringstream ss;
                            for (size_t j = 0; j < input_shape_.size(); j++) {
                                ss << input_shape_[j];
                                if (j < input_shape_.size() - 1) ss << ", ";
                            }
                            return ss.str();
                        }().c_str());
        }

        // Get output names
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_[i] = output_name.get(); // Store as std::string
            RCLCPP_INFO(get_logger(), "Output %zu: %s", i, output_names_[i].c_str());
        }
        
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(get_logger(), "ONNX Runtime error: %s", e.what());
        throw;
    }

    if (!session_) {
        RCLCPP_ERROR(get_logger(), "Failed to create ONNX Runtime session");
        throw std::runtime_error("Failed to create ONNX Runtime session");
    }
    RCLCPP_INFO(get_logger(), "%sONNX Runtime session created successfully%s", GREEN, RESET);
}

void SegmentationNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    if (camera_info_received_) {
        return;
    }
    
    // Store camera info to the struct
    camera_info_.fromMsg(msg);
    camera_info_received_ = true;

    // Get width and height required by model
    int model_width = input_shape_[3];
    int model_height = input_shape_[2];
    camera_info_.calculate_undistorted_params(model_width, model_height);
    
    RCLCPP_INFO(get_logger(), "%sCamera info received%s:\nresolution %dx%d,\nfx=%.2f, fy=%.2f,\ncx=%.2f, cy=%.2f",
                GREEN, RESET,
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
    
    RCLCPP_INFO(get_logger(), "%sSynchronization setup complete.%s", GREEN, RESET);
}

void SegmentationNode::synchronized_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
{
    // Preprocess image
    preprocess_image(image_msg);

    // Convert point cloud to PCL format
    point_cloud_->clear();
    try {
        pcl::fromROSMsg(*cloud_msg, *point_cloud_);
        RCLCPP_INFO(get_logger(), "Point cloud converted with %zu points", point_cloud_->size());
    } catch (const pcl::PCLException& e) {
        RCLCPP_ERROR(get_logger(), "Error converting point cloud: %s", e.what());
        return;
    }

    // Get the transformation from lidar to camera frame
    get_transform(lidar_frame_, camera_frame_, lidar_to_camera_tf_);

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
        RCLCPP_WARN(get_logger(), "Could not find transformation from %s to %s: %s", 
                    source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }
}

void SegmentationNode::preprocess_image(
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg)
{
    // Convert ROS image to OpenCV format
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, image_msg->encoding);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Change encoding to RGB
    cv::Mat rgb_image;
    if (cv_ptr->encoding == sensor_msgs::image_encodings::BGR8) {
        cv::Mat bgr_image = cv_ptr->image;
        cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    } 
    else {
        rgb_image = cv_ptr->image;
    }

    // Undistort image using camera intrinsic parameters
    cv::Mat undistorted_image;
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = camera_info_.fx;
    K.at<double>(1, 1) = camera_info_.fy;
    K.at<double>(0, 2) = camera_info_.cx;
    K.at<double>(1, 2) = camera_info_.cy;
    cv::Mat dist_coeffs = cv::Mat(1, camera_info_.d.size(), CV_64F);
    for (size_t i = 0; i < camera_info_.d.size(); i++) {
        dist_coeffs.at<double>(0, i) = camera_info_.d[i];
    }
    cv::Mat new_K = cv::getOptimalNewCameraMatrix(
        K, dist_coeffs, cv::Size(rgb_image.cols, rgb_image.rows), 0.0);
    cv::undistort(rgb_image, undistorted_image, K, dist_coeffs, new_K);

    // Resize image to model input size
    cv::Mat resized_image;
    cv::resize(undistorted_image, resized_image, cv::Size(input_shape_[3], input_shape_[2]));

    // Normalize image with mean and std values from Imagenet
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;  // R channel
    channels[1] = (channels[1] - 0.456) / 0.224;  // G channel
    channels[2] = (channels[2] - 0.406) / 0.225;  // B channel
    cv::merge(channels, image_);
}

void SegmentationNode::postprocess_output()
{
    // TODO: Implement output postprocessing
}

void CameraInfo::fromMsg(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg){
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

bool CameraInfo::isValid() const {
    return fx > 0 && fy > 0 && width > 0 && height > 0;
}

void CameraInfo::calculate_undistorted_params(int resized_width, int resized_height) {
    // Convert to OpenCV format for undistortion
    cv::Mat K_cv = cv::Mat::eye(3, 3, CV_64F);
    K_cv.at<double>(0,0) = fx;
    K_cv.at<double>(1,1) = fy;
    K_cv.at<double>(0,2) = cx;
    K_cv.at<double>(1,2) = cy;
    
    cv::Mat dist_coeffs = cv::Mat(1, d.size(), CV_64F);
    for (size_t i = 0; i < d.size(); i++) {
        dist_coeffs.at<double>(0,i) = d[i];
    }
    
    // Get optimal new camera matrix 
    cv::Mat new_K = cv::getOptimalNewCameraMatrix(
        K_cv, dist_coeffs, cv::Size(width, height), 0.0);
    
    // Extract the undistorted parameters
    undistorted_fx = new_K.at<double>(0,0);
    undistorted_fy = new_K.at<double>(1,1);
    undistorted_cx = new_K.at<double>(0,2);
    undistorted_cy = new_K.at<double>(1,2);

    // If resizing is requested, adjust parameters
    if (resized_width > 0 && resized_height > 0) {
        double width_scale = static_cast<double>(resized_width) / width;
        double height_scale = static_cast<double>(resized_height) / height;
        
        // Scale intrinsic parameters
        undistorted_fx *= width_scale;
        undistorted_fy *= height_scale;
        undistorted_cx *= width_scale;
        undistorted_cy *= height_scale;
        undistorted_width = resized_width;
        undistorted_height = resized_height;
    }
    
    // Update undistorted K matrix
    undistorted_K = Eigen::Matrix3d::Zero();
    undistorted_K(0,0) = undistorted_fx;
    undistorted_K(1,1) = undistorted_fy;
    undistorted_K(0,2) = undistorted_cx;
    undistorted_K(1,2) = undistorted_cy;
    undistorted_K(2,2) = 1.0;
}

bool CameraInfo::point2pixel_undistorted(const Eigen::Vector3d& point, int& u, int& v) const {
    if (!isValid()) {
        return false;
    }
    
    // Check if point is in front of camera
    if (point.z() <= 0) {
        return false;
    }
    
    // Project 3D point to 2D image coordinates
    double x = point.x() / point.z();
    double y = point.y() / point.z();
    u = static_cast<int>(std::round(undistorted_fx * x + undistorted_cx));
    v = static_cast<int>(std::round(undistorted_fy * y + undistorted_cy));
    
    // Check if pixel is within image bounds
    return (u >= 0 && u < undistorted_width && v >= 0 && v < undistorted_height);
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegmentationNode>());
    rclcpp::shutdown();
    return 0;
}