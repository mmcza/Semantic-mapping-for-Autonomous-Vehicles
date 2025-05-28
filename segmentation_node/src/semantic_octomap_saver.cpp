#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>
#include <memory>
#include <string>

class SemanticOctomapSaver : public rclcpp::Node
{
public:
    SemanticOctomapSaver(const std::string &output_file)
        : Node("semantic_octomap_saver"), output_file_(output_file)
    {
        point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/octomap_point_cloud_centers", 10, std::bind(&SemanticOctomapSaver::point_cloud_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::seconds(1), std::bind(&SemanticOctomapSaver::timer_callback, this));

        last_message_time_ = this->now().seconds();

        RCLCPP_INFO(this->get_logger(), "SemanticOctomapSaver initialized, waiting for messages...");
    }

private:
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        last_message_ = msg;
        last_message_time_ = this->now().seconds();
    }

    void timer_callback()
    {
        double current_time = this->now().seconds();
        if (last_message_ && (current_time - last_message_time_ > 10.0)) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::fromROSMsg(*last_message_, *cloud);

            if (pcl::io::savePCDFile(output_file_, *cloud) == 0) {
                RCLCPP_INFO(this->get_logger(), "Successfully saved %zu points to %s", cloud->size(), output_file_.c_str());
                rclcpp::shutdown();
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save to %s", output_file_.c_str());
            }
        }
    }
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_message_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::string output_file_;
    double last_message_time_;
};

int main(int argc, char **argv)
{
    if (argc < 2) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Usage: ros2 run segmentation_node semantic_octomap_saver <output_file>");
        return 1;
    }
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SemanticOctomapSaver>(argv[1]);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}