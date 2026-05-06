#ifndef DRONE_DETECT_DRONE_DETECTER_H
#define DRONE_DETECT_DRONE_DETECTER_H

#include <memory>
#include <mutex>

#include "BaseInfer.hpp"
#include "base_interface/msg/polar3f.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolos.hpp"

namespace drone::drone_detecter {

class DroneDetecterNode final : public rclcpp::Node {
public:
    explicit DroneDetecterNode(const rclcpp::NodeOptions& options);

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void cameraInfoCallback(
        const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    yolo::BoxArray filterDetectionsByColor(
        const cv::Mat& image, const yolo::BoxArray& detections) const;
    bool getCameraIntrinsics(double& fx, double& fy, double& cx,
                             double& cy) const;
    void publishGuideTarget(const yolo::Box& box, double fx, double fy,
                            double cx, double cy) const;

    std::shared_ptr<tdt_radar::Infer<yolo::BoxArray>> yolo_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        camera_info_sub_;
    rclcpp::Publisher<base_interface::msg::Polar3f>::SharedPtr guide_pub_;

    mutable std::mutex mutex_;
    double             camera_fx_ = 0.0;
    double             camera_fy_ = 0.0;
    double             camera_cx_ = 0.0;
    double             camera_cy_ = 0.0;
    bool               camera_info_ready_ = false;

    // true表示当前需要蓝色目标，false表示当前需要红色目标。
    bool detect_blue_target_ = true;
    double guide_yaw_offset_rad_ = 0.0;
    double guide_pitch_offset_rad_ = 0.0;
};

}  // namespace drone::drone_detecter

#endif  // DRONE_DETECT_DRONE_DETECTER_H
