#ifndef DRONE_PNP__PNP_H_
#define DRONE_PNP__PNP_H_

#include <cstddef>
#include <mutex>

#include "base_interface/msg/polar3f.hpp"
#include "gary_msgs/msg/auto_aim.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

namespace drone::pnp {

class PnpNode : public rclcpp::Node {
public:
    // 构造函数
    explicit PnpNode(const rclcpp::NodeOptions& options);

private:
    // 本帧pnp所需快照
    struct SolverState {
        // 相机内参矩阵
        cv::Mat camera_matrix;
        // 畸变参数
        cv::Mat dist_coeffs;
        // 相机坐标系到激光发射器的旋转矩阵
        cv::Matx33d r_laser_camera{cv::Matx33d::eye()};
        // 当前云台pitch
        float pitch{0.0F};
        // 当前云台yaw
        float yaw{0.0F};
        // 表示是否需要将得到的相机平移向量转换到外参指定的坐标系
        bool rotate_to_laser{false};
    };

    // yolo框回调
    void
    boxesCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    // 当前云台姿态角回调
    void
    autoaimStatusCallback(const gary_msgs::msg::AutoAIM::SharedPtr msg);
    // 相机参数回调
    void
    cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    // 找置信度最高的目标框
    bool findBestBox(const std_msgs::msg::Float32MultiArray& msg,
                     size_t&                                 best) const;
    // 检查pnp所需参数是否就绪
    bool getSolverState(SolverState& state) const;
    // pnp
    bool solveBox(const std_msgs::msg::Float32MultiArray& msg, size_t box,
                  const SolverState& state, cv::Vec3d& tvec) const;
    // 消息发布
    void publishResult(const cv::Vec3d& tvec, const SolverState& state);

    // yolo框订阅器
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
        boxes_sub_;
    // 云台状态订阅器
    rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr
        autoaim_status_sub_;
    // 相机参数订阅器
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        camera_info_sub_;
    // 极坐标结果发布器
    rclcpp::Publisher<base_interface::msg::Polar3f>::SharedPtr polar_pub_;
    // 下位机发布器
    rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_pub_;

    mutable std::mutex mutex_;
    cv::Mat            camera_matrix_;
    cv::Mat            dist_coeffs_;
    cv::Matx33d        r_laser_camera_{cv::Matx33d::eye()};
    // 类别筛选
    int target_class_id_{-1};
    // 发布到AutoAIM消息里的id
    int autoaim_target_id_{0};
    // 发布到AutoAIM消息里的视觉模式
    int autoaim_vision_mode_{gary_msgs::msg::AutoAIM::VISION_MODE_ARMOR};
    // 是否订阅并使用当前云台角
    bool use_autoaim_status_{true};
    // 是否强制要求先收到当前云台角再进行pnp
    bool require_autoaim_status_{true};
    // 是否允许发射
    bool allow_shoot_{false};
    // 控制发布yaw/pitch的单位，true:角度制，false:弧度制
    bool output_in_degrees_{true};
    // 表示输入检测框来自的图像是否已经去畸变
    bool input_is_undistorted_{false};
    // 有没有相机内参
    bool camera_info_ready_{false};
    // 有没有启用外参旋转
    bool extrinsic_ready_{false};
    // 有没有收到云台pitch/yaw状态
    bool autoaim_status_ready_{false};
    // 用的目标真实尺寸
    double target_width_m_{0.072};
    double target_height_m_{0.050};
    // 当前云台pitch和yaw
    float current_pitch_rad_{0.0F};
    float current_yaw_rad_{0.0F};
};

}  // namespace drone::pnp

#endif  // DRONE_PNP__PNP_H_
