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
  void boxesCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  // 当前云台姿态角回调
  void autoaimStatusCallback(const gary_msgs::msg::AutoAIM::SharedPtr msg);
  // 相机参数回调
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  // 找置信度最高的目标框
  bool findBestBox(const std_msgs::msg::Float32MultiArray& msg, size_t& best) const;
  // 检查pnp所需参数是否就绪
  bool getSolverState(SolverState& state) const;
  // pnp
  bool solveBox(const std_msgs::msg::Float32MultiArray& msg, size_t box,
                const SolverState& state, cv::Vec3d& tvec) const;
  // 消息发布
  void publishResult(const cv::Vec3d& tvec, const SolverState& state);
  
  // yolo框订阅器
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr boxes_sub_;
  // 云台状态订阅器
  rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_status_sub_;
  // 相机参数订阅器
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  // 极坐标结果发布器
  rclcpp::Publisher<base_interface::msg::Polar3f>::SharedPtr polar_pub_;
  // 下位机发布器
  rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_pub_;

  mutable std::mutex mutex_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Matx33d r_laser_camera_{cv::Matx33d::eye()};
  int target_class_id_{-1};
  int autoaim_target_id_{0};
  int autoaim_vision_mode_{gary_msgs::msg::AutoAIM::VISION_MODE_ARMOR};
  bool use_autoaim_status_{true};
  bool require_autoaim_status_{true};
  bool allow_shoot_{false};
  bool output_in_degrees_{true};
  bool input_is_undistorted_{false};
  bool camera_info_ready_{false};
  bool extrinsic_ready_{false};
  bool autoaim_status_ready_{false};
  double target_width_m_{0.072};
  double target_height_m_{0.050};
  float current_pitch_rad_{0.0F};
  float current_yaw_rad_{0.0F};
};

}  // namespace drone::pnp

#endif  // DRONE_PNP__PNP_H_
