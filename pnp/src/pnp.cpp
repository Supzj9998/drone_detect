#include "pnp.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "opencv2/calib3d.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace drone::pnp {
namespace {
    // 一个检测框占10个float
    constexpr size_t kBoxStride = 10U;
    // 弧度转角度系数
    constexpr double          kRadToDeg = 57.29577951308232;
    constexpr const char*     kPolarTopic = "pnp/polar";
    constexpr const char*     kAutoaimTopic = "/autoaim/target";
    constexpr const char*     kBoxesTopic = "detect/boxes";
    constexpr const char*     kCameraInfoTopic = "camera_info";
    constexpr const char*     kAutoaimStatusTopic = "/autoaim/status";
    const std::vector<double> kLaserRpyDeg{0.0, 0.0, 0.0};

    // 角度转换成旋转矩阵
    cv::Matx33d rpyDegToMat(const std::vector<double>& rpy_deg)
    {
        if (rpy_deg.size() != 3U) {
            return cv::Matx33d::eye();
        }

        const double roll = rpy_deg[0] / kRadToDeg;
        const double pitch = rpy_deg[1] / kRadToDeg;
        const double yaw = rpy_deg[2] / kRadToDeg;
        const double cr = std::cos(roll);
        const double sr = std::sin(roll);
        const double cp = std::cos(pitch);
        const double sp = std::sin(pitch);
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        return cv::Matx33d(cy * cp, cy * sp * sr - sy * cr,
                           cy * sp * cr + sy * sr, sy * cp,
                           sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                           -sp, cp * sr, cp * cr);
    }

}  // namespace

PnpNode::PnpNode(const rclcpp::NodeOptions& options)
    : Node("pnp_node", options)
{
    r_laser_camera_ = rpyDegToMat(kLaserRpyDeg);
    extrinsic_ready_ = true;

    // pnp发布器
    polar_pub_ =
        create_publisher<base_interface::msg::Polar3f>(kPolarTopic, 10);
    // autoaim发布器
    autoaim_pub_ =
        create_publisher<gary_msgs::msg::AutoAIM>(kAutoaimTopic, 10);
    // 检测框话题订阅器
    boxes_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        kBoxesTopic, rclcpp::SensorDataQoS(),
        std::bind(&PnpNode::boxesCallback, this, std::placeholders::_1));
    // 相机内参话题订阅器
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        kCameraInfoTopic, rclcpp::SensorDataQoS(),
        std::bind(&PnpNode::cameraInfoCallback, this,
                  std::placeholders::_1));
    if (use_autoaim_status_) {
        // 云台当前状态订阅器
        autoaim_status_sub_ = create_subscription<gary_msgs::msg::AutoAIM>(
            kAutoaimStatusTopic, rclcpp::SensorDataQoS(),
            std::bind(&PnpNode::autoaimStatusCallback, this,
                      std::placeholders::_1));
    }
}

// 检测框回调
void PnpNode::boxesCallback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    size_t      best = 0U;
    SolverState state;
    cv::Vec3d   tvec;
    // 依次执行三个函数，如有执行失败的直接返回
    if (!findBestBox(*msg, best) || !getSolverState(state) ||
        !solveBox(*msg, best, state, tvec)) {
        return;
    }
    // 解算成功后发布结果
    publishResult(tvec, state);
}

// 找最佳框
bool PnpNode::findBestBox(const std_msgs::msg::Float32MultiArray& msg,
                          size_t& best) const
{
    if (msg.data.empty() || msg.data.size() % kBoxStride != 0U) {
        return false;
    }
    best = msg.data.size();
    // 最小可接受置信度设为0.6
    float best_score = 0.6F;
    // 每10个float遍历一次
    for (size_t i = 0; i < msg.data.size(); i += kBoxStride) {
        // 取类别和置信度
        const int   class_id = static_cast<int>(msg.data[i]);
        const float score = msg.data[i + 1];
        // 取符合类别且置信度更高的框
        if ((target_class_id_ < 0 || class_id == target_class_id_) &&
            score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best != msg.data.size();
}

bool PnpNode::getSolverState(SolverState& state) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!camera_info_ready_ || camera_matrix_.empty() ||
        (use_autoaim_status_ && require_autoaim_status_ &&
         !autoaim_status_ready_)) {
        return false;
    }

    state.camera_matrix = camera_matrix_.clone();
    state.dist_coeffs = dist_coeffs_.clone();
    state.r_laser_camera = r_laser_camera_;
    state.rotate_to_laser = extrinsic_ready_;
    state.pitch = current_pitch_rad_;
    state.yaw = current_yaw_rad_;
    return true;
}

// 用最佳检测框做pnp
bool PnpNode::solveBox(const std_msgs::msg::Float32MultiArray& msg,
                       size_t box, const SolverState& state,
                       cv::Vec3d& tvec) const
{
    const float half_w = static_cast<float>(target_width_m_ * 0.5);
    const float half_h = static_cast<float>(target_height_m_ * 0.5);

    // 定义物体坐标系
    const std::vector<cv::Point3f> object_points{{-half_w, -half_h, 0.0F},
                                                 {half_w, -half_h, 0.0F},
                                                 {half_w, half_h, 0.0F},
                                                 {-half_w, half_h, 0.0F}};
    // 定义图像上的四个角点
    const std::vector<cv::Point2f> image_points{
        {msg.data[box + 2], msg.data[box + 3]},
        {msg.data[box + 4], msg.data[box + 5]},
        {msg.data[box + 6], msg.data[box + 7]},
        {msg.data[box + 8], msg.data[box + 9]}};

    cv::Vec3d rvec;
    // pnp解算
    if (!cv::solvePnP(object_points, image_points, state.camera_matrix,
                      state.dist_coeffs, rvec, tvec, false,
                      cv::SOLVEPNP_ITERATIVE)) {
        return false;
    }
    if (state.rotate_to_laser) {
        tvec = state.r_laser_camera * tvec;
    }
    return true;
}

// 发布消息
void PnpNode::publishResult(const cv::Vec3d& tvec, const SolverState& state)
{
    // 把三维坐标转成角度距离并发布
    const double x = tvec[0];
    const double y = tvec[1];
    const double z = tvec[2];

    // 取相机坐标系坐标
    const double distance = std::sqrt(x * x + y * y + z * z);
    const double base_yaw = -std::atan2(x, z);
    const double base_pitch = -std::atan2(-y, std::sqrt(x * x + z * z));
    double       yaw = base_yaw;
    double       pitch = base_pitch;
    // 启用云台状态就叠加云台角
    if (use_autoaim_status_) {
        yaw += state.yaw;
        pitch += state.pitch;
    }
    // 创建并发布极坐标消息
    base_interface::msg::Polar3f polar;
    polar.yaw =
        static_cast<float>(output_in_degrees_ ? yaw * kRadToDeg : yaw);
    polar.pitch =
        static_cast<float>(output_in_degrees_ ? pitch * kRadToDeg : pitch);
    polar.distance = static_cast<float>(distance);
    polar_pub_->publish(polar);
    // AutoAIM keeps radians and preserves the previous doubled visual-angle
    // convention.
    if (autoaim_pub_) {
        // 创建消息并填信息
        gary_msgs::msg::AutoAIM autoaim;
        autoaim.header.stamp = get_clock()->now();
        autoaim.yaw = static_cast<float>(
            base_yaw * 2.0 + (use_autoaim_status_ ? state.yaw : 0.0F));
        autoaim.pitch = static_cast<float>(
            base_pitch * 2.0 + (use_autoaim_status_ ? state.pitch : 0.0F));
        autoaim.target_id =
            static_cast<uint8_t>(std::clamp(autoaim_target_id_, 0, 7));
        autoaim.target_distance = static_cast<float>(distance);
        autoaim.vision_mode =
            static_cast<uint8_t>(std::clamp(autoaim_vision_mode_, 1, 4));
        autoaim.shoot_command = allow_shoot_
                                    ? gary_msgs::msg::AutoAIM::ALLOW_SHOOT
                                    : gary_msgs::msg::AutoAIM::CEASE_FIRE;
        autoaim_pub_->publish(autoaim);
    }
}

void PnpNode::autoaimStatusCallback(
    const gary_msgs::msg::AutoAIM::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    current_pitch_rad_ = msg->pitch;
    current_yaw_rad_ = msg->yaw;
    autoaim_status_ready_ = true;
}
void PnpNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (msg->k[0] <= 0.0 || msg->k[4] <= 0.0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        camera_matrix_.at<double>(i / 3, i % 3) =
            msg->k[static_cast<size_t>(i)];
    }
    dist_coeffs_ =
        cv::Mat::zeros(static_cast<int>(msg->d.size()), 1, CV_64F);
    if (!input_is_undistorted_) {
        for (size_t i = 0; i < msg->d.size(); ++i) {
            dist_coeffs_.at<double>(static_cast<int>(i), 0) = msg->d[i];
        }
    }
    camera_info_ready_ = true;
}

}  // namespace drone::pnp

RCLCPP_COMPONENTS_REGISTER_NODE(drone::pnp::PnpNode)
