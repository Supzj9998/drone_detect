#ifndef DRONE_DETECT_GUIDE_MANAGER_H
#define DRONE_DETECT_GUIDE_MANAGER_H

#include <mutex>
#include <string>

#include "base_interface/msg/polar3f.hpp"
#include "gary_msgs/msg/auto_aim.hpp"
#include "rclcpp/rclcpp.hpp"

namespace drone::guide_manager {

class GuideManagerNode final : public rclcpp::Node {
public:
    explicit GuideManagerNode(const rclcpp::NodeOptions& options);

private:
    void guideCallback(const base_interface::msg::Polar3f::SharedPtr msg);
    void statusCallback(const gary_msgs::msg::AutoAIM::SharedPtr msg);
    void timerCallback();

    rclcpp::Subscription<base_interface::msg::Polar3f>::SharedPtr
        guide_sub_;
    rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr
        status_sub_;
    rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr target_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    mutable std::mutex mutex_;
    float              latest_guide_yaw_rad_ = 0.0F;
    float              latest_guide_pitch_rad_ = 0.0F;
    rclcpp::Time       latest_guide_stamp_;
    bool               guide_ready_ = false;
    float              current_yaw_rad_ = 0.0F;
    float              current_pitch_rad_ = 0.0F;
    bool               status_ready_ = false;

    double guide_timeout_sec_ = 0.3;
    double yaw_offset_rad_ = 0.0;
    double pitch_offset_rad_ = 0.0;
    double yaw_gain_ = 1.0;
    double pitch_gain_ = 1.0;
    bool   require_status_ = true;
    int    autoaim_target_id_ = 0;
    int    autoaim_vision_mode_ = gary_msgs::msg::AutoAIM::VISION_MODE_ARMOR;
};

}  // namespace drone::guide_manager

#endif  // DRONE_DETECT_GUIDE_MANAGER_H
