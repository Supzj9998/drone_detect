#include "guide_manager.h"

#include <algorithm>
#include <chrono>
#include <functional>

#include "rclcpp_components/register_node_macro.hpp"

namespace drone::guide_manager {
namespace {
    constexpr const char* kDefaultGuideTopic = "drone_detecter/guide_polar";
    constexpr const char* kDefaultStatusTopic = "/autoaim/status";
    constexpr const char* kDefaultTargetTopic = "/autoaim/target";
}  // namespace

GuideManagerNode::GuideManagerNode(const rclcpp::NodeOptions& options)
    : Node("guide_manager_node", options)
{
    const auto guide_topic =
        declare_parameter<std::string>("guide_topic", kDefaultGuideTopic);
    const auto status_topic =
        declare_parameter<std::string>("status_topic", kDefaultStatusTopic);
    const auto target_topic =
        declare_parameter<std::string>("target_topic", kDefaultTargetTopic);
    const auto publish_hz = declare_parameter<double>("publish_hz", 50.0);
    guide_timeout_sec_ =
        declare_parameter<double>("guide_timeout_sec", guide_timeout_sec_);
    yaw_offset_rad_ =
        declare_parameter<double>("yaw_offset_rad", yaw_offset_rad_);
    pitch_offset_rad_ =
        declare_parameter<double>("pitch_offset_rad", pitch_offset_rad_);
    yaw_gain_ = declare_parameter<double>("yaw_gain", yaw_gain_);
    pitch_gain_ = declare_parameter<double>("pitch_gain", pitch_gain_);
    require_status_ =
        declare_parameter<bool>("require_status", require_status_);
    autoaim_target_id_ =
        declare_parameter<int>("autoaim_target_id", autoaim_target_id_);
    autoaim_vision_mode_ =
        declare_parameter<int>("autoaim_vision_mode", autoaim_vision_mode_);

    guide_sub_ = create_subscription<base_interface::msg::Polar3f>(
        guide_topic, rclcpp::SensorDataQoS(),
        std::bind(&GuideManagerNode::guideCallback, this,
                  std::placeholders::_1));
    status_sub_ = create_subscription<gary_msgs::msg::AutoAIM>(
        status_topic, rclcpp::SensorDataQoS(),
        std::bind(&GuideManagerNode::statusCallback, this,
                  std::placeholders::_1));
    target_pub_ =
        create_publisher<gary_msgs::msg::AutoAIM>(target_topic, 10);

    const double safe_publish_hz = std::max(1.0, publish_hz);
    const auto timer_period = std::chrono::duration<double>(
        1.0 / safe_publish_hz);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
        std::bind(&GuideManagerNode::timerCallback, this));
}

void GuideManagerNode::guideCallback(
    const base_interface::msg::Polar3f::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    latest_guide_yaw_rad_ = msg->yaw;
    latest_guide_pitch_rad_ = msg->pitch;
    latest_guide_stamp_ = get_clock()->now();
    guide_ready_ = true;
}

void GuideManagerNode::statusCallback(
    const gary_msgs::msg::AutoAIM::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    current_yaw_rad_ = msg->yaw;
    current_pitch_rad_ = msg->pitch;
    status_ready_ = true;
}

void GuideManagerNode::timerCallback()
{
    float        guide_yaw = 0.0F;
    float        guide_pitch = 0.0F;
    float        current_yaw = 0.0F;
    float        current_pitch = 0.0F;
    rclcpp::Time guide_stamp;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!guide_ready_ || (require_status_ && !status_ready_)) {
            return;
        }

        guide_yaw = latest_guide_yaw_rad_;
        guide_pitch = latest_guide_pitch_rad_;
        current_yaw = current_yaw_rad_;
        current_pitch = current_pitch_rad_;
        guide_stamp = latest_guide_stamp_;
    }

    if ((get_clock()->now() - guide_stamp).seconds() > guide_timeout_sec_) {
        return;
    }

    gary_msgs::msg::AutoAIM target;
    target.header.stamp = get_clock()->now();
    target.yaw = static_cast<float>(current_yaw + guide_yaw * yaw_gain_ +
                                    yaw_offset_rad_);
    target.pitch = static_cast<float>(current_pitch +
                                      guide_pitch * pitch_gain_ +
                                      pitch_offset_rad_);
    target.target_id =
        static_cast<uint8_t>(std::clamp(autoaim_target_id_, 0, 7));
    target.target_distance = 0.0F;
    target.vision_mode =
        static_cast<uint8_t>(std::clamp(autoaim_vision_mode_, 1, 4));
    target.shoot_command = gary_msgs::msg::AutoAIM::CEASE_FIRE;
    target_pub_->publish(target);
}

}  // namespace drone::guide_manager

RCLCPP_COMPONENTS_REGISTER_NODE(drone::guide_manager::GuideManagerNode)
