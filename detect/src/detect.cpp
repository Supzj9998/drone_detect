#include "detect.h"

#include <functional>
#include <string>

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace drone::detect {

Detect::Detect(const rclcpp::NodeOptions& options)
    : Node("detect_node", options)
{
    // 读取参数和模型路径
    const auto engine_path = declare_parameter<std::string>(
        "engine_path", "model/TensorRT/best.engine");
    const auto model_path = declare_parameter<std::string>(
        "model_path", "model/ONNX/best.onnx");
    // 读取TensorRT构建engine使用的workspace大小
    const auto workspace_size_mb =
        declare_parameter<int>("trt_workspace_size_mb", 1024);

    // 指定yolo类型
    const auto yolo_type = ::yolo::Type::V11;
    RCLCPP_INFO(get_logger(), "Using YOLO decoder: %s",
                ::yolo::type_name(yolo_type));
    // 加载YOLO/TensorRT推理引擎
    yolo = ::yolo::load_or_build(
        engine_path, model_path, yolo_type,
        static_cast<float>(
            declare_parameter<double>("conf_threshold", 0.25)),
        static_cast<float>(
            declare_parameter<double>("nms_threshold", 0.45)),
        static_cast<size_t>(workspace_size_mb) * 1024ULL * 1024ULL);
    if (!yolo) {
        throw std::runtime_error("Failed to load yolo engine");
    }

    // yolo检测后的图像发布者
    image_pub = create_publisher<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("output_image_topic",
                                       "detect/image_with_boxes"),
        10);
    // yolo检测框发布者
    boxes_pub = create_publisher<std_msgs::msg::Float32MultiArray>(
        declare_parameter<std::string>("output_boxes_topic",
                                       "detect/boxes"),
        10);
    // 原始图像订阅者
    image_sub = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("image_topic", "image_raw"),
        rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));

    if (show_debug_image) {
        cv::namedWindow("yolo_debug", cv::WINDOW_NORMAL);
    }
}

Detect::~Detect()
{
    if (show_debug_image) {
        cv::destroyWindow("yolo_debug");
    }
}

void Detect::callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    // 把ros图像消息转换成Opencv图像
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                              "cv_bridge conversion failed: %s", e.what());
        return;
    }

    // 构造推理输入
    tdt_radar::Image image(cv_ptr->image.data, cv_ptr->image.cols,
                           cv_ptr->image.rows);
    // yolo推理
    const auto detections = yolo->forward(image);
    // 发布检测框
    publishDetections(detections);
    // 在图像上画框
    drawDetections(cv_ptr->image, detections);
    // OpenCV调试显示：每帧只刷新一次，避免多目标时重复阻塞。
    if (show_debug_image) {
        cv::imshow("yolo_debug", cv_ptr->image);
        cv::waitKey(1);
    }
    // 发布画框后的图像
    image_pub->publish(
        *cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8,
                            cv_ptr->image)
             .toImageMsg());
}

void Detect::publishDetections(const yolo::BoxArray& detections) const
{
    // 创建ros2消息
    std_msgs::msg::Float32MultiArray msg;
    // 提前预留容量
    msg.data.reserve(detections.size() * 10U);
    // yolo框转换ros2消息
    for (const auto& box : detections) {
        msg.data.insert(msg.data.end(),
                        {static_cast<float>(box.class_label),
                         box.confidence, box.left, box.top, box.right,
                         box.top, box.right, box.bottom, box.left,
                         box.bottom});
    }

    // 发布消息
    boxes_pub->publish(msg);
}

void Detect::drawDetections(cv::Mat&              image,
                            const yolo::BoxArray& detections) const
{
    // 遍历每一个检测框
    for (const auto& box : detections) {
        // 将浮点坐标转换成画图所需的整数坐标
        const cv::Point left_top(static_cast<int>(box.left),
                                 static_cast<int>(box.top));
        const cv::Point right_bottom(static_cast<int>(box.right),
                                     static_cast<int>(box.bottom));
        // 画绿色矩形
        cv::rectangle(image, left_top, right_bottom, cv::Scalar(0, 255, 0),
                      2);
        const std::string label = cv::format("%.2f", box.confidence);
        cv::putText(image, label, left_top, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
    }
}

}  // namespace drone::detect

RCLCPP_COMPONENTS_REGISTER_NODE(drone::detect::Detect)
