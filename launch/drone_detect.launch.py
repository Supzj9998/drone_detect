from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # 模型路径与 engine 路径分开配置：engine 不存在时 model_detecter 节点会尝试由 ONNX 自动构建。
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="model/ONNX/model.onnx",
        description="ONNX model path used to build a TensorRT engine when needed.",
    )

    engine_path_arg = DeclareLaunchArgument(
        "engine_path",
        default_value="model/TensorRT/model.engine",
        description="TensorRT engine path for model_detecter node.",
    )

    trt_workspace_size_mb_arg = DeclareLaunchArgument(
        "trt_workspace_size_mb",
        default_value="1024",
        description="TensorRT builder workspace size in MB.",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="image_raw",
        description="Input image topic for model_detecter node.",
    )

    enable_model_detecter_arg = DeclareLaunchArgument(
        "enable_model_detecter",
        default_value="true",
        description="Enable model_detecter_node.",
    )

    enable_guide_manager_arg = DeclareLaunchArgument(
        "enable_guide_manager",
        default_value="false",
        description="Enable wide-camera guide manager publishing /autoaim/target.",
    )

    enable_drone_detecter_arg = DeclareLaunchArgument(
        "enable_drone_detecter",
        default_value="false",
        description="Enable drone_detecter_node publishing drone_detecter/guide_polar.",
    )

    enable_pnp_arg = DeclareLaunchArgument(
        "enable_pnp",
        default_value="true",
        description="Enable pnp_node publishing pnp result and /autoaim/target.",
    )

    guide_yaw_offset_rad_arg = DeclareLaunchArgument(
        "guide_yaw_offset_rad",
        default_value="0.0",
        description="Yaw offset added by guide_manager_node, in radians.",
    )

    guide_pitch_offset_rad_arg = DeclareLaunchArgument(
        "guide_pitch_offset_rad",
        default_value="0.0",
        description="Pitch offset added by guide_manager_node, in radians.",
    )

    # YOLO/TensorRT 检测节点：输入图像，输出带框图像和 Float32MultiArray 检测框。
    model_detecter_node = Node(
        package="drone_detect",
        executable="model_detecter_node",
        name="model_detecter_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_model_detecter")),
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "engine_path": LaunchConfiguration("engine_path"),
                "trt_workspace_size_mb": LaunchConfiguration("trt_workspace_size_mb"),
                "image_topic": LaunchConfiguration("image_topic"),
            }
        ],
    )

    # PnP 节点：把检测框转换为目标三维方向/距离，并可发布 AutoAIM 消息。
    pnp_node = Node(
        package="drone_detect",
        executable="pnp_node",
        name="pnp_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_pnp")),
    )

    # 固定广角相机无人机检测节点：默认关闭，启用后发布 drone_detecter/guide_polar。
    drone_detecter_node = Node(
        package="drone_detect",
        executable="drone_detecter_node",
        name="drone_detecter_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_drone_detecter")),
    )

    # 广角引导节点：默认关闭，避免和 pnp_node 同时发布 /autoaim/target。
    guide_manager_node = Node(
        package="drone_detect",
        executable="guide_manager_node",
        name="guide_manager_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_guide_manager")),
        parameters=[
            {
                "yaw_offset_rad": LaunchConfiguration("guide_yaw_offset_rad"),
                "pitch_offset_rad": LaunchConfiguration("guide_pitch_offset_rad"),
            }
        ],
    )

    return LaunchDescription(
        [
            model_path_arg,
            engine_path_arg,
            trt_workspace_size_mb_arg,
            image_topic_arg,
            enable_model_detecter_arg,
            enable_guide_manager_arg,
            enable_drone_detecter_arg,
            enable_pnp_arg,
            guide_yaw_offset_rad_arg,
            guide_pitch_offset_rad_arg,
            model_detecter_node,
            pnp_node,
            drone_detecter_node,
            guide_manager_node,
        ]
    )
