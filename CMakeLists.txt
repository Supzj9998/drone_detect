cmake_minimum_required(VERSION 3.8)
project(drone_detect)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(common_infer REQUIRED)

include_directories(
  "detect/include"
  "pnp/include"
)

# detect_node 通过 common_infer 复用 TensorRT 推理封装。
set(DRONE_DETECT "drone_detect")
file(GLOB_RECURSE DRONE_DETECT_SRCS "detect/src/*.cpp")

ament_auto_add_library(${DRONE_DETECT} SHARED
  ${DRONE_DETECT_SRCS}
)

ament_target_dependencies(${DRONE_DETECT}
  common_infer
)

rclcpp_components_register_node(${DRONE_DETECT}
  PLUGIN "drone::detect::Detect"
  EXECUTABLE detect_node
)

# pnp_node 只依赖 ROS/OpenCV 依赖，用于几何解算和消息转换。
set(DRONE_PNP "drone_pnp")
file(GLOB_RECURSE DRONE_PNP_SRCS "pnp/src/*.cpp")

ament_auto_add_library(${DRONE_PNP} SHARED
  ${DRONE_PNP_SRCS}
)

rclcpp_components_register_node(${DRONE_PNP}
  PLUGIN "drone::pnp::PnpNode"
  EXECUTABLE pnp_node
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  launch
)
