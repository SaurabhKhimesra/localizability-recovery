cmake_minimum_required(VERSION 3.10)
project(locrec_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Localizability.msg"
  "msg/MarkerObservation.msg"
  "msg/MarkerDrop.msg"
  "msg/ScanReport.msg"
  "msg/SharedLandmark.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_package()
