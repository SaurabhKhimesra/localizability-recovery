cmake_minimum_required(VERSION 3.18)
project(locrec_estimator)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # -Wno-array-bounds: GCC's vectoriser reports false positives inside Eigen's fixed-size blocks
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-array-bounds)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(locrec_msgs REQUIRED)
find_package(locrec_core REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}_lib SHARED
  src/calibration.cpp
  src/conversions.cpp
  src/detector.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC locrec_core::locrec_core Eigen3::Eigen)
ament_target_dependencies(${PROJECT_NAME}_lib PUBLIC
  nav_msgs sensor_msgs std_msgs builtin_interfaces locrec_msgs)

add_executable(localizability_node src/localizability_node.cpp)
target_link_libraries(localizability_node ${PROJECT_NAME}_lib)
ament_target_dependencies(localizability_node rclcpp locrec_msgs nav_msgs sensor_msgs std_msgs)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS localizability_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  foreach(test_name test_calibration test_prior_pairer test_conversions test_detector)
    ament_add_gtest(${test_name} test/${test_name}.cpp)
    target_link_libraries(${test_name} ${PROJECT_NAME}_lib)
  endforeach()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(locrec_core locrec_msgs nav_msgs sensor_msgs std_msgs Eigen3)
ament_package()
