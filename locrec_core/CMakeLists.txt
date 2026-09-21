cmake_minimum_required(VERSION 3.18)
project(locrec_core)

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
find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)

# small_gicp is the registration backend. It has no rosdep key, so either point the build at a
# checkout with -DSMALL_GICP_SOURCE_DIR=<path> (offline builds, packaging) or let CMake fetch the
# release this package was developed against. Only two of its translation units are needed, and
# they are compiled straight into this library, so nothing extra has to be installed at runtime.
set(SMALL_GICP_SOURCE_DIR "" CACHE PATH "Existing small_gicp checkout; empty fetches v1.0.1")
if(SMALL_GICP_SOURCE_DIR STREQUAL "")
  include(FetchContent)
  FetchContent_Declare(
    small_gicp
    GIT_REPOSITORY https://github.com/koide3/small_gicp.git
    GIT_TAG v1.0.1
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR include  # no CMakeLists there, so this populates without configuring it
  )
  FetchContent_MakeAvailable(small_gicp)
  set(SMALL_GICP_SOURCE_DIR ${small_gicp_SOURCE_DIR})
endif()
if(NOT EXISTS ${SMALL_GICP_SOURCE_DIR}/include/small_gicp/registration/registration_helper.hpp)
  message(FATAL_ERROR "SMALL_GICP_SOURCE_DIR=${SMALL_GICP_SOURCE_DIR} is not a small_gicp checkout")
endif()

add_library(${PROJECT_NAME} SHARED
  src/gaze.cpp
  src/localizability.cpp
  src/landmarks.cpp
  src/odometry.cpp
  src/registration.cpp
  ${SMALL_GICP_SOURCE_DIR}/src/small_gicp/registration/registration_helper.cpp
  ${SMALL_GICP_SOURCE_DIR}/src/small_gicp/registration/registration.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# small_gicp is an implementation detail: no public header of this package includes it. SYSTEM so
# this package's warning level does not apply to someone else's headers.
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${SMALL_GICP_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
set_source_files_properties(
  ${SMALL_GICP_SOURCE_DIR}/src/small_gicp/registration/registration_helper.cpp
  ${SMALL_GICP_SOURCE_DIR}/src/small_gicp/registration/registration.cpp
  PROPERTIES COMPILE_OPTIONS "-w"  # third-party source, not ours to warn about
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  foreach(test_name test_se3 test_localizability test_landmarks test_odometry test_policies
      test_gaze)
    ament_add_gtest(${test_name} test/${test_name}.cpp)
    target_link_libraries(${test_name} ${PROJECT_NAME} Eigen3::Eigen)
  endforeach()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_package()
