cmake_minimum_required(VERSION 3.15)
project(neuratron)
set(CMAKE_CXX_STANDARD 17)
include_directories(${neuratron_SOURCE_DIR}/include)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(neuratron SHARED src/neuratron.cpp src/deep_model.cpp)
target_link_libraries (neuratron Eigen3::Eigen)

install(TARGETS neuratron DESTINATION lib)
