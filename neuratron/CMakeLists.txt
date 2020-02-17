cmake_minimum_required(VERSION 3.15)
project(neuratron)
set(CMAKE_CXX_STANDARD 14)
include_directories(${neuratron_SOURCE_DIR}/include)

add_library(neuratron SHARED src/neuratron.cpp include/neuratron.hpp)

install(TARGETS neuratron DESTINATION usr/lib)
