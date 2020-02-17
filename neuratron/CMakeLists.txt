cmake_minimum_required(VERSION 3.15)
project(neuratron)
set(CMAKE_C_STANDARD 99)
include_directories(${neuratron_SOURCE_DIR}/include)

add_library(neuratron SHARED src/neuratron.c)

install(TARGETS neuratron DESTINATION lib)
