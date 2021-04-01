cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

project(cldl)

add_library(cldl
        lib/Layer.cpp
        lib/Net.cpp
        lib/Neuron.cpp)

target_include_directories(cldl PUBLIC include)

add_subdirectory(tests)