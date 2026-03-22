cmake_minimum_required(VERSION 3.15)
project(Backend)

# Generate compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV and Pybind11 globally
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# Build C++ code into a Python module
pybind11_add_module(backend main_binding.cpp)
pybind11_add_module(task2_backend task2_shapes.cpp)

# Link the required libraries
target_include_directories(backend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(backend PRIVATE ${OpenCV_LIBS})

target_include_directories(task2_backend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task2_backend PRIVATE ${OpenCV_LIBS})