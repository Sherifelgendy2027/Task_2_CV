# cmake_minimum_required(VERSION 3.15)
# project(MyBackend)

# # 1. Automatically download pybind11 directly from GitHub
# include(FetchContent)
# FetchContent_Declare(
#   pybind11
#   GIT_REPOSITORY https://github.com/pybind/pybind11.git
#   GIT_TAG        v2.13.6 # Latest stable version
# )
# FetchContent_MakeAvailable(pybind11)

# # 2. Build your C++ code into a Python module
# pybind11_add_module(my_backend backend.cpp)
cmake_minimum_required(VERSION 3.15)
project(Backend)

# Generate compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV and Pybind11 globally
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# Build C++ code into a Python module
pybind11_add_module(backend main_binding.cpp)

# Link the required libraries
target_include_directories(backend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(backend PRIVATE ${OpenCV_LIBS})
