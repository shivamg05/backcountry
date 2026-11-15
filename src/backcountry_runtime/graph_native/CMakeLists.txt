cmake_minimum_required(VERSION 3.27)
# MLX's oldest macOS, so the binary never needs a newer one than its MLX library
set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0 CACHE STRING "")
project(_graph_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(MLX CONFIG REQUIRED)
nanobind_add_module(_graph_native NB_STATIC NB_DOMAIN mlx graph_native.cpp)
target_link_libraries(_graph_native PRIVATE mlx)
# libmlx is found already loaded (mlx.core is imported first), not by a path on the building Mac
set_target_properties(_graph_native PROPERTIES SKIP_BUILD_RPATH ON)
