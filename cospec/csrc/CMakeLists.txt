# libsmctrl — SM (TPC) masking library for fine-grained GPU partitioning.
# Adopted from BulletServe (https://github.com/NUS-HPC-AI-Lab/BulletServe).
# Original library by Joshua Bakita (2022-2025).

cmake_minimum_required(VERSION 3.17)
project(SMCtrl LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES not defined")
endif()

add_library(smctrl SHARED src/libsmctrl_core.c src/libsmctrl_validator.cu)
target_include_directories(smctrl PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(smctrl PRIVATE cuda)
target_link_options(smctrl PRIVATE LINKER:-rpath,${CUDAToolkit_LIBRARY_DIR})
set_target_properties(smctrl PROPERTIES
    PUBLIC_HEADER src/libsmctrl.h
    OUTPUT_NAME smctrl)
