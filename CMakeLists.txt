cmake_minimum_required(VERSION 3.4.1)

project("MA")

set(CMAKE_CXX_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER nvcc)

add_compile_options(-g -Wall -ansi -Wformat -std=c++17) 

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

# COMPILE CU FILES
file(GLOB CUDA_FILES "./" CUDAFLERP.cu)
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86; -std=c++17")
CUDA_COMPILE(CU_O ${CUDA_FILES})

cuda_add_executable(MA main.cpp ${CU_O})

target_link_libraries(  # Specifies the target library.
                        MA
                        ${OpenCV_LIBS}
                        -lpthread
                        -lboost_serialization
                        )
