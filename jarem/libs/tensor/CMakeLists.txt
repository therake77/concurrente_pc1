cmake_minimum_required(VERSION 3.10)


add_library( tensor
    src/core/Shape.cpp
    src/core/Tensor.cpp
    src/hardware/CpuDevice.cpp
    src/hardware/DeviceManager.cpp
    src/math/CpuMath.cpp
    src/math/TensorMath.cpp
)

target_include_directories( tensor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CUDA_ENABLED)
    set(CUDA_ARCHITECTURES native)
    message("CUDA Toolkit found. Building CUDA-dependent targets")
    target_compile_definitions(
        tensor
        PUBLIC 
        __USE_CUDA__
    )
    target_sources(tensor
        PRIVATE
        src/hardware/GpuDevice.cu
        src/math/GpuMath.cu
    )
    target_link_libraries(
        tensor
        PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::curand
    )
else()
    message("CUDA Toolkit not found. Not building CUDA-dependent targets")
endif()



add_library(
    MyTensors::MyTensors ALIAS tensor
)