cmake_minimum_required(VERSION 3.10)


project(DL_Algorithms
    LANGUAGES
    CXX
)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_ENABLED ON)
    set(CUDA_ARCHITECTURES native)
else()
    set(CUDA_ENABLED OFF)
endif()


add_subdirectory(libs/tensor)
add_subdirectory(libs/dllayers)

target_include_directories(
    tensor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(
    dllayers
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(
    main
    main.cpp
    src/SimpleNetwork.cpp
)

target_include_directories(
    main
    PRIVATE
    $(CMAKE_CURRENT_SOURCE_DIR)/include
)

target_link_libraries(
    main
    PRIVATE
    MyTensors::MyTensors   
)

target_link_libraries(
    main
    PRIVATE
    DLLib::DLLib
)
