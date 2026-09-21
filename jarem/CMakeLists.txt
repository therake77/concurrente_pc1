cmake_minimum_required(VERSION 3.10)


project(DL_Algorithms
    LANGUAGES
    CXX
)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()

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
    main_mlp
    main.cpp
    src/SimpleNetwork.cpp
)

add_executable(
    main_cnn
    main2.cpp
    src/SimpleNetwork.cpp
)

add_executable(
    test
    tests.cpp
)


target_include_directories(
    main_mlp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    main_mlp
    PRIVATE
    MyTensors::MyTensors   
)

target_link_libraries(
    main_mlp
    PRIVATE
    DLLib::DLLib
)

target_include_directories(
    main_cnn
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    main_cnn
    PRIVATE
    MyTensors::MyTensors   
)

target_link_libraries(
    main_cnn
    PRIVATE
    DLLib::DLLib
)



target_link_libraries(
    test
    PRIVATE
    MyTensors::MyTensors
)