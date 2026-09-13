cmake_minimum_required(VERSION 3.10)



add_library( dllayers
    src/ActivationLayer.cpp
    src/FullyConnected.cpp
    src/ConvLayer.cpp
)

target_include_directories( dllayers
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries( dllayers
    PUBLIC
    MyTensors::MyTensors
)


add_library(
    DLLib::DLLib ALIAS dllayers
)
