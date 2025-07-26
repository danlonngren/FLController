cmake_minimum_required(VERSION 3.28.3)
project(FLControllerProject LANGUAGES CXX) 

option(BUILD_TESTS "Build GoogleTests tests" OFF)

add_library(FLController STATIC 
    FLController.cpp
    FLController.h
)

target_include_directories(FLController
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include  # for FLControllerInterface.h
        ${CMAKE_CURRENT_SOURCE_DIR}             # for FLController.h
)

target_compile_features(FLController PUBLIC cxx_std_20)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()