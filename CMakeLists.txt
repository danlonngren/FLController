cmake_minimum_required(VERSION 3.28.3)
project(sPIDControllerLib LANGUAGES CXX) 

option(BUILD_TESTS "Build sPIDController tests" OFF)

add_subdirectory(source)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(external/googletest)
endif()
