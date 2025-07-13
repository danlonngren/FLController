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
