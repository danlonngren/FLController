add_executable(FLControllerTests 
    FLControllerTests.cpp 
)

target_link_libraries(FLControllerTests 
    PRIVATE 
        GTest::gtest_main
        FLController
)

target_include_directories(FLControllerTests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/source
)

include (GoogleTest)
gtest_discover_tests(FLControllerTests)