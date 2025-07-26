include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0  # Use any tag, commit hash, or branch
)

# Prevent GoogleTest from overriding the parent project's compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

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

include(GoogleTest)
gtest_discover_tests(FLControllerTests)