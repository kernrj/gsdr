cmake_minimum_required(VERSION 3.17)

set(TEST_PROJECT_NAME ${LIB_PROJECT_NAME}_tests)
set(TEST_TARGET_NAME ${TEST_PROJECT_NAME})

project(${TEST_PROJECT_NAME})

find_package(GTest REQUIRED)

# Enable testing
enable_testing()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/../include)
include_directories(${PROJECT_BINARY_DIR}/../include)

# Find all test source files
file(GLOB TEST_SOURCES "*.cpp")

# Create test executable
add_executable(${TEST_TARGET_NAME} ${TEST_SOURCES})

# Link with the main library and GTest
target_link_libraries(${TEST_TARGET_NAME}
    ${LIB_TARGET_NAME}
    GTest::gtest_main
    GTest::gtest
)

# Set CUDA architecture if needed
if (CMAKE_CUDA_COMPILER)
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        CUDA_ARCHITECTURES "75"
    )
endif()

# Add tests
add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})

# Test coverage (if enabled)
if (ENABLE_COVERAGE)
    target_compile_options(${TEST_TARGET_NAME} PRIVATE --coverage)
    target_link_options(${TEST_TARGET_NAME} PRIVATE --coverage)
endif()

# Output directory
set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)