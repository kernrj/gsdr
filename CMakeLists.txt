cmake_minimum_required(VERSION 3.17)

set(LIB_PROJECT_NAME gsdr)
set(LIB_TARGET_NAME ${LIB_PROJECT_NAME})
set(PROJECT_MACRO_PREFIX GSDR)

set(
    LIB_SOURCES
    src/add_const.cu
    src/adjustFrequency.cu
    src/adjustFrequency.cuh
    src/am.cu
    src/conversion.cu
    src/fir.cu
    src/fm.cu
    src/magnitude.cu
    src/multiply.cu
    src/qpsk.cu
    src/qpsk256.cu
    src/quad_demod.cu
    src/trig.cu)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

# Common

project(${LIB_PROJECT_NAME})

find_package(CUDAToolkit REQUIRED)

set(LIB_DEV_PUBLIC_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")
file(GLOB_RECURSE PUBLIC_HEADERS "${LIB_DEV_PUBLIC_HEADER_DIR}" *.h)

add_library(
    ${LIB_TARGET_NAME}
    ${PUBLIC_HEADERS}
    ${LIB_SOURCES}
)

set(GENERATED_HEADER_DIR "${PROJECT_BINARY_DIR}/include")
set(EXPORT_FILE "${GENERATED_HEADER_DIR}/${LIB_PROJECT_NAME}/${LIB_PROJECT_NAME}_export.h")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

include(GenerateExportHeader)
generate_export_header(
    ${LIB_TARGET_NAME}
    BASE_NAME ${PROJECT_MACRO_PREFIX}
    EXPORT_MACRO_NAME ${PROJECT_MACRO_PREFIX}_PUBLIC
    NO_EXPORT_MACRO_NAME ${PROJECT_MACRO_PREFIX}_PRIVATE
    STATIC_DEFINE ${PROJECT_MACRO_PREFIX}_STATIC_BUILD
    EXPORT_FILE_NAME ${EXPORT_FILE}
)

target_include_directories(${LIB_TARGET_NAME} INTERFACE "$<INSTALL_INTERFACE:include>")
target_include_directories(${LIB_TARGET_NAME} BEFORE PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(${LIB_TARGET_NAME} SYSTEM PRIVATE "${GENERATED_HEADER_DIR}")

target_compile_definitions(${LIB_TARGET_NAME} PRIVATE $<$<CONFIG:Debug>:DEBUG>)

if (WIN32)
    install(
        TARGETS ${LIB_TARGET_NAME}
        EXPORT ${LIB_TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
    )

    install(FILES $<TARGET_PDB_FILE:${LIB_TARGET_NAME}> DESTINATION bin OPTIONAL)
else ()
    install(
        TARGETS ${LIB_TARGET_NAME}
        EXPORT ${LIB_TARGET_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif ()

install(DIRECTORY "${LIB_DEV_PUBLIC_HEADER_DIR}/${LIB_PROJECT_NAME}" DESTINATION include/)
install(DIRECTORY "${GENERATED_HEADER_DIR}/${LIB_PROJECT_NAME}" DESTINATION include/)
install(EXPORT "${LIB_TARGET_NAME}" DESTINATION "lib/cmake/${LIB_PROJECT_NAME}" FILE "${LIB_TARGET_NAME}-config.cmake")

option(USE_TESTS "Enable unit testing" ON)
if (USE_TESTS AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/tests")
    message("Tests enabled.")
    enable_testing()
    add_subdirectory(tests)
endif ()

# End of common section

target_include_directories(
    ${LIB_TARGET_NAME}
    SYSTEM PRIVATE
    /usr/include
    /usr/local/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(
    ${LIB_TARGET_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

target_link_directories(
    ${LIB_TARGET_NAME}
    SYSTEM PRIVATE
    /usr/local/lib/x86_64-linux-gnu
    /lib/x86_64-linux-gnu/
)

if (MSVC)
    target_compile_options(${LIB_TARGET_NAME} PRIVATE /W4)
else ()
    set(
        WARNING_FLAGS
        -Wall
        -Wextra
        -Werror=unused-result
        -Werror=return-type
    )

    list(JOIN WARNING_FLAGS "," CUDA_WARNING_FLAGS)

    target_compile_options(
        ${LIB_TARGET_NAME}
        PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options ${CUDA_WARNING_FLAGS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${WARNING_FLAGS}>"
        "$<$<COMPILE_LANGUAGE:C>:${WARNING_FLAGS}>"
    )
endif ()

#target_compile_options(${LIB_TARGET_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

get_target_property(LIB_TARGET_TYPE ${LIB_TARGET_NAME} TYPE)
if (LIB_TARGET_TYPE STREQUAL STATIC_LIBRARY)
    set(DEPENDENCY_VISIBILITY PUBLIC)
    set_property(TARGET ${LIB_TARGET_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
    set(DEPENDENCY_VISIBILITY PRIVATE)
endif()

target_link_libraries(${LIB_TARGET_NAME}
    ${DEPENDENCY_VISIBILITY}
    CUDA::cudart
)
