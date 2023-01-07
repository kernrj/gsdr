cmake_minimum_required(VERSION 3.14)

project(gsdr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

include(FetchContent)
FetchContent_Declare(
        remez
        GIT_REPOSITORY https://github.com/kernrj/remez-exchange.git
        GIT_TAG master
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(remez)

set(GSDR_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")

FILE(GLOB headers "${GSDR_HEADER_DIR}/gsdr/*")

add_library(gsdr
        ${headers}
        src/add_const.cu
        src/conversion.cu
        src/fir.cu
        src/magnitude.cu
        src/multiply.cu
        src/quad_demod.cu
        src/trig.cu)

target_include_directories(gsdr
        SYSTEM PRIVATE
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
)

set_target_properties(gsdr PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_directories(
        gsdr SYSTEM PRIVATE
        /usr/local/lib/x86_64-linux-gnu
        /lib/x86_64-linux-gnu/
)

target_link_libraries(gsdr remez)

target_include_directories( gsdr PRIVATE "${GSDR_HEADER_DIR}" )
target_include_directories( gsdr INTERFACE "$<INSTALL_INTERFACE:include>" )

set_target_properties(gsdr PROPERTIES CUDA_ARCHITECTURES "75")

include( GenerateExportHeader )

set( EXPORT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/gsdr/gsdr_export.h" )

generate_export_header(
        gsdr
        BASE_NAME GSDR
        EXPORT_MACRO_NAME GSDR_PUBLIC
        NO_EXPORT_MACRO_NAME GSDR_PRIVATE
        STATIC_DEFINE GSDR_STATIC_BUILD
        EXPORT_FILE_NAME ${EXPORT_FILE}
)

if( WIN32 )
    install(
            TARGETS gsdr
            EXPORT gsdr
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION bin
            ARCHIVE DESTINATION lib
    )

    install( FILES $<TARGET_PDB_FILE:gsdr> DESTINATION bin OPTIONAL )
else()
    install(
            TARGETS gsdr
            EXPORT gsdr
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
    )
endif()

FILE(GLOB PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/gsdr/*")

install( FILES ${PUBLIC_HEADERS} DESTINATION include/gsdr )
install( EXPORT gsdr DESTINATION lib/cmake/gsdr FILE gsdr-config.cmake )
