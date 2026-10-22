cmake_minimum_required(VERSION 3.16)

project(demucs.onnx)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    # The upstream release flags are a GCC fast-math recipe; /fp:fast is the
    # closest MSVC equivalent. /bigobj is needed because Eigen instantiates a
    # lot of templates in single translation units.
    add_compile_options(/W3 /O2 /fp:fast /EHsc /bigobj /Zc:__cplusplus)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -DEIGEN_INTERNAL_DEBUGGING")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")
endif()

# Path to the ONNX Runtime library:
#  - Linux: minimal library built by ./scripts/build-ort-linux.sh
#  - Windows: prebuilt NuGet payload staged by build-win.ps1:
#      deps/ort      CPU only
#      deps/ort-dml  CPU + DirectML  (DEMUCS_EP=dml)
#      deps/ort-cuda CPU + CUDA      (DEMUCS_EP=cuda)
#    Only one can be linked (they ship different onnxruntime.dll builds), so
#    DEMUCS_ORT_VARIANT picks it. "auto" prefers cuda, then dml, then cpu.
if(WIN32)
    set(DEMUCS_ORT_VARIANT "auto" CACHE STRING "ONNX Runtime build: auto|cpu|dml|cuda")
    set_property(CACHE DEMUCS_ORT_VARIANT PROPERTY STRINGS auto cpu dml cuda)

    set(DEMUCS_ORT_DIR "")
    if(DEMUCS_ORT_VARIANT STREQUAL "cuda")
        set(DEMUCS_ORT_DIR ${CMAKE_SOURCE_DIR}/../deps/ort-cuda)
    elseif(DEMUCS_ORT_VARIANT STREQUAL "dml")
        set(DEMUCS_ORT_DIR ${CMAKE_SOURCE_DIR}/../deps/ort-dml)
    elseif(DEMUCS_ORT_VARIANT STREQUAL "cpu")
        set(DEMUCS_ORT_DIR ${CMAKE_SOURCE_DIR}/../deps/ort)
    else()
        foreach(candidate ort-cuda ort-dml ort)
            if(DEMUCS_ORT_DIR STREQUAL "" AND EXISTS "${CMAKE_SOURCE_DIR}/../deps/${candidate}/lib/onnxruntime.lib")
                set(DEMUCS_ORT_DIR ${CMAKE_SOURCE_DIR}/../deps/${candidate})
            endif()
        endforeach()
    endif()

    if(DEMUCS_ORT_DIR MATCHES "ort-cuda")
        add_compile_definitions(DEMUCS_HAS_CUDA)
        message(STATUS "ONNX Runtime: CUDA (GPU) build from deps/ort-cuda")
    elseif(DEMUCS_ORT_DIR MATCHES "ort-dml")
        add_compile_definitions(DEMUCS_HAS_DML)
        message(STATUS "ONNX Runtime: DirectML (GPU) build from deps/ort-dml")
    else()
        message(STATUS "ONNX Runtime: CPU-only build from deps/ort")
    endif()

    set(ONNX_RUNTIME_INCLUDE ${DEMUCS_ORT_DIR}/include)
    set(ONNX_RUNTIME_LIB ${DEMUCS_ORT_DIR}/lib/onnxruntime.lib)
    if(NOT EXISTS "${ONNX_RUNTIME_LIB}")
        message(FATAL_ERROR "ONNX Runtime is not staged at ${ONNX_RUNTIME_LIB}. Run ./scripts/build-win.ps1 -Variant <cpu|dml|cuda>.")
    endif()
else()
    set(ONNX_RUNTIME_INCLUDE ${CMAKE_SOURCE_DIR}/../vendor/onnxruntime/include)
    set(ONNX_RUNTIME_LIB ${CMAKE_SOURCE_DIR}/../build/build-ort-linux/MinSizeRel/libonnxruntime.so)
endif()

set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/libnyquist ${CMAKE_CURRENT_BINARY_DIR}/libnyquist_build)

include_directories(${ONNX_RUNTIME_INCLUDE})
# provider headers (dml_provider_factory.h) include onnxruntime_c_api.h as a
# sibling, but they live in a different subdirectory of the SDK
include_directories(${ONNX_RUNTIME_INCLUDE}/onnxruntime/core/session)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../onnx-models/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/libnyquist/include)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_cli/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../onnx-models/model/*.ort.c")
add_executable(demucs ${SOURCES})

target_link_libraries(demucs ${ONNX_RUNTIME_LIB} libnyquist)

if(WIN32)
    # onnxruntime.lib is only an import library: the DLL(s) must sit next to the
    # exe. EP providers are loaded lazily from onnxruntime_providers_*.dll.
    add_custom_command(TARGET demucs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DEMUCS_ORT_DIR}/bin/onnxruntime.dll"
                "$<TARGET_FILE_DIR:demucs>"
        COMMENT "Copying onnxruntime.dll next to demucs.exe")
    if(EXISTS "${DEMUCS_ORT_DIR}/bin/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET demucs POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DEMUCS_ORT_DIR}/bin/onnxruntime_providers_shared.dll"
                    "$<TARGET_FILE_DIR:demucs>")
    endif()
    if(EXISTS "${DEMUCS_ORT_DIR}/bin/onnxruntime_providers_cuda.dll")
        add_custom_command(TARGET demucs POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DEMUCS_ORT_DIR}/bin/onnxruntime_providers_cuda.dll"
                    "$<TARGET_FILE_DIR:demucs>")
    endif()
endif()
