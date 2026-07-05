cmake_minimum_required(VERSION 3.23)

project(demucs_onnx_cli LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ONNX_RUNTIME_INCLUDE_DIR "")
set(ONNX_RUNTIME_LIB "")

set(DEMUCS_ONNX_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(NEURALPLAYER_ROOT "${DEMUCS_ONNX_ROOT}/..")

if(DEFINED DEMUCS_ONNXRUNTIME_ROOT AND EXISTS "${DEMUCS_ONNXRUNTIME_ROOT}")
    if(EXISTS "${DEMUCS_ONNXRUNTIME_ROOT}/include")
        set(ONNX_RUNTIME_INCLUDE_DIR "${DEMUCS_ONNXRUNTIME_ROOT}/include")
    endif()

    set(_onnxruntime_root_lib_candidates
        "${DEMUCS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib"
        "${DEMUCS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        "${DEMUCS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.a"
        "${DEMUCS_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    foreach(_candidate IN LISTS _onnxruntime_root_lib_candidates)
        if(EXISTS "${_candidate}")
            set(ONNX_RUNTIME_LIB "${_candidate}")
            break()
        endif()
    endforeach()
endif()

if(ONNX_RUNTIME_LIB STREQUAL "" AND DEFINED DEMUCS_ONNX_RUNTIME_LIB AND EXISTS "${DEMUCS_ONNX_RUNTIME_LIB}")
    set(ONNX_RUNTIME_LIB "${DEMUCS_ONNX_RUNTIME_LIB}")
endif()

if(ONNX_RUNTIME_LIB STREQUAL "")
    set(_onnxruntime_candidates
        "${DEMUCS_ONNX_ROOT}/third_party/onnxruntime-release/current/lib/libonnxruntime.dylib"
        "${DEMUCS_ONNX_ROOT}/third_party/onnxruntime-release/current/lib/libonnxruntime.so"
        "${DEMUCS_ONNX_ROOT}/third_party/onnxruntime-release/current/lib/libonnxruntime.a"
        "${NEURALPLAYER_ROOT}/NeuralNote/ThirdParty/onnxruntime/lib/libonnxruntime.a")
    foreach(_candidate IN LISTS _onnxruntime_candidates)
        if(EXISTS "${_candidate}")
            set(ONNX_RUNTIME_LIB "${_candidate}")
            break()
        endif()
    endforeach()
endif()

if(ONNX_RUNTIME_INCLUDE_DIR STREQUAL "")
    if(DEFINED DEMUCS_ONNXRUNTIME_ROOT AND EXISTS "${DEMUCS_ONNXRUNTIME_ROOT}/include")
        set(ONNX_RUNTIME_INCLUDE_DIR "${DEMUCS_ONNXRUNTIME_ROOT}/include")
    elseif(EXISTS "${DEMUCS_ONNX_ROOT}/third_party/onnxruntime-release/current/include")
        set(ONNX_RUNTIME_INCLUDE_DIR "${DEMUCS_ONNX_ROOT}/third_party/onnxruntime-release/current/include")
    elseif(EXISTS "${DEMUCS_ONNX_ROOT}/vendor/onnxruntime/include")
        set(ONNX_RUNTIME_INCLUDE_DIR "${DEMUCS_ONNX_ROOT}/vendor/onnxruntime/include")
    endif()
endif()

if(ONNX_RUNTIME_LIB STREQUAL "")
    message(FATAL_ERROR
        "Could not find ONNX Runtime library for Demucs CLI. "
        "Run scripts/fetch-onnxruntime-release.py or pass "
        "-DDEMUCS_ONNXRUNTIME_ROOT=/path/to/onnxruntime-package "
        "or -DDEMUCS_ONNX_RUNTIME_LIB=/absolute/path/to/libonnxruntime.")
endif()

if(ONNX_RUNTIME_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR
        "Could not find ONNX Runtime headers. "
        "Pass -DDEMUCS_ONNXRUNTIME_ROOT=/path/to/onnxruntime-package.")
endif()

set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example" FORCE)
add_subdirectory("${DEMUCS_ONNX_ROOT}/vendor/libnyquist" "${CMAKE_CURRENT_BINARY_DIR}/libnyquist_build")

add_executable(demucs
    "${DEMUCS_ONNX_ROOT}/src/dsp.cpp"
    "${DEMUCS_ONNX_ROOT}/src/model_apply.cpp"
    "${DEMUCS_ONNX_ROOT}/src/model_inference.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/demucs.cpp")

target_include_directories(demucs
    PRIVATE
        "${ONNX_RUNTIME_INCLUDE_DIR}"
        "${DEMUCS_ONNX_ROOT}/src"
        "${DEMUCS_ONNX_ROOT}/vendor/eigen"
        "${DEMUCS_ONNX_ROOT}/vendor/libnyquist/include")

target_link_libraries(demucs PRIVATE "${ONNX_RUNTIME_LIB}" libnyquist)

get_filename_component(ONNX_RUNTIME_LIB_DIR "${ONNX_RUNTIME_LIB}" DIRECTORY)
if(UNIX AND ONNX_RUNTIME_LIB MATCHES "\\.(so|dylib)$")
    set_target_properties(demucs PROPERTIES
        BUILD_RPATH "${ONNX_RUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNX_RUNTIME_LIB_DIR}")
endif()

message(STATUS "Using ONNX Runtime library: ${ONNX_RUNTIME_LIB}")
message(STATUS "Using ONNX Runtime include dir: ${ONNX_RUNTIME_INCLUDE_DIR}")
