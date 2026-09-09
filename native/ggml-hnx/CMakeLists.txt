# ggml-hnx — HyperNix sub-1-bit types for ggml / llama.cpp.
#
# Builds two ways:
#
#   standalone, to test the decoder on its own:
#       cmake -S . -B build && cmake --build build && ctest --test-dir build
#
#   as part of llama.cpp, which is what build.sh arranges:
#       add_subdirectory(ggml-hnx) and link ggml-hnx into ggml
#
# The standalone build is the one that matters day to day. The decoder
# has no ggml dependency precisely so it can be built and checked
# without cloning 200 MB of upstream first -- and so that the bit order
# is verified against the Python encoder rather than assumed.
cmake_minimum_required(VERSION 3.16)
project(ggml_hnx C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CUDA is opt-in and off by default: the CPU decoder is the thing that
# must build anywhere, and requiring a CUDA toolkit to compile a
# 300-line C file would be a bad trade. ON adds the kernels; the CPU
# path stays as the fallback either way, because a machine can have the
# toolkit and no device.
option(GGML_HNX_CUDA "Build the CUDA kernels for the sub-bit types" OFF)

add_library(ggml-hnx STATIC ggml-hnx.c)
target_include_directories(ggml-hnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# -fPIC because ggml is often built shared, and a static library linked
# into a shared one needs it.
set_property(TARGET ggml-hnx PROPERTY POSITION_INDEPENDENT_CODE ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ggml-hnx PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GGML_HNX_CUDA)
    enable_language(CUDA)
    target_sources(ggml-hnx PRIVATE ggml-hnx-cuda.cu)
    set_target_properties(ggml-hnx PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # 61 first, deliberately: Pascal is the floor this project supports
    # (see hypernix.system.pascal), and a GTX 1080 is exactly the card a
    # sub-bit model exists for. 75/86/89 cover Turing through Ada.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_property(TARGET ggml-hnx PROPERTY CUDA_ARCHITECTURES 61 75 86 89)
    endif()
    target_compile_definitions(ggml-hnx PUBLIC GGML_HNX_CUDA)
endif()

# Only when built on its own. Inside llama.cpp the parent project owns
# testing, and adding a second test binary to its ctest run is noise.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()

    add_executable(hnx_selftest hnx_selftest.c)
    target_link_libraries(hnx_selftest PRIVATE ggml-hnx m)

    add_test(NAME hnx_selftest COMMAND hnx_selftest)

    # The cross-check against hypernix.quant.subbit. Skipped rather than
    # failed when Python cannot import hypernix: the self-contained
    # checks above are still worth running on a machine that has only a
    # C compiler.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(
            NAME hnx_vectors
            COMMAND ${CMAKE_COMMAND}
                -DPYTHON=${Python3_EXECUTABLE}
                -DGEN=${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_vectors.py
                -DSELFTEST=$<TARGET_FILE:hnx_selftest>
                -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_vectors.cmake
        )
    endif()
endif()
