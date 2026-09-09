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

add_library(ggml-hnx STATIC ggml-hnx.c)
target_include_directories(ggml-hnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# -fPIC because ggml is often built shared, and a static library linked
# into a shared one needs it.
set_property(TARGET ggml-hnx PROPERTY POSITION_INDEPENDENT_CODE ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ggml-hnx PRIVATE -Wall -Wextra -Wpedantic)
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
