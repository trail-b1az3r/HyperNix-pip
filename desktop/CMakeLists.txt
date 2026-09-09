# HyperNix Studio — a desktop client for a HyperNix server.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/hypernix-studio
#
# Qt 6.4 or newer. On Debian/Ubuntu:
#   sudo apt install qt6-base-dev qt6-declarative-dev cmake g++
# On Fedora:
#   sudo dnf install qt6-qtbase-devel qt6-qtdeclarative-devel cmake gcc-c++
# On Arch:
#   sudo pacman -S qt6-base qt6-declarative cmake
#
# The two libraries build and test WITHOUT Qt
# ------------------------------------------
# `hnx-studio-core` is the security-critical half -- the tool policy and
# the runner -- in plain C++17 with no Qt at all, and `ctest` runs its
# tests. That is deliberate: the part that stands between a language
# model and somebody's filesystem should be checkable with a compiler
# and nothing else, on any machine, including one with no Qt installed.
#
#   cmake -S . -B build -DSTUDIO_CORE_ONLY=ON
#   cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.19)
project(hypernix-studio VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(STUDIO_CORE_ONLY
       "Build and test only the Qt-free core. For CI and for machines with no Qt."
       OFF)

# Local models. OFF by default and that is the right default: the server
# path does not need llama.cpp, and making the harder dependency
# mandatory would stop Studio building for everyone who only wants to
# connect to a HyperNix box.
#
#   cmake -S . -B build -DSTUDIO_LOCAL_LLAMA=ON -DLLAMA_ROOT=~/llama.cpp
#
# LLAMA_ROOT wants a llama.cpp that has been built, and -- for the
# HyperNix sub-bit types -- one that native/ggml-hnx/tools/patch_llamacpp.py
# has been run against. Without the patch a stock build loads everything
# else and refuses IQ0.5_XXXL, which LocalEngine reports in as many words.
option(STUDIO_LOCAL_LLAMA
       "Run models in-process via llama.cpp, with no server."
       OFF)
set(LLAMA_ROOT "" CACHE PATH "A built llama.cpp checkout.")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
endif()

# --- the core: no Qt ---------------------------------------------------
add_library(hnx-studio-core STATIC
    src/ToolPolicy.cpp
    src/ToolRunner.cpp
    src/ModelCatalogue.cpp
    src/LocalEngine.cpp
)
target_include_directories(hnx-studio-core PUBLIC src)

# --- local inference, optional ----------------------------------------
if(STUDIO_LOCAL_LLAMA)
    if(NOT LLAMA_ROOT)
        message(FATAL_ERROR
            "STUDIO_LOCAL_LLAMA=ON needs -DLLAMA_ROOT=/path/to/llama.cpp "
            "(a checkout that has been built). Said here rather than "
            "failing later with a missing llama.h.")
    endif()
    find_path(LLAMA_INCLUDE_DIR llama.h
              HINTS "${LLAMA_ROOT}/include" "${LLAMA_ROOT}"
              NO_DEFAULT_PATH)
    find_library(LLAMA_LIBRARY NAMES llama
                 HINTS "${LLAMA_ROOT}/build/bin" "${LLAMA_ROOT}/build"
                       "${LLAMA_ROOT}/build/src"
                 NO_DEFAULT_PATH)
    find_library(GGML_BASE_LIBRARY NAMES ggml-base
                 HINTS "${LLAMA_ROOT}/build/bin" "${LLAMA_ROOT}/build"
                 NO_DEFAULT_PATH)
    find_library(GGML_LIBRARY NAMES ggml
                 HINTS "${LLAMA_ROOT}/build/bin" "${LLAMA_ROOT}/build"
                 NO_DEFAULT_PATH)
    if(NOT LLAMA_INCLUDE_DIR OR NOT LLAMA_LIBRARY)
        message(FATAL_ERROR
            "STUDIO_LOCAL_LLAMA=ON but ${LLAMA_ROOT} has no llama.h or no "
            "built libllama. Build llama.cpp first: cmake -S . -B build && "
            "cmake --build build -j")
    endif()
    target_compile_definitions(hnx-studio-core PUBLIC HNX_STUDIO_LOCAL_LLAMA)
    target_include_directories(hnx-studio-core PUBLIC
        "${LLAMA_INCLUDE_DIR}" "${LLAMA_ROOT}/ggml/include")
    target_link_libraries(hnx-studio-core PUBLIC
        "${LLAMA_LIBRARY}" "${GGML_BASE_LIBRARY}" "${GGML_LIBRARY}")
    message(STATUS "Studio: local inference ON (${LLAMA_ROOT})")
else()
    message(STATUS "Studio: local inference OFF "
                   "(-DSTUDIO_LOCAL_LLAMA=ON -DLLAMA_ROOT=... to enable)")
endif()

find_package(Threads REQUIRED)
target_link_libraries(hnx-studio-core PUBLIC Threads::Threads)

enable_testing()

add_executable(tool_policy_test tests/tool_policy_test.cpp)
target_link_libraries(tool_policy_test PRIVATE hnx-studio-core)
add_test(NAME tool_policy COMMAND tool_policy_test)

add_executable(tool_runner_test tests/tool_runner_test.cpp)
target_link_libraries(tool_runner_test PRIVATE hnx-studio-core)
add_test(NAME tool_runner COMMAND tool_runner_test)

add_executable(model_catalogue_test tests/model_catalogue_test.cpp)
target_link_libraries(model_catalogue_test PRIVATE hnx-studio-core)
add_test(NAME model_catalogue COMMAND model_catalogue_test)

# Runs in both builds and checks different things in each: with
# llama.cpp it loads a model and generates; without, it checks the stub
# refuses honestly rather than pretending. A test that only existed in
# one configuration would let the other rot.
add_executable(local_engine_test tests/local_engine_test.cpp)
target_link_libraries(local_engine_test PRIVATE hnx-studio-core)
add_test(NAME local_engine COMMAND local_engine_test)

if(STUDIO_CORE_ONLY)
    message(STATUS "STUDIO_CORE_ONLY: skipping the Qt application")
    return()
endif()

# --- the application --------------------------------------------------
# 6.4, not 6.5. Nothing here needs a 6.5 API, and 6.4 is what Ubuntu
# 24.04 and Debian 12 ship -- requiring 6.5 meant the app would not
# configure on the distributions most likely to run it, for no gain.
# Raise it when something actually needs a newer API, and say what.
find_package(Qt6 6.4 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Network)

qt_standard_project_setup(REQUIRES 6.4)

qt_add_executable(hypernix-studio
    src/main.cpp
    src/HyperLinkClient.cpp
    src/LocalSession.cpp
    src/StudioBridge.cpp
)

qt_add_resources(hypernix-studio "studio"
    PREFIX "/"
    BASE "."
    FILES
        qml/Main.qml
        qml/Theme.qml
        qml/qmldir
        qml/Sidebar.qml
        qml/ConnectPanel.qml
        qml/ChatView.qml
        qml/MessageBubble.qml
        qml/CodeView.qml
        qml/ModelsView.qml
        qml/MachineView.qml
        qml/Meter.qml
        qml/ToolApproval.qml
        qml/StudioButton.qml
        qml/StudioField.qml
        qml/ErrorBanner.qml
)

target_link_libraries(hypernix-studio PRIVATE
    hnx-studio-core
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Network
)

set_target_properties(hypernix-studio PROPERTIES
    WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON
)

include(GNUInstallDirs)
install(TARGETS hypernix-studio RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES resources/hypernix-studio.desktop
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/applications)
install(FILES ../assets/logo-new/hypernix-icon.svg
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/icons/hicolor/scalable/apps
        RENAME hypernix-studio.svg)
