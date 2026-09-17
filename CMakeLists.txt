cmake_minimum_required(VERSION 3.28)
project(cudalab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUDALAB_NATIVE_ARCH "Build the host for this machine's CPU" OFF)

find_package(CUDAToolkit 13.0 REQUIRED)
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDALAB_CUDA_INCLUDE_DIR)
list(LENGTH CUDAToolkit_INCLUDE_DIRS CUDALAB_CUDA_INCLUDE_COUNT)
if(CUDALAB_CUDA_INCLUDE_COUNT GREATER 1)
  list(GET CUDAToolkit_INCLUDE_DIRS 1 CUDALAB_CCCL_INCLUDE_DIR)
else()
  set(CUDALAB_CCCL_INCLUDE_DIR "${CUDALAB_CUDA_INCLUDE_DIR}")
endif()
find_package(OpenGL REQUIRED)
include(FetchContent)
find_package(SDL3 CONFIG QUIET)
if(NOT SDL3_FOUND)
  FetchContent_Declare(
    SDL3
    URL https://github.com/libsdl-org/SDL/archive/refs/tags/release-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
  set(SDL_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(SDL3)
endif()

FetchContent_Declare(
  imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.4-docking.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(cudalab_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(cudalab_imgui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(cudalab_imgui PUBLIC SDL3::SDL3 OpenGL::GL)
target_compile_definitions(cudalab_imgui PRIVATE GL_GLEXT_PROTOTYPES)

add_executable(cudalab
  src/main.cpp
  src/application.cpp
  src/cuda_runtime.cpp
  src/demo_catalog.cpp
)
target_include_directories(cudalab PRIVATE include)
target_compile_definitions(cudalab PRIVATE
  GL_GLEXT_PROTOTYPES
  CUDALAB_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
  CUDALAB_CUDA_INCLUDE_DIR="${CUDALAB_CUDA_INCLUDE_DIR}"
  CUDALAB_CCCL_INCLUDE_DIR="${CUDALAB_CCCL_INCLUDE_DIR}"
  CUDALAB_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(cudalab PRIVATE
  cudalab_imgui
  SDL3::SDL3
  OpenGL::GL
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::nvrtc
)

if(MSVC)
  target_compile_options(cudalab PRIVATE /W4 /permissive- /Zc:__cplusplus)
else()
  target_compile_options(cudalab PRIVATE -Wall -Wextra -Wpedantic)
  if(CUDALAB_NATIVE_ARCH)
    target_compile_options(cudalab PRIVATE -march=native)
  endif()
endif()

add_custom_command(TARGET cudalab POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/examples
          $<TARGET_FILE_DIR:cudalab>/examples
)

include(CTest)
if(BUILD_TESTING)
  add_executable(cudalab_tests tests/catalog_tests.cpp src/demo_catalog.cpp)
  target_include_directories(cudalab_tests PRIVATE include)
  target_compile_definitions(cudalab_tests PRIVATE CUDALAB_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME catalog COMMAND cudalab_tests)
endif()

install(TARGETS cudalab RUNTIME DESTINATION bin BUNDLE DESTINATION .)
install(DIRECTORY examples DESTINATION share/cudalab)
