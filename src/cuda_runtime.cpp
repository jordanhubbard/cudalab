#include "cudalab/cuda_runtime.hpp"

#include <SDL3/SDL_opengl.h>
#include <cuda_gl_interop.h>
#include <nvrtc.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cudalab {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void driver_check(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* detail = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &detail);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA error") +
                             (detail ? std::string(" — ") + detail : ""));
  }
}

void nvrtc_check(nvrtcResult result, const char* operation) {
  if (result != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " + nvrtcGetErrorString(result));
  }
}

}  // namespace

CudaRuntime::CudaRuntime() {
  // This must precede any call that implicitly creates a CUDA runtime context.
  // The context then inherits interoperability with the active OpenGL context.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  cuda_check(cudaGLSetGLDevice(0), "bind CUDA device to OpenGL context");
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  cuda_check(cudaFree(nullptr), "initialize CUDA runtime");
  int device = 0;
  cuda_check(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  device_.name = prop.name;
  device_.compute_major = prop.major;
  device_.compute_minor = prop.minor;
  device_.multiprocessors = prop.multiProcessorCount;
  device_.memory_bytes = prop.totalGlobalMem;
  cudaDriverGetVersion(&device_.driver_version);
  cudaRuntimeGetVersion(&device_.runtime_version);
  cuda_check(cudaEventCreate(&start_), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
}

CudaRuntime::~CudaRuntime() {
  release_surface();
  release_module();
  if (start_) cudaEventDestroy(start_);
  if (stop_) cudaEventDestroy(stop_);
}

CompileResult CudaRuntime::compile(const std::string& source, const std::filesystem::path& source_name) {
  const auto before = std::chrono::steady_clock::now();
  CompileResult result;
  nvrtcProgram program = nullptr;
  try {
    nvrtc_check(nvrtcCreateProgram(&program, source.c_str(), source_name.filename().string().c_str(),
                                   0, nullptr, nullptr), "nvrtcCreateProgram");
    const std::string arch = "--gpu-architecture=compute_" + std::to_string(device_.compute_major) +
                             std::to_string(device_.compute_minor);
    const std::string project_include = std::string("--include-path=") + CUDALAB_SOURCE_ROOT + "/include";
    const std::string cuda_include = std::string("--include-path=") + CUDALAB_CUDA_INCLUDE_DIR;
    const std::vector<const char*> options = {
      "--std=c++20", arch.c_str(), "--use_fast_math", "--extra-device-vectorization",
      "--restrict", "--device-as-default-execution-space", "--brief-diagnostics=true",
      project_include.c_str(), cuda_include.c_str()
    };
    const nvrtcResult status = nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
      result.log.resize(log_size);
      nvrtcGetProgramLog(program, result.log.data());
      if (!result.log.empty() && result.log.back() == '\0') result.log.pop_back();
    }
    if (status != NVRTC_SUCCESS) {
      result.log = "NVRTC " + std::string(nvrtcGetErrorString(status)) + "\n" + result.log;
      nvrtcDestroyProgram(&program);
      result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
      return result;
    }
    std::size_t ptx_size = 0;
    nvrtc_check(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
    std::vector<char> ptx(ptx_size);
    nvrtc_check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");

    CUmodule candidate = nullptr;
    CUfunction function = nullptr;
    driver_check(cuModuleLoadData(&candidate, ptx.data()), "cuModuleLoadData");
    const auto function_status = cuModuleGetFunction(&function, candidate, "render");
    if (function_status != CUDA_SUCCESS) {
      cuModuleUnload(candidate);
      driver_check(function_status, "find required render kernel");
    }
    release_module();
    module_ = candidate;
    function_ = function;
    result.ok = true;
    if (result.log.empty()) result.log = "Compiled cleanly for " + arch.substr(19) + ".";
    nvrtcDestroyProgram(&program);
  } catch (const std::exception& error) {
    if (program) nvrtcDestroyProgram(&program);
    result.log += (result.log.empty() ? "" : "\n") + std::string(error.what());
  }
  result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
  return result;
}

void CudaRuntime::resize(int width, int height) {
  if (width <= 0 || height <= 0 || (width == width_ && height == height_)) return;
  release_surface();
  width_ = width;
  height_ = height;
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glGenBuffers(1, &pbo_);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * 4, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  cuda_check(cudaGraphicsGLRegisterBuffer(&graphics_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard),
             "register CUDA/OpenGL pixel buffer");
}

float CudaRuntime::render(const FrameParams& params) {
  if (!function_ || !graphics_) return 0.0f;
  cuda_check(cudaGraphicsMapResources(1, &graphics_), "map CUDA/OpenGL resource");
  void* pixels = nullptr;
  std::size_t bytes = 0;
  cuda_check(cudaGraphicsResourceGetMappedPointer(&pixels, &bytes, graphics_), "get mapped pixel buffer");
  void* args[] = {&pixels, const_cast<FrameParams*>(&params)};
  const unsigned block_x = 16;
  const unsigned block_y = 16;
  cuda_check(cudaEventRecord(start_), "record start event");
  driver_check(cuLaunchKernel(function_, (params.width + block_x - 1) / block_x,
                              (params.height + block_y - 1) / block_y, 1,
                              block_x, block_y, 1, 0, nullptr, args, nullptr), "launch render kernel");
  cuda_check(cudaEventRecord(stop_), "record stop event");
  cuda_check(cudaEventSynchronize(stop_), "wait for render kernel");
  float milliseconds = 0.0f;
  cuda_check(cudaEventElapsedTime(&milliseconds, start_, stop_), "measure render kernel");
  cuda_check(cudaGraphicsUnmapResources(1, &graphics_), "unmap CUDA/OpenGL resource");
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  return milliseconds;
}

void CudaRuntime::release_module() {
  function_ = nullptr;
  if (module_) cuModuleUnload(std::exchange(module_, nullptr));
}

void CudaRuntime::release_surface() {
  if (graphics_) cudaGraphicsUnregisterResource(std::exchange(graphics_, nullptr));
  if (pbo_) glDeleteBuffers(1, &pbo_);
  if (texture_) glDeleteTextures(1, &texture_);
  pbo_ = texture_ = 0;
  width_ = height_ = 0;
}

}  // namespace cudalab
