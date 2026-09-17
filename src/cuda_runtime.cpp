#include "cudalab/cuda_runtime.hpp"

#include <SDL3/SDL_opengl.h>
#include <cuda_gl_interop.h>
#include <nvrtc.h>

#include <algorithm>
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

} // namespace

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
  cuda_check(cudaEventCreateWithFlags(&simulation_done_, cudaEventDisableTiming),
             "cudaEventCreate(simulation_done)");
  cuda_check(cudaStreamCreateWithFlags(&render_stream_, cudaStreamNonBlocking), "create render stream");
  cuda_check(cudaStreamCreateWithFlags(&simulation_stream_, cudaStreamNonBlocking),
             "create simulation stream");
  cuda_check(cudaStreamCreateWithFlags(&audio_stream_, cudaStreamNonBlocking), "create audio stream");
}

CudaRuntime::~CudaRuntime() {
  release_surface();
  release_module();
  release_resources();
  if (state_)
    cudaFreeAsync(state_, simulation_stream_);
  if (audio_device_)
    cudaFreeAsync(audio_device_, audio_stream_);
  if (simulation_stream_)
    cudaStreamSynchronize(simulation_stream_);
  if (audio_stream_)
    cudaStreamSynchronize(audio_stream_);
  if (render_stream_)
    cudaStreamDestroy(render_stream_);
  if (simulation_stream_)
    cudaStreamDestroy(simulation_stream_);
  if (audio_stream_)
    cudaStreamDestroy(audio_stream_);
  if (simulation_done_)
    cudaEventDestroy(simulation_done_);
  if (start_)
    cudaEventDestroy(start_);
  if (stop_)
    cudaEventDestroy(stop_);
}

CompileResult CudaRuntime::compile(const std::string& source, const std::filesystem::path& source_name) {
  const auto before = std::chrono::steady_clock::now();
  CompileResult result;
  nvrtcProgram program = nullptr;
  try {
    nvrtc_check(nvrtcCreateProgram(
                    &program, source.c_str(), source_name.filename().string().c_str(), 0, nullptr, nullptr),
                "nvrtcCreateProgram");
    const std::string arch = "--gpu-architecture=compute_" + std::to_string(device_.compute_major) +
                             std::to_string(device_.compute_minor);
    const std::string project_include = std::string("--include-path=") + CUDALAB_SOURCE_ROOT + "/include";
    const std::string cuda_include = std::string("--include-path=") + CUDALAB_CUDA_INCLUDE_DIR;
    const std::string cccl_include = std::string("--include-path=") + CUDALAB_CCCL_INCLUDE_DIR;
    const std::vector<const char*> options = {"--std=c++20",
                                              arch.c_str(),
                                              "--use_fast_math",
                                              "--extra-device-vectorization",
                                              "--restrict",
                                              "--device-as-default-execution-space",
                                              "--brief-diagnostics=true",
                                              "--diag-suppress=1444",
                                              project_include.c_str(),
                                              cuda_include.c_str(),
                                              cccl_include.c_str()};
    const nvrtcResult status = nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
      result.log.resize(log_size);
      nvrtcGetProgramLog(program, result.log.data());
      if (!result.log.empty() && result.log.back() == '\0')
        result.log.pop_back();
    }
    if (status != NVRTC_SUCCESS) {
      result.log = "NVRTC " + std::string(nvrtcGetErrorString(status)) + "\n" + result.log;
      nvrtcDestroyProgram(&program);
      result.milliseconds =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
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
    CUfunction reset_function = nullptr;
    CUfunction simulate_function = nullptr;
    CUfunction composite_function = nullptr;
    CUfunction audio_function = nullptr;
    cuModuleGetFunction(&reset_function, candidate, "reset");
    cuModuleGetFunction(&simulate_function, candidate, "simulate");
    cuModuleGetFunction(&composite_function, candidate, "composite");
    cuModuleGetFunction(&audio_function, candidate, "audio");
    release_module();
    module_ = candidate;
    function_ = function;
    reset_function_ = reset_function;
    simulate_function_ = simulate_function;
    composite_function_ = composite_function;
    audio_function_ = audio_function;
    audio_offset_ = 0;
    reset_pending_ = true;
    result.ok = true;
    if (result.log.empty())
      result.log = "Compiled cleanly for " + arch.substr(19) + ".";
    nvrtcDestroyProgram(&program);
  } catch (const std::exception& error) {
    if (program)
      nvrtcDestroyProgram(&program);
    result.log += (result.log.empty() ? "" : "\n") + std::string(error.what());
  }
  result.milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
  return result;
}

void CudaRuntime::configure(std::size_t state_bytes,
                            int work_items,
                            const std::vector<ResourceSpec>& resources,
                            bool use_graph) {
  cuda_check(cudaStreamSynchronize(render_stream_), "finish render before reconfiguration");
  cuda_check(cudaStreamSynchronize(simulation_stream_), "finish simulation before reconfiguration");
  cuda_check(cudaStreamSynchronize(audio_stream_), "finish audio before reconfiguration");
  release_graph();
  release_resources();
  work_items_ = work_items;
  graph_enabled_ = use_graph;
  if (state_) {
    cuda_check(cudaFreeAsync(state_, simulation_stream_), "release demo state");
    state_ = nullptr;
  }
  state_bytes_ = state_bytes;
  if (state_bytes_) {
    cuda_check(cudaMallocAsync(&state_, state_bytes_, simulation_stream_), "allocate persistent demo state");
    cuda_check(cudaMemsetAsync(state_, 0, state_bytes_, simulation_stream_), "clear persistent demo state");
  }

  std::vector<CudalabResource> table;
  table.reserve(resources.size());
  for (const auto& spec : resources) {
    ResourceAllocation allocation;
    allocation.spec = spec;
    CudalabResource descriptor{};
    if (spec.kind == ResourceKind::buffer) {
      cuda_check(cudaMallocAsync(&allocation.data, spec.bytes, simulation_stream_), "allocate named buffer");
      descriptor.data = allocation.data;
      descriptor.bytes = spec.bytes;
    } else {
      const auto channel = cudaCreateChannelDesc<float4>();
      cuda_check(
          cudaMallocArray(&allocation.array, &channel, spec.width, spec.height, cudaArraySurfaceLoadStore),
          "allocate named surface");
      cudaResourceDesc resource{};
      resource.resType = cudaResourceTypeArray;
      resource.res.array.array = allocation.array;
      cudaTextureDesc texture{};
      texture.addressMode[0] = cudaAddressModeWrap;
      texture.addressMode[1] = cudaAddressModeWrap;
      texture.filterMode = cudaFilterModeLinear;
      texture.readMode = cudaReadModeElementType;
      texture.normalizedCoords = 1;
      cuda_check(cudaCreateTextureObject(&allocation.texture, &resource, &texture, nullptr),
                 "create named texture");
      cuda_check(cudaCreateSurfaceObject(&allocation.surface, &resource), "create named surface object");
      descriptor.bytes = static_cast<unsigned long long>(spec.width) * spec.height * sizeof(float4);
      descriptor.texture = allocation.texture;
      descriptor.surface = allocation.surface;
      descriptor.width = spec.width;
      descriptor.height = spec.height;
    }
    table.push_back(descriptor);
    resources_.push_back(std::move(allocation));
  }
  resource_count_ = static_cast<int>(table.size());
  if (!table.empty()) {
    cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&resource_table_),
                               table.size() * sizeof(CudalabResource),
                               simulation_stream_),
               "allocate resource table");
    cuda_check(cudaMemcpyAsync(resource_table_,
                               table.data(),
                               table.size() * sizeof(CudalabResource),
                               cudaMemcpyHostToDevice,
                               simulation_stream_),
               "upload resource table");
  }
  clear_resources(simulation_stream_);
  reset_pending_ = true;
}

void CudaRuntime::reset() {
  if (state_)
    cuda_check(cudaMemsetAsync(state_, 0, state_bytes_, simulation_stream_), "clear demo state");
  clear_resources(simulation_stream_);
  audio_offset_ = 0;
  reset_pending_ = true;
}

void CudaRuntime::resize(int width, int height) {
  if (width <= 0 || height <= 0 || (width == width_ && height == height_))
    return;
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
  glBufferData(
      GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * 4, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  cuda_check(cudaGraphicsGLRegisterBuffer(&graphics_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard),
             "register CUDA/OpenGL pixel buffer");
}

float CudaRuntime::render(const FrameParams& params) {
  if (!function_ || !graphics_)
    return 0.0f;
  cuda_check(cudaGraphicsMapResources(1, &graphics_, render_stream_), "map CUDA/OpenGL resource");
  void* pixels = nullptr;
  std::size_t bytes = 0;
  cuda_check(cudaGraphicsResourceGetMappedPointer(&pixels, &bytes, graphics_), "get mapped pixel buffer");
  const auto state_bytes = static_cast<unsigned long long>(state_bytes_);
  void* stage_args[] = {&state_,
                        const_cast<unsigned long long*>(&state_bytes),
                        const_cast<FrameParams*>(&params),
                        &resource_table_,
                        &resource_count_};
  void* render_args[] = {&pixels,
                         &state_,
                         const_cast<unsigned long long*>(&state_bytes),
                         const_cast<FrameParams*>(&params),
                         &resource_table_,
                         &resource_count_};
  const unsigned block_x = 16;
  const unsigned block_y = 16;
  const unsigned work = work_items_ > 0 ? static_cast<unsigned>(work_items_)
                                        : static_cast<unsigned>(params.width * params.height);
  if (reset_pending_) {
    if (state_)
      cuda_check(cudaMemsetAsync(state_, 0, state_bytes_, simulation_stream_), "reset persistent demo state");
    clear_resources(simulation_stream_);
    if (reset_function_)
      driver_check(cuLaunchKernel(reset_function_,
                                  (work + 255) / 256,
                                  1,
                                  1,
                                  256,
                                  1,
                                  1,
                                  0,
                                  reinterpret_cast<CUstream>(simulation_stream_),
                                  stage_args,
                                  nullptr),
                   "launch reset kernel");
    reset_pending_ = false;
  }
  if (simulate_function_)
    driver_check(cuLaunchKernel(simulate_function_,
                                (work + 255) / 256,
                                1,
                                1,
                                256,
                                1,
                                1,
                                0,
                                reinterpret_cast<CUstream>(simulation_stream_),
                                stage_args,
                                nullptr),
                 "launch simulate kernel");
  cuda_check(cudaEventRecord(simulation_done_, simulation_stream_), "record simulation completion");
  cuda_check(cudaStreamWaitEvent(render_stream_, simulation_done_), "wait for simulation");
  cuda_check(cudaEventRecord(start_, render_stream_), "record start event");
  if (graph_enabled_) {
    launch_render_graph(pixels, params, work);
  } else {
    driver_check(cuLaunchKernel(function_,
                                (params.width + block_x - 1) / block_x,
                                (params.height + block_y - 1) / block_y,
                                1,
                                block_x,
                                block_y,
                                1,
                                0,
                                reinterpret_cast<CUstream>(render_stream_),
                                render_args,
                                nullptr),
                 "launch render kernel");
    if (composite_function_)
      driver_check(cuLaunchKernel(composite_function_,
                                  (work + 255) / 256,
                                  1,
                                  1,
                                  256,
                                  1,
                                  1,
                                  0,
                                  reinterpret_cast<CUstream>(render_stream_),
                                  render_args,
                                  nullptr),
                   "launch composite kernel");
  }
  cuda_check(cudaEventRecord(stop_, render_stream_), "record stop event");
  cuda_check(cudaEventSynchronize(stop_), "wait for render kernel");
  float milliseconds = 0.0f;
  cuda_check(cudaEventElapsedTime(&milliseconds, start_, stop_), "measure render kernel");
  cuda_check(cudaGraphicsUnmapResources(1, &graphics_, render_stream_), "unmap CUDA/OpenGL resource");
  cuda_check(cudaStreamSynchronize(render_stream_), "finish CUDA/OpenGL frame");
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  return milliseconds;
}

const std::vector<float2>&
CudaRuntime::synthesize_audio(const FrameParams& params, int frames, int sample_rate) {
  if (!audio_function_ || frames <= 0) {
    audio_host_.clear();
    return audio_host_;
  }
  frames = std::min(frames, 4096);
  audio_host_.resize(frames);
  if (audio_offset_ == 0 && params.time > 0.0f) {
    audio_offset_ = static_cast<unsigned long long>(params.time * sample_rate);
  }
  if (!audio_device_)
    cuda_check(
        cudaMallocAsync(reinterpret_cast<void**>(&audio_device_), sizeof(float2) * 4096, audio_stream_),
        "allocate GPU audio buffer");
  const auto state_bytes = static_cast<unsigned long long>(state_bytes_);
  void* args[] = {&audio_device_,
                  &state_,
                  const_cast<unsigned long long*>(&state_bytes),
                  const_cast<FrameParams*>(&params),
                  &audio_offset_,
                  &frames,
                  &sample_rate,
                  &resource_table_,
                  &resource_count_};
  cuda_check(cudaStreamWaitEvent(audio_stream_, simulation_done_), "wait for simulation before audio");
  driver_check(cuLaunchKernel(audio_function_,
                              (frames + 255) / 256,
                              1,
                              1,
                              256,
                              1,
                              1,
                              0,
                              reinterpret_cast<CUstream>(audio_stream_),
                              args,
                              nullptr),
               "launch audio kernel");
  cuda_check(
      cudaMemcpyAsync(
          audio_host_.data(), audio_device_, sizeof(float2) * frames, cudaMemcpyDeviceToHost, audio_stream_),
      "copy GPU audio");
  cuda_check(cudaStreamSynchronize(audio_stream_), "finish GPU audio");
  audio_offset_ += frames;
  return audio_host_;
}

void CudaRuntime::release_module() {
  release_graph();
  function_ = nullptr;
  reset_function_ = simulate_function_ = composite_function_ = audio_function_ = nullptr;
  if (module_)
    cuModuleUnload(std::exchange(module_, nullptr));
}

void CudaRuntime::release_surface() {
  if (render_stream_)
    cudaStreamSynchronize(render_stream_);
  if (graphics_)
    cudaGraphicsUnregisterResource(std::exchange(graphics_, nullptr));
  if (pbo_)
    glDeleteBuffers(1, &pbo_);
  if (texture_)
    glDeleteTextures(1, &texture_);
  pbo_ = texture_ = 0;
  width_ = height_ = 0;
}

void CudaRuntime::clear_resources(cudaStream_t stream) {
  for (const auto& resource : resources_) {
    if (resource.data) {
      cuda_check(cudaMemsetAsync(resource.data, 0, resource.spec.bytes, stream), "clear named buffer");
    } else if (resource.array) {
      const auto row_bytes = static_cast<std::size_t>(resource.spec.width) * sizeof(float4);
      std::vector<float4> zeros(static_cast<std::size_t>(resource.spec.width) * resource.spec.height);
      cuda_check(cudaMemcpy2DToArray(resource.array,
                                     0,
                                     0,
                                     zeros.data(),
                                     row_bytes,
                                     row_bytes,
                                     resource.spec.height,
                                     cudaMemcpyHostToDevice),
                 "clear named surface");
    }
  }
}

void CudaRuntime::release_resources() {
  if (resources_.empty() && !resource_table_)
    return;
  if (render_stream_)
    cudaStreamSynchronize(render_stream_);
  if (simulation_stream_)
    cudaStreamSynchronize(simulation_stream_);
  if (audio_stream_)
    cudaStreamSynchronize(audio_stream_);
  for (auto& resource : resources_) {
    if (resource.surface)
      cudaDestroySurfaceObject(resource.surface);
    if (resource.texture)
      cudaDestroyTextureObject(resource.texture);
    if (resource.array)
      cudaFreeArray(resource.array);
    if (resource.data)
      cudaFreeAsync(resource.data, simulation_stream_);
  }
  resources_.clear();
  if (resource_table_) {
    cudaFreeAsync(resource_table_, simulation_stream_);
    resource_table_ = nullptr;
  }
  resource_count_ = 0;
  if (simulation_stream_)
    cudaStreamSynchronize(simulation_stream_);
}

void CudaRuntime::release_graph() {
  if (graph_exec_)
    cuGraphExecDestroy(std::exchange(graph_exec_, nullptr));
  if (graph_)
    cuGraphDestroy(std::exchange(graph_, nullptr));
  render_node_ = composite_node_ = nullptr;
}

void CudaRuntime::launch_render_graph(void* pixels, const FrameParams& params, unsigned work) {
  const auto state_bytes = static_cast<unsigned long long>(state_bytes_);
  void* arguments[] = {&pixels,
                       &state_,
                       const_cast<unsigned long long*>(&state_bytes),
                       const_cast<FrameParams*>(&params),
                       &resource_table_,
                       &resource_count_};
  CUDA_KERNEL_NODE_PARAMS render_params{};
  render_params.func = function_;
  render_params.gridDimX = (params.width + 15) / 16;
  render_params.gridDimY = (params.height + 15) / 16;
  render_params.gridDimZ = 1;
  render_params.blockDimX = 16;
  render_params.blockDimY = 16;
  render_params.blockDimZ = 1;
  render_params.kernelParams = arguments;

  CUDA_KERNEL_NODE_PARAMS composite_params{};
  composite_params.func = composite_function_;
  composite_params.gridDimX = (work + 255) / 256;
  composite_params.gridDimY = 1;
  composite_params.gridDimZ = 1;
  composite_params.blockDimX = 256;
  composite_params.blockDimY = 1;
  composite_params.blockDimZ = 1;
  composite_params.kernelParams = arguments;

  if (!graph_exec_) {
    driver_check(cuGraphCreate(&graph_, 0), "create render graph");
    driver_check(cuGraphAddKernelNode(&render_node_, graph_, nullptr, 0, &render_params),
                 "add graph render node");
    if (composite_function_) {
      driver_check(cuGraphAddKernelNode(&composite_node_, graph_, &render_node_, 1, &composite_params),
                   "add graph composite node");
    }
    driver_check(cuGraphInstantiate(&graph_exec_, graph_, 0), "instantiate render graph");
  } else {
    driver_check(cuGraphExecKernelNodeSetParams(graph_exec_, render_node_, &render_params),
                 "update graph render node");
    if (composite_node_) {
      driver_check(cuGraphExecKernelNodeSetParams(graph_exec_, composite_node_, &composite_params),
                   "update graph composite node");
    }
  }
  driver_check(cuGraphLaunch(graph_exec_, reinterpret_cast<CUstream>(render_stream_)), "launch render graph");
}

} // namespace cudalab
