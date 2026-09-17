#pragma once

#include <cudalab.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cudalab/demo_catalog.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cudalab {

using FrameParams = CudalabParams;

struct CompileResult {
  bool ok = false;
  double milliseconds = 0.0;
  std::string log;
};

struct DeviceInfo {
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  int multiprocessors = 0;
  std::size_t memory_bytes = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

class CudaRuntime {
public:
  CudaRuntime();
  ~CudaRuntime();
  CudaRuntime(const CudaRuntime&) = delete;
  CudaRuntime& operator=(const CudaRuntime&) = delete;

  [[nodiscard]] const DeviceInfo& device() const {
    return device_;
  }
  CompileResult compile(const std::string& source, const std::filesystem::path& source_name);
  void configure(std::size_t state_bytes,
                 int work_items,
                 const std::vector<ResourceSpec>& resources = {},
                 bool use_graph = false);
  void reset();
  void resize(int width, int height);
  float render(const FrameParams& params);
  const std::vector<float2>&
  synthesize_audio(const FrameParams& params, int frames = 1024, int sample_rate = 48000);
  [[nodiscard]] bool has_audio() const {
    return audio_function_ != nullptr;
  }
  [[nodiscard]] unsigned int texture() const {
    return texture_;
  }
  [[nodiscard]] bool ready() const {
    return function_ != nullptr;
  }
  [[nodiscard]] bool graph_enabled() const {
    return graph_enabled_;
  }
  [[nodiscard]] std::size_t resource_count() const {
    return resources_.size();
  }

private:
  struct ResourceAllocation {
    ResourceSpec spec;
    void* data = nullptr;
    cudaArray_t array = nullptr;
    cudaTextureObject_t texture = 0;
    cudaSurfaceObject_t surface = 0;
  };

  void release_module();
  void release_surface();
  void release_resources();
  void release_graph();
  void clear_resources(cudaStream_t stream);
  void launch_render_graph(void* pixels, const FrameParams& params, unsigned work);

  DeviceInfo device_;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  CUfunction reset_function_ = nullptr;
  CUfunction simulate_function_ = nullptr;
  CUfunction composite_function_ = nullptr;
  CUfunction audio_function_ = nullptr;
  unsigned int texture_ = 0;
  unsigned int pbo_ = 0;
  cudaGraphicsResource_t graphics_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  cudaStream_t render_stream_ = nullptr;
  cudaStream_t simulation_stream_ = nullptr;
  cudaStream_t audio_stream_ = nullptr;
  cudaEvent_t simulation_done_ = nullptr;
  void* state_ = nullptr;
  std::size_t state_bytes_ = 0;
  int work_items_ = 0;
  std::vector<ResourceAllocation> resources_;
  CudalabResource* resource_table_ = nullptr;
  int resource_count_ = 0;
  bool reset_pending_ = false;
  bool graph_enabled_ = false;
  CUgraph graph_ = nullptr;
  CUgraphExec graph_exec_ = nullptr;
  CUgraphNode render_node_ = nullptr;
  CUgraphNode composite_node_ = nullptr;
  float2* audio_device_ = nullptr;
  std::vector<float2> audio_host_;
  unsigned long long audio_offset_ = 0;
  int width_ = 0;
  int height_ = 0;
};

} // namespace cudalab
