#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <filesystem>
#include <string>
#include <vector>

namespace cudalab {

struct FrameParams {
  int width;
  int height;
  float time;
  float delta;
  float mouse_x;
  float mouse_y;
  int frame;
  int quality;
};

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

  [[nodiscard]] const DeviceInfo& device() const { return device_; }
  CompileResult compile(const std::string& source, const std::filesystem::path& source_name);
  void resize(int width, int height);
  float render(const FrameParams& params);
  [[nodiscard]] unsigned int texture() const { return texture_; }
  [[nodiscard]] bool ready() const { return function_ != nullptr; }

 private:
  void release_module();
  void release_surface();

  DeviceInfo device_;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
  unsigned int texture_ = 0;
  unsigned int pbo_ = 0;
  cudaGraphicsResource_t graphics_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace cudalab
