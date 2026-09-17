#pragma once

#include "cudalab/cuda_runtime.hpp"
#include "cudalab/demo_catalog.hpp"

#include <SDL3/SDL_video.h>

struct SDL_AudioStream;

#include <filesystem>
#include <memory>
#include <string>

class TextEditor;

namespace cudalab {

class Application {
 public:
  explicit Application(bool hidden = false);
  ~Application();
  int run();
  int smoke_test();

 private:
  void draw_dockspace();
  void draw_catalog();
  void draw_editor();
  void draw_preview();
  void draw_output();
  void load_demo(std::size_t index);
  void compile();
  void save();
  void format();
  void update_diagnostic_markers(const std::string& log);
  void reset_layout();
  static std::filesystem::path find_examples();

  SDL_Window* window_ = nullptr;
  SDL_GLContext gl_context_ = nullptr;
  SDL_AudioStream* audio_stream_ = nullptr;
  DemoCatalog catalog_;
  std::unique_ptr<CudaRuntime> cuda_;
  std::unique_ptr<TextEditor> editor_;
  std::size_t selected_ = 0;
  std::filesystem::path source_path_;
  std::string source_;
  std::string saved_source_;
  std::string output_;
  std::string filter_;
  float elapsed_ = 0.0f;
  float delta_ = 0.0f;
  float gpu_ms_ = 0.0f;
  float mouse_x_ = 0.0f;
  float mouse_y_ = 0.0f;
  int frame_ = 0;
  int quality_ = 2;
  bool paused_ = false;
  bool first_layout_ = true;
  bool running_ = true;
  bool last_compile_ok_ = false;
  bool audio_enabled_ = true;
};

}  // namespace cudalab
