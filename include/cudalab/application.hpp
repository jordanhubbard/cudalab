#pragma once

#include "cudalab/cuda_runtime.hpp"
#include "cudalab/demo_catalog.hpp"

#include <SDL3/SDL_video.h>

struct SDL_AudioStream;

#include <array>
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
  int interaction_probe(const std::filesystem::path& output_directory);
  bool select_demo(const std::string& demo_name);
  int snapshot(const std::string& demo_name,
               float time,
               const std::filesystem::path& output_path,
               float mouse_x = .58f,
               float mouse_y = .43f,
               int beaufort = 4);

private:
  void draw_dockspace();
  void draw_catalog();
  void draw_editor();
  void draw_preview();
  void draw_output();
  void load_demo(std::size_t index);
  void compile();
  void save();
  void request_new_piece(bool clone);
  bool create_piece();
  void capture_frame();
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
  float mouse_x_ = 0.5f;
  float mouse_y_ = 0.5f;
  float mouse_dx_ = 0.0f;
  float mouse_dy_ = 0.0f;
  int mouse_down_ = 0;
  int frame_ = 0;
  int quality_ = 2;
  int beaufort_ = 4;
  int preview_width_ = 0;
  int preview_height_ = 0;
  std::array<char, 64> new_piece_name_{};
  std::array<char, 128> new_piece_title_{};
  bool new_piece_popup_ = false;
  bool clone_piece_ = false;
  bool paused_ = false;
  bool first_layout_ = true;
  bool running_ = true;
  bool last_compile_ok_ = false;
  bool audio_enabled_ = true;
  bool render_requested_ = true;
  bool preview_hovered_ = false;
};

} // namespace cudalab
