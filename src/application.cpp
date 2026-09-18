#include "cudalab/application.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>
#include <TextEditor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace cudalab {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot read " + path.string());
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

std::string lower(std::string value) {
  std::ranges::transform(
      value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

const TextEditor::Language* cuda_language() {
  static const TextEditor::Language language = [] {
    auto result = *TextEditor::Language::Cpp();
    result.name = "CUDA C++";
    result.keywords.insert({"__device__",
                            "__global__",
                            "__host__",
                            "__shared__",
                            "__constant__",
                            "__managed__",
                            "__restrict__",
                            "__launch_bounds__",
                            "__grid_constant__",
                            "__forceinline__",
                            "__noinline__",
                            "__syncthreads",
                            "__syncwarp",
                            "__threadfence",
                            "__threadfence_block",
                            "__threadfence_system"});
    result.declarations.insert({"dim3",
                                "cudaStream_t",
                                "cudaEvent_t",
                                "cudaTextureObject_t",
                                "cudaSurfaceObject_t",
                                "half",
                                "half2",
                                "nv_bfloat16",
                                "float2",
                                "float3",
                                "float4",
                                "double2",
                                "int2",
                                "int3",
                                "int4",
                                "uint2",
                                "uint3",
                                "uint4",
                                "uchar4",
                                "CudalabParams",
                                "CudalabResource"});
    result.identifiers.insert({"threadIdx",
                               "blockIdx",
                               "blockDim",
                               "gridDim",
                               "warpSize",
                               "clock64",
                               "atomicAdd",
                               "atomicCAS",
                               "__shfl_sync",
                               "__shfl_down_sync",
                               "__ballot_sync",
                               "__activemask",
                               "cooperative_groups",
                               "cub",
                               "thrust",
                               "wmma",
                               "CUDALAB_RENDER",
                               "CUDALAB_SIMULATE",
                               "CUDALAB_RESET",
                               "CUDALAB_COMPOSITE",
                               "CUDALAB_AUDIO",
                               "cudalab_buffer",
                               "tex2D",
                               "surf2Dwrite"});
    return result;
  }();
  return &language;
}

int run_clang_format(const std::filesystem::path& path) {
  const auto filename = path.string();
#if defined(_WIN32)
  return static_cast<int>(
      _spawnlp(_P_WAIT, "clang-format", "clang-format", "-i", "--style=file", filename.c_str(), nullptr));
#else
  const char* arguments[] = {"clang-format", "-i", "--style=file", filename.c_str(), nullptr};
  pid_t process = 0;
  const int spawn_result =
      posix_spawnp(&process, "clang-format", nullptr, nullptr, const_cast<char* const*>(arguments), environ);
  if (spawn_result != 0)
    return spawn_result;
  int status = 0;
  if (waitpid(process, &status, 0) < 0 || !WIFEXITED(status))
    return -1;
  return WEXITSTATUS(status);
#endif
}

void set_studio_theme() {
  ImGui::StyleColorsDark();
  auto& style = ImGui::GetStyle();
  style.WindowRounding = 2.0f;
  style.FrameRounding = 3.0f;
  style.TabRounding = 3.0f;
  style.GrabRounding = 3.0f;
  style.WindowPadding = {8, 8};
  style.FramePadding = {7, 5};
  style.ItemSpacing = {7, 6};
  auto* c = style.Colors;
  c[ImGuiCol_WindowBg] = ImVec4(.055f, .06f, .095f, 1);
  c[ImGuiCol_ChildBg] = ImVec4(.07f, .075f, .12f, 1);
  c[ImGuiCol_PopupBg] = ImVec4(.075f, .08f, .135f, 1);
  c[ImGuiCol_Border] = ImVec4(.20f, .30f, .52f, .36f);
  c[ImGuiCol_FrameBg] = ImVec4(.035f, .04f, .075f, 1);
  c[ImGuiCol_FrameBgHovered] = ImVec4(.10f, .16f, .28f, 1);
  c[ImGuiCol_Header] = ImVec4(.13f, .24f, .48f, .60f);
  c[ImGuiCol_HeaderHovered] = ImVec4(.18f, .36f, .72f, .72f);
  c[ImGuiCol_HeaderActive] = ImVec4(.20f, .40f, .82f, .88f);
  c[ImGuiCol_Button] = ImVec4(.12f, .25f, .50f, .70f);
  c[ImGuiCol_ButtonHovered] = ImVec4(.20f, .40f, .82f, 1);
  c[ImGuiCol_Tab] = ImVec4(.08f, .11f, .20f, 1);
  c[ImGuiCol_TabSelected] = ImVec4(.12f, .27f, .55f, 1);
  c[ImGuiCol_TitleBgActive] = ImVec4(.07f, .10f, .19f, 1);
  c[ImGuiCol_DockingPreview] = ImVec4(.30f, .56f, 1, .72f);
}

} // namespace

Application::Application(bool hidden) : catalog_(DemoCatalog::scan(find_examples())) {
#if defined(__linux__)
  // NVIDIA CUDA/OpenGL interop currently requires the GLX path on Wayland desktops.
  // Respect an explicit user choice; otherwise prefer X11 and retain Wayland fallback.
  if (!std::getenv("SDL_VIDEO_DRIVER"))
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11,wayland");
#endif
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_EVENTS))
    throw std::runtime_error(SDL_GetError());
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_WindowFlags window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  if (hidden)
    window_flags |= SDL_WINDOW_HIDDEN;
  window_ = SDL_CreateWindow("Cudalab — CUDA Creative Studio", 1600, 960, window_flags);
  if (!window_)
    throw std::runtime_error(SDL_GetError());
  gl_context_ = SDL_GL_CreateContext(window_);
  if (!gl_context_)
    throw std::runtime_error(SDL_GetError());
  SDL_GL_MakeCurrent(window_, gl_context_);
  SDL_GL_SetSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.IniFilename = nullptr;
  set_studio_theme();
  ImGui_ImplSDL3_InitForOpenGL(window_, gl_context_);
  ImGui_ImplOpenGL3_Init("#version 330 core");

  editor_ = std::make_unique<TextEditor>();
  editor_->SetLanguage(cuda_language());
  editor_->SetTabSize(2);
  editor_->SetInsertSpacesOnTabs(true);
  editor_->SetAutoIndentEnabled(true);
  editor_->SetShowLineNumbersEnabled(true);
  editor_->SetShowScrollbarMiniMapEnabled(true);
  editor_->SetShowMatchingBrackets(true);
  editor_->SetCompletePairedGlyphs(true);

  cuda_ = std::make_unique<CudaRuntime>();
  SDL_AudioSpec audio_spec{SDL_AUDIO_F32, 2, 48000};
  audio_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &audio_spec, nullptr, nullptr);
  if (audio_stream_)
    SDL_ResumeAudioStreamDevice(audio_stream_);
  if (!catalog_.demos().empty())
    load_demo(0);
  for (const auto& error : catalog_.errors())
    output_ += "[catalog] " + error + "\n";
}

Application::~Application() {
  cuda_.reset();
  if (audio_stream_)
    SDL_DestroyAudioStream(audio_stream_);
  editor_.reset();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  if (gl_context_)
    SDL_GL_DestroyContext(gl_context_);
  if (window_)
    SDL_DestroyWindow(window_);
  SDL_Quit();
}

int Application::run() {
  auto previous = std::chrono::steady_clock::now();
  while (running_) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
        running_ = false;
      bool shortcut_handled = false;
      if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
        const bool command = (event.key.mod & SDL_KMOD_CTRL) != 0;
        const bool alt = (event.key.mod & SDL_KMOD_ALT) != 0;
        const bool shift = (event.key.mod & SDL_KMOD_SHIFT) != 0;
        const bool ocean_controls = preview_hovered_ && !catalog_.demos().empty() &&
                                    catalog_.demos()[selected_].name == "ocean-procession";
        if (command && shift && event.key.key == SDLK_S) {
          request_new_piece(true);
          shortcut_handled = true;
        } else if (command && event.key.key == SDLK_N) {
          request_new_piece(false);
          shortcut_handled = true;
        } else if (event.key.key == SDLK_F5 || (command && event.key.key == SDLK_RETURN)) {
          compile();
          shortcut_handled = true;
        } else if (command && event.key.key == SDLK_S) {
          save();
          shortcut_handled = true;
        } else if (command && alt && event.key.key == SDLK_F) {
          format();
          shortcut_handled = true;
        } else if (event.key.key == SDLK_SPACE && !ImGui::GetIO().WantTextInput) {
          paused_ = !paused_;
          shortcut_handled = true;
        } else if ((!ImGui::GetIO().WantTextInput || ocean_controls) && event.key.key >= SDLK_0 &&
                   event.key.key <= SDLK_9) {
          beaufort_ = static_cast<int>(event.key.key - SDLK_0);
          render_requested_ = true;
          shortcut_handled = true;
        } else if ((!ImGui::GetIO().WantTextInput || ocean_controls) && event.key.key >= SDLK_KP_1 &&
                   event.key.key <= SDLK_KP_9) {
          beaufort_ = static_cast<int>(event.key.key - SDLK_KP_1) + 1;
          render_requested_ = true;
          shortcut_handled = true;
        } else if ((!ImGui::GetIO().WantTextInput || ocean_controls) && event.key.key == SDLK_KP_0) {
          beaufort_ = 0;
          render_requested_ = true;
          shortcut_handled = true;
        }
      }
      if (!shortcut_handled)
        ImGui_ImplSDL3_ProcessEvent(&event);
    }
    const auto now = std::chrono::steady_clock::now();
    delta_ = std::min(.1f, std::chrono::duration<float>(now - previous).count());
    previous = now;
    if (!paused_) {
      elapsed_ += delta_;
      if (!catalog_.demos().empty()) {
        const float duration = catalog_.demos()[selected_].timeline_seconds;
        if (duration > 0.0f && elapsed_ >= duration) {
          elapsed_ = std::fmod(elapsed_, duration);
          frame_ = static_cast<int>(elapsed_ * 60.0f);
          cuda_->reset();
        }
      }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    draw_dockspace();
    draw_catalog();
    draw_editor();
    draw_preview();
    draw_output();
    if (!paused_ && audio_enabled_ && audio_stream_ && cuda_->has_audio() &&
        SDL_GetAudioStreamQueued(audio_stream_) < 4096 * 8) {
      const int authored_frame = !catalog_.demos().empty() && catalog_.demos()[selected_].timeline_seconds > 0
                                     ? static_cast<int>(elapsed_ * 60.0f)
                                     : frame_;
      FrameParams audio_params{0,
                               0,
                               elapsed_,
                               delta_,
                               mouse_x_,
                               mouse_y_,
                               authored_frame,
                               quality_,
                               mouse_dx_,
                               mouse_dy_,
                               mouse_down_,
                               beaufort_};
      const auto& audio = cuda_->synthesize_audio(audio_params);
      SDL_PutAudioStreamData(audio_stream_, audio.data(), static_cast<int>(audio.size() * sizeof(float2)));
    }

    ImGui::Render();
    int width = 0, height = 0;
    SDL_GetWindowSizeInPixels(window_, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(.025f, .028f, .045f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window_);
    ++frame_;
  }
  if (source_ != saved_source_)
    save();
  return 0;
}

int Application::smoke_test() {
  int failures = 0;
  cuda_->resize(640, 360);
  for (std::size_t i = 0; i < catalog_.demos().size(); ++i) {
    load_demo(i);
    if (!last_compile_ok_) {
      ++failures;
      continue;
    }
    try {
      for (int f = 0; f < 3; ++f) {
        FrameParams params{640, 360, f / 60.0f, 1 / 60.0f, .5f, .5f, f, 2};
        cuda_->render(params);
      }
      if (cuda_->has_audio()) {
        FrameParams params{0, 0, 0, 1 / 60.0f, .5f, .5f, 0, 2};
        if (cuda_->synthesize_audio(params).size() != 1024)
          throw std::runtime_error("audio kernel returned wrong frame count");
      }
      SDL_Log("[smoke] %s rendered successfully", catalog_.demos()[i].name.c_str());
    } catch (const std::exception& error) {
      SDL_Log("[smoke] %s failed: %s", catalog_.demos()[i].name.c_str(), error.what());
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}

int Application::snapshot(const std::string& demo_name,
                          float time,
                          const std::filesystem::path& output_path,
                          float mouse_x,
                          float mouse_y,
                          int beaufort) {
  const auto match = std::ranges::find_if(
      catalog_.demos(), [&](const Demo& demo) { return demo.name == demo_name || demo.title == demo_name; });
  if (match == catalog_.demos().end()) {
    throw std::runtime_error("Unknown demo for snapshot: " + demo_name);
  }
  load_demo(static_cast<std::size_t>(std::distance(catalog_.demos().begin(), match)));
  if (!last_compile_ok_)
    return 1;

  constexpr int width = 960;
  constexpr int height = 540;
  constexpr float step = 1.0f / 60.0f;
  mouse_x = std::clamp(mouse_x, 0.0f, 1.0f);
  mouse_y = std::clamp(mouse_y, 0.0f, 1.0f);
  beaufort = std::clamp(beaufort, 0, 9);
  cuda_->resize(width, height);
  const float start = std::max(0.0f, time - 2.0f);
  int frame = static_cast<int>(start * 60.0f);
  for (float t = start; t <= time + step * .5f; t += step, ++frame) {
    FrameParams params{width, height, t, step, mouse_x, mouse_y, frame, 2, 0.0f, 0.0f, 0, beaufort};
    cuda_->render(params);
  }

  std::vector<unsigned char> rgba(static_cast<std::size_t>(width) * height * 4);
  glBindTexture(GL_TEXTURE_2D, cuda_->texture());
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error("Cannot write snapshot " + output_path.string());
  output << "P6\n" << width << ' ' << height << "\n255\n";
  for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
    output.write(reinterpret_cast<const char*>(rgba.data() + i * 4), 3);
  }
  SDL_Log("[snapshot] %s at %.2f seconds -> %s", demo_name.c_str(), time, output_path.string().c_str());
  return 0;
}

void Application::draw_dockspace() {
  const auto* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::Begin("CudalabDockspace", nullptr, flags);
  ImGui::PopStyleVar(3);
  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("New piece", "Ctrl+N"))
        request_new_piece(false);
      if (ImGui::MenuItem("Clone piece / Save As", "Ctrl+Shift+S"))
        request_new_piece(true);
      ImGui::Separator();
      if (ImGui::MenuItem("Save", "Ctrl+S"))
        save();
      if (ImGui::MenuItem("Capture frame"))
        capture_frame();
      if (ImGui::MenuItem("Format CUDA source", "Ctrl+Alt+F"))
        format();
      if (ImGui::MenuItem("Exit"))
        running_ = false;
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Run")) {
      if (ImGui::MenuItem("Compile and run", "Ctrl+Enter / F5"))
        compile();
      if (ImGui::MenuItem(paused_ ? "Resume" : "Pause", "Space"))
        paused_ = !paused_;
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Reset layout"))
        reset_layout();
      ImGui::EndMenu();
    }
    ImGui::Separator();
    ImGui::TextColored(ImVec4(.35f, .62f, 1, 1), "CUDA//LAB");
    ImGui::Separator();
    ImGui::TextDisabled("%s  |  sm_%d%d  |  %.1f GB",
                        cuda_->device().name.c_str(),
                        cuda_->device().compute_major,
                        cuda_->device().compute_minor,
                        cuda_->device().memory_bytes / 1073741824.0);
    ImGui::EndMenuBar();
  }
  const ImGuiID dock_id = ImGui::GetID("CudalabDock");
  ImGui::DockSpace(dock_id, {0, 0}, ImGuiDockNodeFlags_PassthruCentralNode);
  if (first_layout_) {
    first_layout_ = false;
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, viewport->WorkSize);
    ImGuiID center = dock_id;
    const ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, .17f, nullptr, &center);
    const ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, .48f, nullptr, &center);
    const ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, .25f, nullptr, &center);
    ImGui::DockBuilderDockWindow("Gallery", left);
    ImGui::DockBuilderDockWindow("Kernel", center);
    ImGui::DockBuilderDockWindow("Preview", right);
    ImGui::DockBuilderDockWindow("Output", bottom);
    ImGui::DockBuilderFinish(dock_id);
  }
  ImGui::End();
}

void Application::draw_catalog() {
  ImGui::Begin("Gallery");
  char filter[128]{};
  std::strncpy(filter, filter_.c_str(), sizeof(filter) - 1);
  ImGui::SetNextItemWidth(-1);
  if (ImGui::InputTextWithHint("##filter", "Find a demo...", filter, sizeof(filter)))
    filter_ = filter;
  ImGui::SeparatorText("BEST OF CUDA");
  const auto needle = lower(filter_);
  for (std::size_t i = 0; i < catalog_.demos().size(); ++i) {
    const auto& demo = catalog_.demos()[i];
    if (!needle.empty() &&
        lower(demo.title + demo.description + demo.category).find(needle) == std::string::npos)
      continue;
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::Selectable(demo.title.c_str(), selected_ == i, ImGuiSelectableFlags_AllowDoubleClick))
      load_demo(i);
    ImGui::TextDisabled("%s", demo.category.c_str());
    if (!demo.techniques.empty())
      ImGui::TextColored(ImVec4(.35f, .62f, 1, 1), "%s", demo.techniques.c_str());
    if (demo.use_graph || !demo.resources.empty() || demo.timeline_seconds > 0) {
      ImGui::TextColored(ImVec4(.78f, .48f, 1, 1),
                         "%s%s%s",
                         demo.use_graph ? "CUDA GRAPH" : "",
                         demo.use_graph && !demo.resources.empty() ? " · " : "",
                         !demo.resources.empty()
                             ? (std::to_string(demo.resources.size()) + " NAMED RESOURCES").c_str()
                             : "");
    }
    ImGui::PushTextWrapPos();
    ImGui::TextColored(ImVec4(.52f, .62f, .76f, 1), "%s", demo.description.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    ImGui::PopID();
  }
  ImGui::End();
}

void Application::draw_editor() {
  ImGui::Begin("Kernel");
  const bool dirty = source_ != saved_source_;
  if (ImGui::Button("New"))
    request_new_piece(false);
  ImGui::SameLine();
  if (ImGui::Button("Clone / Save As"))
    request_new_piece(true);
  ImGui::SameLine();
  if (ImGui::Button("Run  Ctrl+Enter"))
    compile();
  ImGui::SameLine();
  if (ImGui::Button("Save  Ctrl+S"))
    save();
  ImGui::SameLine();
  if (ImGui::Button("Format  Ctrl+Alt+F"))
    format();
  ImGui::SameLine();
  ImGui::TextColored(dirty ? ImVec4(1, .74f, .25f, 1) : ImVec4(.3f, .85f, .6f, 1),
                     dirty ? "modified" : "saved");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", source_path_.filename().string().c_str());
  const auto available = ImGui::GetContentRegionAvail();
  if (editor_) {
    editor_->Render("##cuda-source", available, true);
    source_ = editor_->GetText();
  }
  if (new_piece_popup_) {
    ImGui::OpenPopup("Create CUDA artwork");
    new_piece_popup_ = false;
  }
  if (ImGui::BeginPopupModal("Create CUDA artwork", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted(clone_piece_ ? "Clone the current artwork into a new package."
                                        : "Create a new live CUDA artwork.");
    ImGui::InputTextWithHint(
        "Package", "lowercase-package-name", new_piece_name_.data(), new_piece_name_.size());
    ImGui::InputTextWithHint("Title", "Artwork title", new_piece_title_.data(), new_piece_title_.size());
    if (ImGui::Button("Create", {120, 0})) {
      if (create_piece())
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", {120, 0}))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  ImGui::End();
}

void Application::draw_preview() {
  ImGui::Begin("Preview");
  if (!catalog_.demos().empty()) {
    const auto& demo = catalog_.demos()[selected_];
    ImGui::TextColored(ImVec4(.78f, .86f, 1, 1), "%s", demo.title.c_str());
  }
  ImGui::SameLine(ImGui::GetWindowWidth() - 250);
  ImGui::Text("%.2f ms  %.0f FPS", gpu_ms_, ImGui::GetIO().Framerate);
  ImGui::SameLine();
  if (ImGui::SmallButton(audio_enabled_ ? "Audio: on" : "Audio: muted")) {
    audio_enabled_ = !audio_enabled_;
    if (!audio_enabled_ && audio_stream_)
      SDL_ClearAudioStream(audio_stream_);
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Capture"))
    capture_frame();
  if (!catalog_.demos().empty() && catalog_.demos()[selected_].timeline_seconds > 0.0f) {
    const auto& demo = catalog_.demos()[selected_];
    if (ImGui::SmallButton(paused_ ? "Play" : "Pause"))
      paused_ = !paused_;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-90.0f);
    float playhead = elapsed_;
    if (ImGui::SliderFloat("##timeline", &playhead, 0.0f, demo.timeline_seconds, "%.2f s")) {
      elapsed_ = playhead;
      frame_ = static_cast<int>(elapsed_ * 60.0f);
      cuda_->reset();
      render_requested_ = true;
      if (audio_stream_)
        SDL_ClearAudioStream(audio_stream_);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%.1f BPM", demo.bpm);
  }
  if (!catalog_.demos().empty() && catalog_.demos()[selected_].name == "ocean-procession") {
    float wind_x = (mouse_x_ - .5f) * 2.0f;
    float wind_z = (mouse_y_ - .5f) * 2.0f;
    if (std::hypot(wind_x, wind_z) < .24f) {
      wind_x = 0.0f;
      wind_z = -1.0f;
    }
    const float heading = std::atan2(wind_x, -wind_z) * 57.2957795f;
    ImGui::TextDisabled("Wind: Beaufort %d  |  %+03.0f deg from horizon  |  pointer steers, 0-9 sets force",
                        beaufort_,
                        heading);
  }
  const ImVec2 area = ImGui::GetContentRegionAvail();
  const int width = std::max(64, static_cast<int>(area.x));
  const int height = std::max(64, static_cast<int>(area.y));
  preview_width_ = width;
  preview_height_ = height;
  cuda_->resize(width, height);
  const int authored_frame = !catalog_.demos().empty() && catalog_.demos()[selected_].timeline_seconds > 0
                                 ? static_cast<int>(elapsed_ * 60.0f)
                                 : frame_;
  FrameParams params{width,
                     height,
                     elapsed_,
                     delta_,
                     mouse_x_,
                     mouse_y_,
                     authored_frame,
                     quality_,
                     mouse_dx_,
                     mouse_dy_,
                     mouse_down_,
                     beaufort_};
  if ((!paused_ || render_requested_) && cuda_->ready()) {
    gpu_ms_ = cuda_->render(params);
    render_requested_ = false;
  }
  const ImVec2 top_left = ImGui::GetCursorScreenPos();
  const bool firefly_view =
      !catalog_.demos().empty() && catalog_.demos()[selected_].name == "firefly-constellation";
  const ImVec2 preview_uv0 = firefly_view ? ImVec2(.5f, .5f) : ImVec2(0, 0);
  ImGui::Image(static_cast<ImTextureID>(cuda_->texture()), area, preview_uv0, {1, 1});
  preview_hovered_ = ImGui::IsItemHovered();
  if (!catalog_.demos().empty()) {
    const auto& demo = catalog_.demos()[selected_];
    auto* draw = ImGui::GetWindowDrawList();
    const ImVec2 overlay_end(top_left.x + area.x, top_left.y + 43.0f);
    draw->PushClipRect(top_left, ImVec2(top_left.x + area.x, top_left.y + area.y), true);
    draw->AddRectFilled(top_left, overlay_end, IM_COL32(5, 9, 18, 205));
    draw->AddLine(ImVec2(top_left.x, overlay_end.y), overlay_end, IM_COL32(75, 130, 220, 180));
    draw->AddText(
        ImVec2(top_left.x + 12.0f, top_left.y + 6.0f), IM_COL32(205, 222, 255, 255), demo.title.c_str());
    const std::string instructions =
        demo.controls.empty() ? "Watch, listen, and edit the CUDA source." : demo.controls;
    draw->AddText(
        ImVec2(top_left.x + 12.0f, top_left.y + 23.0f), IM_COL32(145, 174, 220, 255), instructions.c_str());
    draw->PopClipRect();
  }
  if (preview_hovered_) {
    const auto mouse = ImGui::GetMousePos();
    const float next_x = (mouse.x - top_left.x) / std::max(1.0f, area.x);
    const float next_y = (mouse.y - top_left.y) / std::max(1.0f, area.y);
    mouse_dx_ = next_x - mouse_x_;
    mouse_dy_ = next_y - mouse_y_;
    mouse_x_ = next_x;
    mouse_y_ = next_y;
    mouse_down_ = ImGui::IsMouseDown(ImGuiMouseButton_Left) ? 1 : 0;
  } else {
    mouse_dx_ = mouse_dy_ = 0.0f;
    mouse_down_ = 0;
  }
  ImGui::End();
}

void Application::draw_output() {
  ImGui::Begin("Output");
  if (ImGui::Button("Clear"))
    output_.clear();
  ImGui::SameLine();
  const auto& d = cuda_->device();
  ImGui::TextDisabled("driver %d.%d  runtime %d.%d  %d SMs  |  3 streams  |  %zu resources%s",
                      d.driver_version / 1000,
                      (d.driver_version % 1000) / 10,
                      d.runtime_version / 1000,
                      (d.runtime_version % 1000) / 10,
                      d.multiprocessors,
                      cuda_->resource_count(),
                      cuda_->graph_enabled() ? "  |  graph replay" : "");
  ImGui::Separator();
  ImGui::BeginChild("log");
  ImGui::PushTextWrapPos();
  ImGui::TextUnformatted(output_.c_str());
  ImGui::PopTextWrapPos();
  ImGui::EndChild();
  ImGui::End();
}

void Application::load_demo(std::size_t index) {
  if (index >= catalog_.demos().size())
    return;
  if (source_ != saved_source_)
    save();
  selected_ = index;
  if (audio_stream_)
    SDL_ClearAudioStream(audio_stream_);
  source_path_ = catalog_.demos()[index].directory / catalog_.demos()[index].entry;
  cuda_->configure(catalog_.demos()[index].state_bytes,
                   catalog_.demos()[index].work_items,
                   catalog_.demos()[index].resources,
                   catalog_.demos()[index].use_graph);
  source_ = saved_source_ = read_file(source_path_);
  editor_->SetText(source_);
  editor_->ClearMarkers();
  elapsed_ = 0;
  frame_ = 0;
  render_requested_ = true;
  compile();
}

void Application::compile() {
  if (source_path_.empty())
    return;
  source_ = editor_->GetText();
  const auto result = cuda_->compile(source_, source_path_);
  update_diagnostic_markers(result.log);
  std::ostringstream message;
  message << (result.ok ? "[ok] " : "[error] ") << source_path_.filename().string() << " — " << std::fixed
          << std::setprecision(1) << result.milliseconds << " ms\n"
          << result.log << "\n\n";
  SDL_Log("%s", message.str().c_str());
  output_ = message.str() + output_;
  last_compile_ok_ = result.ok;
  std::ostringstream title;
  title << "Cudalab — "
        << (catalog_.demos().empty() ? source_path_.filename().string() : catalog_.demos()[selected_].title)
        << " — " << (result.ok ? "live" : "compile error") << " — " << std::fixed << std::setprecision(1)
        << result.milliseconds << " ms";
  SDL_SetWindowTitle(window_, title.str().c_str());
  if (result.ok) {
    elapsed_ = 0;
    frame_ = 0;
    render_requested_ = true;
  }
}

void Application::save() {
  if (source_path_.empty())
    return;
  source_ = editor_->GetText();
  std::ofstream output(source_path_, std::ios::binary | std::ios::trunc);
  if (!output) {
    output_ = "[error] Cannot save " + source_path_.string() + "\n" + output_;
    return;
  }
  output << source_;
  saved_source_ = source_;
  output_ = "[saved] " + source_path_.string() + "\n" + output_;
}

void Application::request_new_piece(bool clone) {
  clone_piece_ = clone;
  new_piece_name_.fill('\0');
  new_piece_title_.fill('\0');
  std::string suggested_name =
      clone && !catalog_.demos().empty() ? catalog_.demos()[selected_].name + "-study" : "untitled-artwork";
  std::string suggested_title =
      clone && !catalog_.demos().empty() ? catalog_.demos()[selected_].title + " Study" : "Untitled Artwork";
  std::copy_n(suggested_name.c_str(),
              std::min(suggested_name.size(), new_piece_name_.size() - 1),
              new_piece_name_.data());
  std::copy_n(suggested_title.c_str(),
              std::min(suggested_title.size(), new_piece_title_.size() - 1),
              new_piece_title_.data());
  new_piece_popup_ = true;
}

bool Application::create_piece() {
  const std::string name = new_piece_name_.data();
  const std::string title = new_piece_title_.data();
  const std::regex valid_name("[a-z0-9]+(?:-[a-z0-9]+)*");
  if (!std::regex_match(name, valid_name) || title.empty()) {
    output_ =
        "[error] Package names use lowercase letters, digits, and single hyphens; title is required.\n" +
        output_;
    return false;
  }
  const auto directory = find_examples() / name;
  if (std::filesystem::exists(directory)) {
    output_ = "[error] Package already exists: " + directory.string() + "\n" + output_;
    return false;
  }

  auto escape_json = [](const std::string& value) {
    std::string escaped;
    for (const char c : value) {
      if (c == '\\' || c == '"')
        escaped.push_back('\\');
      escaped.push_back(c);
    }
    return escaped;
  };
  std::error_code error;
  if (!std::filesystem::create_directories(directory, error)) {
    output_ = "[error] Cannot create " + directory.string() + ": " + error.message() + "\n" + output_;
    return false;
  }

  const auto entry = name + ".cu";
  std::ofstream manifest(directory / "cudalab.json", std::ios::binary);
  std::ofstream kernel(directory / entry, std::ios::binary);
  if (!manifest || !kernel) {
    output_ = "[error] Cannot write the new package in " + directory.string() + "\n" + output_;
    return false;
  }
  manifest << "{\n"
           << "  \"name\": \"" << name << "\",\n"
           << "  \"title\": \"" << escape_json(title) << "\",\n"
           << "  \"description\": \"A new CUDA artwork, ready to become something impossible.\",\n"
           << "  \"category\": \"SKETCHBOOK\",\n"
           << "  \"entry\": \"" << entry << "\",\n"
           << "  \"controls\": \"Move the pointer to shape the artwork\",\n"
           << "  \"techniques\": \"live CUDA C++ · procedural color\"\n"
           << "}\n";
  if (clone_piece_) {
    kernel << editor_->GetText();
  } else {
    kernel << R"(#include <cudalab.cuh>

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  float2 uv = make_float2((2.0f * x - params.width) / params.height,
                          (params.height - 2.0f * y) / params.height);
  float distance = hypotf(uv.x - (params.mouse_x - .5f) * 2.0f,
                          uv.y + (params.mouse_y - .5f) * 2.0f);
  float pulse = .5f + .5f * sinf(distance * 24.0f - params.time * 3.0f);
  float3 color = make_float3(.04f + pulse * .72f, .02f + pulse * pulse * .22f,
                             .10f + (1.0f - pulse) * .85f);
  color = cudalab_tonemap(color);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(color.x), .4545f),
                                             255 * powf(cudalab_saturate(color.y), .4545f),
                                             255 * powf(cudalab_saturate(color.z), .4545f), 255);
}
)";
  }
  manifest.close();
  kernel.close();

  catalog_ = DemoCatalog::scan(find_examples());
  const auto created = std::ranges::find(catalog_.demos(), name, &Demo::name);
  if (created == catalog_.demos().end()) {
    output_ = "[error] Created package did not enter the catalog.\n" + output_;
    return false;
  }
  load_demo(static_cast<std::size_t>(std::distance(catalog_.demos().begin(), created)));
  output_ = "[created] " + directory.string() + "\n" + output_;
  return true;
}

void Application::capture_frame() {
  if (preview_width_ <= 0 || preview_height_ <= 0 || !cuda_->texture()) {
    output_ = "[error] Preview is not ready to capture.\n" + output_;
    return;
  }
  const auto capture_root = std::filesystem::current_path() / "captures";
  std::error_code error;
  std::filesystem::create_directories(capture_root, error);
  if (error) {
    output_ = "[error] Cannot create capture directory: " + error.message() + "\n" + output_;
    return;
  }
  const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  const std::string name = catalog_.demos().empty() ? "artwork" : catalog_.demos()[selected_].name;
  const auto path = capture_root / (name + "-" + std::to_string(stamp) + ".ppm");
  std::vector<unsigned char> rgba(static_cast<std::size_t>(preview_width_) * preview_height_ * 4);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, cuda_->texture());
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    output_ = "[error] Cannot write " + path.string() + "\n" + output_;
    return;
  }
  output << "P6\n" << preview_width_ << ' ' << preview_height_ << "\n255\n";
  for (std::size_t i = 0; i < static_cast<std::size_t>(preview_width_) * preview_height_; ++i)
    output.write(reinterpret_cast<const char*>(rgba.data() + i * 4), 3);
  output_ = "[captured] " + path.string() + "\n" + output_;
}

void Application::format() {
  if (source_path_.empty())
    return;
  source_ = editor_->GetText();
  {
    std::ofstream file(source_path_, std::ios::binary | std::ios::trunc);
    if (!file) {
      output_ = "[error] Cannot save before formatting " + source_path_.string() + "\n" + output_;
      return;
    }
    file << source_;
  }
  saved_source_ = source_;
  const int result = run_clang_format(source_path_);
  if (result != 0) {
    output_ = "[error] clang-format failed (exit " + std::to_string(result) +
              "). Install clang-format and ensure it is on PATH.\n" + output_;
    return;
  }
  source_ = saved_source_ = read_file(source_path_);
  editor_->SetText(source_);
  editor_->ClearMarkers();
  output_ = "[formatted] " + source_path_.string() + "\n" + output_;
}

void Application::update_diagnostic_markers(const std::string& log) {
  editor_->ClearMarkers();
  static const std::regex location(R"((?:\(|:)([0-9]+)(?::[0-9]+)?(?:\)|:))");
  std::istringstream lines(log);
  std::string line;
  int first_error = -1;
  while (std::getline(lines, line)) {
    std::smatch match;
    if (!std::regex_search(line, match, location))
      continue;
    const int line_number = std::stoi(match[1].str()) - 1;
    const bool warning = line.find("warning") != std::string::npos;
    const ImU32 gutter = warning ? IM_COL32(236, 177, 65, 210) : IM_COL32(245, 78, 107, 220);
    const ImU32 background = warning ? IM_COL32(130, 92, 20, 50) : IM_COL32(145, 30, 55, 60);
    editor_->AddMarker(line_number, gutter, background, line, line);
    if (!warning && first_error < 0)
      first_error = line_number;
  }
  if (first_error >= 0) {
    editor_->SetCursor(first_error, 0);
    editor_->ScrollToLine(first_error, TextEditor::Scroll::alignMiddle);
  }
}

void Application::reset_layout() {
  first_layout_ = true;
}

std::filesystem::path Application::find_examples() {
  const std::vector<std::filesystem::path> candidates = {
      std::filesystem::path(CUDALAB_SOURCE_ROOT) / "examples", std::filesystem::current_path() / "examples"};
  for (const auto& path : candidates)
    if (std::filesystem::is_directory(path))
      return path;
  return candidates.front();
}

} // namespace cudalab
