#include "cudalab/application.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cudalab {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot read " + path.string());
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

std::string lower(std::string value) {
  std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
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

}  // namespace

Application::Application() : catalog_(DemoCatalog::scan(find_examples())) {
#if defined(__linux__)
  // NVIDIA CUDA/OpenGL interop currently requires the GLX path on Wayland desktops.
  // Respect an explicit user choice; otherwise prefer X11 and retain Wayland fallback.
  if (!std::getenv("SDL_VIDEO_DRIVER")) SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11,wayland");
#endif
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) throw std::runtime_error(SDL_GetError());
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  window_ = SDL_CreateWindow("Cudalab — CUDA Creative Studio", 1600, 960,
                             SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  if (!window_) throw std::runtime_error(SDL_GetError());
  gl_context_ = SDL_GL_CreateContext(window_);
  if (!gl_context_) throw std::runtime_error(SDL_GetError());
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

  cuda_ = std::make_unique<CudaRuntime>();
  if (!catalog_.demos().empty()) load_demo(0);
  for (const auto& error : catalog_.errors()) output_ += "[catalog] " + error + "\n";
}

Application::~Application() {
  cuda_.reset();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  if (gl_context_) SDL_GL_DestroyContext(gl_context_);
  if (window_) SDL_DestroyWindow(window_);
  SDL_Quit();
}

int Application::run() {
  auto previous = std::chrono::steady_clock::now();
  while (running_) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL3_ProcessEvent(&event);
      if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) running_ = false;
      if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
        const bool command = (event.key.mod & SDL_KMOD_CTRL) != 0;
        if (event.key.key == SDLK_F5 || (command && event.key.key == SDLK_RETURN)) compile();
        if (command && event.key.key == SDLK_S) save();
        if (event.key.key == SDLK_SPACE && !ImGui::GetIO().WantTextInput) paused_ = !paused_;
      }
    }
    const auto now = std::chrono::steady_clock::now();
    delta_ = std::min(.1f, std::chrono::duration<float>(now - previous).count());
    previous = now;
    if (!paused_) elapsed_ += delta_;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    draw_dockspace();
    draw_catalog();
    draw_editor();
    draw_preview();
    draw_output();

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
  if (source_ != saved_source_) save();
  return 0;
}

void Application::draw_dockspace() {
  const auto* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::Begin("CudalabDockspace", nullptr, flags);
  ImGui::PopStyleVar(3);
  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Save", "Ctrl+S")) save();
      if (ImGui::MenuItem("Exit")) running_ = false;
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Run")) {
      if (ImGui::MenuItem("Compile and run", "Ctrl+Enter / F5")) compile();
      if (ImGui::MenuItem(paused_ ? "Resume" : "Pause", "Space")) paused_ = !paused_;
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Reset layout")) reset_layout();
      ImGui::EndMenu();
    }
    ImGui::Separator();
    ImGui::TextColored(ImVec4(.35f, .62f, 1, 1), "CUDA//LAB");
    ImGui::Separator();
    ImGui::TextDisabled("%s  |  sm_%d%d  |  %.1f GB",
      cuda_->device().name.c_str(), cuda_->device().compute_major, cuda_->device().compute_minor,
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
  if (ImGui::InputTextWithHint("##filter", "Find a demo...", filter, sizeof(filter))) filter_ = filter;
  ImGui::SeparatorText("BEST OF CUDA");
  const auto needle = lower(filter_);
  for (std::size_t i = 0; i < catalog_.demos().size(); ++i) {
    const auto& demo = catalog_.demos()[i];
    if (!needle.empty() && lower(demo.title + demo.description + demo.category).find(needle) == std::string::npos) continue;
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::Selectable(demo.title.c_str(), selected_ == i, ImGuiSelectableFlags_AllowDoubleClick)) load_demo(i);
    ImGui::TextDisabled("%s", demo.category.c_str());
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
  if (ImGui::Button("Run  Ctrl+Enter")) compile();
  ImGui::SameLine();
  if (ImGui::Button("Save  Ctrl+S")) save();
  ImGui::SameLine();
  ImGui::TextColored(dirty ? ImVec4(1, .74f, .25f, 1) : ImVec4(.3f, .85f, .6f, 1),
                     dirty ? "modified" : "saved");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", source_path_.filename().string().c_str());
  const auto available = ImGui::GetContentRegionAvail();
  if (!editor_buffer_.empty() && ImGui::InputTextMultiline("##source", editor_buffer_.data(), editor_buffer_.size(),
      available, ImGuiInputTextFlags_AllowTabInput)) {
    source_ = editor_buffer_.data();
  }
  ImGui::End();
}

void Application::draw_preview() {
  ImGui::Begin("Preview");
  if (!catalog_.demos().empty()) {
    const auto& demo = catalog_.demos()[selected_];
    ImGui::TextColored(ImVec4(.78f, .86f, 1, 1), "%s", demo.title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("  %s", demo.controls.c_str());
  }
  ImGui::SameLine(ImGui::GetWindowWidth() - 250);
  ImGui::Text("%.2f ms  %.0f FPS", gpu_ms_, ImGui::GetIO().Framerate);
  const ImVec2 area = ImGui::GetContentRegionAvail();
  const int width = std::max(64, static_cast<int>(area.x));
  const int height = std::max(64, static_cast<int>(area.y));
  cuda_->resize(width, height);
  FrameParams params{width, height, elapsed_, delta_, mouse_x_, mouse_y_, frame_, quality_};
  if (!paused_ && cuda_->ready()) gpu_ms_ = cuda_->render(params);
  const ImVec2 top_left = ImGui::GetCursorScreenPos();
  ImGui::Image(static_cast<ImTextureID>(cuda_->texture()), area, {0, 1}, {1, 0});
  if (ImGui::IsItemHovered()) {
    const auto mouse = ImGui::GetMousePos();
    mouse_x_ = (mouse.x - top_left.x) / std::max(1.0f, area.x);
    mouse_y_ = (mouse.y - top_left.y) / std::max(1.0f, area.y);
  }
  ImGui::End();
}

void Application::draw_output() {
  ImGui::Begin("Output");
  if (ImGui::Button("Clear")) output_.clear();
  ImGui::SameLine();
  const auto& d = cuda_->device();
  ImGui::TextDisabled("driver %d.%d  runtime %d.%d  %d SMs",
    d.driver_version / 1000, (d.driver_version % 1000) / 10,
    d.runtime_version / 1000, (d.runtime_version % 1000) / 10, d.multiprocessors);
  ImGui::Separator();
  ImGui::BeginChild("log");
  ImGui::PushTextWrapPos();
  ImGui::TextUnformatted(output_.c_str());
  ImGui::PopTextWrapPos();
  ImGui::EndChild();
  ImGui::End();
}

void Application::load_demo(std::size_t index) {
  if (index >= catalog_.demos().size()) return;
  if (source_ != saved_source_) save();
  selected_ = index;
  source_path_ = catalog_.demos()[index].directory / catalog_.demos()[index].entry;
  source_ = saved_source_ = read_file(source_path_);
  editor_buffer_.assign(std::max<std::size_t>(source_.size() * 3 + 65536, 1024 * 1024), '\0');
  std::copy(source_.begin(), source_.end(), editor_buffer_.begin());
  elapsed_ = 0;
  frame_ = 0;
  compile();
}

void Application::compile() {
  if (source_path_.empty()) return;
  source_ = editor_buffer_.data();
  const auto result = cuda_->compile(source_, source_path_);
  std::ostringstream message;
  message << (result.ok ? "[ok] " : "[error] ") << source_path_.filename().string()
          << " — " << std::fixed << std::setprecision(1) << result.milliseconds << " ms\n"
          << result.log << "\n\n";
  SDL_Log("%s", message.str().c_str());
  output_ = message.str() + output_;
  std::ostringstream title;
  title << "Cudalab — " << (catalog_.demos().empty() ? source_path_.filename().string() : catalog_.demos()[selected_].title)
        << " — " << (result.ok ? "live" : "compile error") << " — "
        << std::fixed << std::setprecision(1) << result.milliseconds << " ms";
  SDL_SetWindowTitle(window_, title.str().c_str());
  if (result.ok) { elapsed_ = 0; frame_ = 0; }
}

void Application::save() {
  if (source_path_.empty()) return;
  source_ = editor_buffer_.data();
  std::ofstream output(source_path_, std::ios::binary | std::ios::trunc);
  if (!output) { output_ = "[error] Cannot save " + source_path_.string() + "\n" + output_; return; }
  output << source_;
  saved_source_ = source_;
  output_ = "[saved] " + source_path_.string() + "\n" + output_;
}

void Application::reset_layout() { first_layout_ = true; }

std::filesystem::path Application::find_examples() {
  const std::vector<std::filesystem::path> candidates = {
    std::filesystem::path(CUDALAB_SOURCE_ROOT) / "examples",
    std::filesystem::current_path() / "examples"
  };
  for (const auto& path : candidates) if (std::filesystem::is_directory(path)) return path;
  return candidates.front();
}

}  // namespace cudalab
