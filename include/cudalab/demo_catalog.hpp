#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cudalab {

enum class ResourceKind { buffer, surface2d };

struct ResourceSpec {
  std::string name;
  ResourceKind kind = ResourceKind::buffer;
  std::size_t bytes = 0;
  int width = 0;
  int height = 0;
};

struct Demo {
  std::string name;
  std::string title;
  std::string description;
  std::string category;
  std::string entry;
  std::string controls;
  std::string techniques;
  std::size_t state_bytes = 0;
  int work_items = 0;
  bool use_graph = false;
  float timeline_seconds = 0.0f;
  float bpm = 120.0f;
  std::vector<ResourceSpec> resources;
  std::filesystem::path directory;
};

class DemoCatalog {
public:
  static DemoCatalog scan(const std::filesystem::path& root);
  [[nodiscard]] const std::vector<Demo>& demos() const {
    return demos_;
  }
  [[nodiscard]] const std::vector<std::string>& errors() const {
    return errors_;
  }

private:
  std::vector<Demo> demos_;
  std::vector<std::string> errors_;
};

} // namespace cudalab
