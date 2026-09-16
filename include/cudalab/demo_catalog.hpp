#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cudalab {

struct Demo {
  std::string name;
  std::string title;
  std::string description;
  std::string category;
  std::string entry;
  std::string controls;
  std::filesystem::path directory;
};

class DemoCatalog {
 public:
  static DemoCatalog scan(const std::filesystem::path& root);
  [[nodiscard]] const std::vector<Demo>& demos() const { return demos_; }
  [[nodiscard]] const std::vector<std::string>& errors() const { return errors_; }

 private:
  std::vector<Demo> demos_;
  std::vector<std::string> errors_;
};

}  // namespace cudalab
