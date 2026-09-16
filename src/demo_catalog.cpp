#include "cudalab/demo_catalog.hpp"

#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace cudalab {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path.string());
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

std::string field(const std::string& json, const char* key, bool required = true) {
  const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    if (required) throw std::runtime_error(std::string("missing string field '") + key + "'");
    return {};
  }
  std::string value = match[1].str();
  value = std::regex_replace(value, std::regex("\\\\n"), "\n");
  value = std::regex_replace(value, std::regex("\\\\\""), "\"");
  value = std::regex_replace(value, std::regex("\\\\\\\\"), "\\");
  return value;
}

}  // namespace

DemoCatalog DemoCatalog::scan(const std::filesystem::path& root) {
  DemoCatalog result;
  if (!std::filesystem::is_directory(root)) {
    result.errors_.push_back("Examples directory not found: " + root.string());
    return result;
  }
  for (const auto& item : std::filesystem::directory_iterator(root)) {
    if (!item.is_directory()) continue;
    const auto manifest = item.path() / "cudalab.json";
    if (!std::filesystem::is_regular_file(manifest)) continue;
    try {
      const auto json = read_file(manifest);
      Demo demo;
      demo.name = field(json, "name");
      demo.title = field(json, "title");
      demo.description = field(json, "description");
      demo.category = field(json, "category");
      demo.entry = field(json, "entry");
      demo.controls = field(json, "controls", false);
      demo.directory = item.path();
      if (!std::filesystem::is_regular_file(demo.directory / demo.entry)) {
        throw std::runtime_error("entry does not exist: " + demo.entry);
      }
      result.demos_.push_back(std::move(demo));
    } catch (const std::exception& error) {
      result.errors_.push_back(manifest.string() + ": " + error.what());
    }
  }
  std::ranges::sort(result.demos_, {}, &Demo::name);
  return result;
}

}  // namespace cudalab
