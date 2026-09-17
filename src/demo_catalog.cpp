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
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

std::string field(const std::string& json, const char* key, bool required = true) {
  const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    if (required)
      throw std::runtime_error(std::string("missing string field '") + key + "'");
    return {};
  }
  std::string value = match[1].str();
  value = std::regex_replace(value, std::regex("\\\\n"), "\n");
  value = std::regex_replace(value, std::regex("\\\\\""), "\"");
  value = std::regex_replace(value, std::regex("\\\\\\\\"), "\\");
  return value;
}

std::size_t integer_field(const std::string& json, const char* key, std::size_t fallback = 0) {
  const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern))
    return fallback;
  return static_cast<std::size_t>(std::stoull(match[1].str()));
}

float number_field(const std::string& json, const char* key, float fallback = 0.0f) {
  const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern))
    return fallback;
  return std::stof(match[1].str());
}

bool boolean_field(const std::string& json, const char* key, bool fallback = false) {
  const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern))
    return fallback;
  return match[1].str() == "true";
}

std::vector<std::string> object_array(const std::string& json, const char* key) {
  const auto key_position = json.find(std::string("\"") + key + "\"");
  if (key_position == std::string::npos)
    return {};
  const auto start = json.find('[', key_position);
  if (start == std::string::npos)
    throw std::runtime_error(std::string("malformed array '") + key + "'");
  std::vector<std::string> objects;
  bool quoted = false;
  bool escaped = false;
  int depth = 0;
  std::size_t object_start = std::string::npos;
  for (std::size_t i = start + 1; i < json.size(); ++i) {
    const char c = json[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '"') {
      quoted = !quoted;
      continue;
    }
    if (quoted)
      continue;
    if (c == '{') {
      if (depth++ == 0)
        object_start = i;
    } else if (c == '}') {
      if (depth <= 0)
        throw std::runtime_error(std::string("malformed object array '") + key + "'");
      if (--depth == 0)
        objects.push_back(json.substr(object_start, i - object_start + 1));
    } else if (c == ']' && depth == 0) {
      return objects;
    }
  }
  throw std::runtime_error(std::string("unterminated array '") + key + "'");
}

} // namespace

DemoCatalog DemoCatalog::scan(const std::filesystem::path& root) {
  DemoCatalog result;
  if (!std::filesystem::is_directory(root)) {
    result.errors_.push_back("Examples directory not found: " + root.string());
    return result;
  }
  for (const auto& item : std::filesystem::directory_iterator(root)) {
    if (!item.is_directory())
      continue;
    const auto manifest = item.path() / "cudalab.json";
    if (!std::filesystem::is_regular_file(manifest))
      continue;
    try {
      const auto json = read_file(manifest);
      Demo demo;
      demo.name = field(json, "name");
      demo.title = field(json, "title");
      demo.description = field(json, "description");
      demo.category = field(json, "category");
      demo.entry = field(json, "entry");
      demo.controls = field(json, "controls", false);
      demo.techniques = field(json, "techniques", false);
      demo.state_bytes = integer_field(json, "state_bytes");
      demo.work_items = static_cast<int>(integer_field(json, "work_items"));
      demo.use_graph = boolean_field(json, "graph");
      demo.timeline_seconds = number_field(json, "timeline_seconds");
      demo.bpm = number_field(json, "bpm", 120.0f);
      for (const auto& resource_json : object_array(json, "resources")) {
        ResourceSpec resource;
        resource.name = field(resource_json, "name");
        const auto kind = field(resource_json, "kind");
        if (kind == "buffer") {
          resource.kind = ResourceKind::buffer;
          resource.bytes = integer_field(resource_json, "bytes");
          if (resource.bytes == 0)
            throw std::runtime_error("buffer resource '" + resource.name + "' has no bytes");
        } else if (kind == "surface2d") {
          resource.kind = ResourceKind::surface2d;
          resource.width = static_cast<int>(integer_field(resource_json, "width"));
          resource.height = static_cast<int>(integer_field(resource_json, "height"));
          if (resource.width <= 0 || resource.height <= 0) {
            throw std::runtime_error("surface2d resource '" + resource.name + "' has invalid dimensions");
          }
        } else {
          throw std::runtime_error("resource '" + resource.name + "' has unknown kind '" + kind + "'");
        }
        if (demo.resources.size() >= 8)
          throw std::runtime_error("at most 8 resources are supported");
        demo.resources.push_back(std::move(resource));
      }
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

} // namespace cudalab
