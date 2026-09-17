#include "cudalab/demo_catalog.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
  const auto catalog = cudalab::DemoCatalog::scan(std::filesystem::path(CUDALAB_SOURCE_ROOT) / "examples");
  for (const auto& error : catalog.errors()) std::cerr << error << '\n';
  assert(catalog.errors().empty());
  assert(catalog.demos().size() >= 13);
  assert(catalog.demos().front().name == "00-spectrum");
  for (const auto& demo : catalog.demos()) {
    assert(!demo.title.empty());
    assert(std::filesystem::exists(demo.directory / demo.entry));
  }
}
