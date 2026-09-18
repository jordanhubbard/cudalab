#include "cudalab/demo_catalog.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <ranges>

int main() {
  const auto catalog = cudalab::DemoCatalog::scan(std::filesystem::path(CUDALAB_SOURCE_ROOT) / "examples");
  for (const auto& error : catalog.errors())
    std::cerr << error << '\n';
  assert(catalog.errors().empty());
  assert(catalog.demos().size() >= 16);
  assert(catalog.demos().front().name == "00-spectrum");
  for (const auto& demo : catalog.demos()) {
    assert(!demo.title.empty());
    assert(!demo.controls.empty());
    assert(std::filesystem::exists(demo.directory / demo.entry));
    assert(demo.resources.size() <= 8);
    for (const auto& resource : demo.resources)
      assert(!resource.name.empty());
  }
  const auto choreograph = std::ranges::find(catalog.demos(), "choreograph", &cudalab::Demo::name);
  assert(choreograph != catalog.demos().end());
  assert(choreograph->use_graph);
  assert(choreograph->timeline_seconds == 32.0f);
  assert(choreograph->bpm == 112.0f);
  assert(choreograph->resources.size() == 3);
  assert(choreograph->resources[0].kind == cudalab::ResourceKind::buffer);
  assert(choreograph->resources[1].kind == cudalab::ResourceKind::surface2d);
  const auto ocean = std::ranges::find(catalog.demos(), "ocean-procession", &cudalab::Demo::name);
  assert(ocean != catalog.demos().end());
  assert(ocean->work_items == 12);
  assert(ocean->state_bytes >= 12 * sizeof(float) * 8);
}
