#include "cudalab/application.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
  try {
    const bool smoke = argc > 1 && std::string_view(argv[1]) == "--smoke-test";
    const bool snapshot = argc > 1 && std::string_view(argv[1]) == "--snapshot";
    if (snapshot && argc != 5) {
      std::cerr << "usage: cudalab --snapshot DEMO TIME OUTPUT.ppm\n";
      return 2;
    }
    cudalab::Application app(smoke || snapshot);
    if (smoke)
      return app.smoke_test();
    if (snapshot)
      return app.snapshot(argv[2], std::stof(argv[3]), argv[4]);
    return app.run();
  } catch (const std::exception& error) {
    std::cerr << "cudalab: " << error.what() << '\n';
    return 1;
  }
}
