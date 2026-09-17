#include "cudalab/application.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  try {
    const bool smoke = argc > 1 && std::string_view(argv[1]) == "--smoke-test";
    cudalab::Application app(smoke);
    if (smoke) return app.smoke_test();
    return app.run();
  } catch (const std::exception& error) {
    std::cerr << "cudalab: " << error.what() << '\n';
    return 1;
  }
}
