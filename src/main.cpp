#include "cudalab/application.hpp"

#include <exception>
#include <iostream>

int main() {
  try {
    cudalab::Application app;
    return app.run();
  } catch (const std::exception& error) {
    std::cerr << "cudalab: " << error.what() << '\n';
    return 1;
  }
}
