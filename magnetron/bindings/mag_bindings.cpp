/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include <nanobind/nanobind.h>
#include <magnetron/magnetron.h>

namespace nb = nanobind;
using namespace nb::literals;

class tensor final {
private:
};

int add(int a, int b) {
    return a + b;
}

NB_MODULE(my_ext, m) {
    m.def("add", &add);
}