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
public:
    tensor() noexcept = default;
    
    tensor(mag_tensor_t *t) noexcept : m_t{t} {}
    
    tensor(const tensor &rhs) noexcept : m_t{rhs.m_t} {
        if (m_t) mag_tensor_incref(m_t);
    }
    
    tensor(tensor &&rhs) noexcept : m_t{rhs.m_t} {
        rhs.m_t = nullptr;
    }
    
    tensor &operator=(const tensor &rhs) noexcept {
        if (this != &rhs) {
            if (rhs.m_t) mag_tensor_incref(rhs.m_t);
            if (m_t) mag_tensor_decref(m_t);
            m_t = rhs.m_t;
        }
        return *this;
    }

    tensor &operator=(tensor &&rhs) noexcept {
        if (this != &rhs) {
            if (m_t) mag_tensor_decref(m_t);
            m_t = rhs.m_t;
            rhs.m_t = nullptr;
        }
        return *this;
    }

    ~tensor() {
        if (m_t) mag_tensor_decref(m_t);
    }

    mag_tensor_t *get() const noexcept { return m_t; }
    mag_tensor_t *operator*() const noexcept { return m_t; }
    explicit operator bool() const noexcept { return m_t != nullptr; }

private:
    mag_tensor_t *m_t = nullptr;
};

int add(int a, int b) {
    return a + b;
}

NB_MODULE(magnetron, m) {
    nb::class_<tensor>(m, "Tensor")
    .def("rank", [](const tensor &self) {
       return mag_tensor_get_rank(*self);
    });
}