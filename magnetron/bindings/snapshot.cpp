/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "prelude.hpp"

namespace mag::bindings {
    [[nodiscard]] static bool ends_with(std::string_view s, std::string_view suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    class snapshot_wrapper final {
    public:
        snapshot_wrapper(const std::string &fname, bool write) : write_mode(write), filename(fname) {
            mag_context_t *ctx = get_ctx();
            p = write ? mag_snapshot_new(ctx) : mag_snapshot_deserialize(ctx, fname.c_str());
            if (!p) throw std::runtime_error("Failed to create or load snapshot");
        }
        ~snapshot_wrapper() {
            if (p) mag_snapshot_free(p);
        }
        snapshot_wrapper(const snapshot_wrapper&) = delete;
        snapshot_wrapper& operator=(const snapshot_wrapper&) = delete;
        snapshot_wrapper(snapshot_wrapper&& o) noexcept {
            p = o.p;
            write_mode = o.write_mode;
            filename = std::move(o.filename);
            o.p = nullptr;
        }
        mag_snapshot_t *operator *() const noexcept { return p; }
        operator bool () const noexcept { return p != nullptr; }
        [[nodiscard]] bool is_write_mode() const noexcept { return write_mode; }

        void serialize_if_needed() {
            if (write_mode && p) {
                if (!mag_snapshot_serialize(p, filename.c_str()))
                    throw std::runtime_error {"Failed to serialize snapshot"};
            }
        }

        void close() {
            if (p) {
                if (write_mode) serialize_if_needed();
                mag_snapshot_free(p);
                p = nullptr;
            }
        }

    private:
        mag_snapshot_t *p = nullptr;
        bool write_mode = false;
        std::string filename = {};
    };

    void init_bindings_snapshot(nb::module_ &m) {
        nb::class_<snapshot_wrapper>(m, "Snapshot")
        .def_static("write", [](const std::string &filename) {
            if (!ends_with(filename, ".mag"))
                throw nb::value_error("Filename must end with .mag");
            return snapshot_wrapper{filename, true};
        })
        .def_static("read", [](const std::string &filename) {
            if (!ends_with(filename, ".mag"))
                throw nb::value_error("Filename must end with .mag");
            return snapshot_wrapper{filename, false};
        })
        .def("__enter__", [](snapshot_wrapper &self) -> snapshot_wrapper& {
            return self;
        }, nb::rv_policy::reference_internal)
        .def("__exit__", [](snapshot_wrapper &self, nb::args) -> bool {
            self.close();
            return false;
        })
        .def("close", [](snapshot_wrapper &self) {
            self.close();
        })
        .def("put_tensor", [](snapshot_wrapper &self, const std::string &name, const tensor_wrapper &tensor) {
            if (!self.is_write_mode()) throw std::runtime_error("Snapshot opened in read mode");
            if (!mag_snapshot_put_tensor(*self, name.c_str(), *tensor)) throw std::runtime_error("Failed to store tensor");
        })
        .def("get_tensor", [](snapshot_wrapper &self, const std::string &name) {
            if (self.is_write_mode()) throw std::runtime_error("Snapshot opened in write mode");
            mag_tensor_t *t = mag_snapshot_get_tensor(*self, name.c_str());
            if (!t) throw std::runtime_error("Tensor not found");
            return tensor_wrapper{t};
        })
        .def("tensor_keys", [](snapshot_wrapper &self) {
            size_t n=0;
            const char **keys = mag_snapshot_get_tensor_keys(*self, &n);
            on_scope_exit defer([keys, n] { mag_snapshot_free_tensor_keys(keys, n); });
            nb::list out;
            for (size_t i=0; i < n; ++i)
                out.append(nb::str{keys[i]});
            return out;
        })
        .def("print_info", [](snapshot_wrapper &self) {
            mag_snapshot_print_info(*self);
        });
    }
}
