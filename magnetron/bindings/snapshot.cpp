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

#include <core/mag_io_snapshot_layout.h>

namespace mag::bindings {
  class snapshot_stream_writer final {
  public:
    snapshot_stream_writer(const std::string &filename, const std::string & meta, uint64_t data_len) {
      mag_error_t err {};
      throw_if_error(
        mag_snapshot_stream_writer_open(
          &err,
          &m_writer,
          get_ctx(),
          filename.c_str(),
          meta.c_str(),
          static_cast<uint64_t>(meta.size()),
          data_len
        ),
        err
      );
    }

    ~snapshot_stream_writer() noexcept {
      abort();
    }
    snapshot_stream_writer(const snapshot_stream_writer &) = delete;
    snapshot_stream_writer &operator=(const snapshot_stream_writer &) = delete;
    snapshot_stream_writer(snapshot_stream_writer &&rhs) noexcept : m_writer {std::exchange(rhs.m_writer, nullptr)} {}
    snapshot_stream_writer &operator=(snapshot_stream_writer &&rhs) noexcept {
      if (this == &rhs)
        return *this;
      abort();
      m_writer = std::exchange(rhs.m_writer, nullptr);
      return *this;
    }

    [[nodiscard]] bool is_open() const noexcept { return m_writer != nullptr; }

    void write(const void *data, std::uint64_t size) {
      require_open();
      mag_error_t err {};
      throw_if_error(mag_snapshot_stream_writer_submit_blob(&err, m_writer, data, size), err);
    }

    void close() {
      if (!m_writer) return;
      auto *writer = std::exchange(m_writer, nullptr);
      mag_error_t err {};
      throw_if_error(mag_snapshot_stream_writer_close(&err, writer), err);
    }

    void abort() noexcept {
      if (auto *writer = std::exchange(m_writer, nullptr))
        mag_snapshot_stream_writer_abort(writer);
    }

  private:
      void require_open() const {
        if (!m_writer)
          throw std::runtime_error {"SnapshotStreamWriter is closed"};
      }
      mag_snapshot_stream_writer_t *m_writer {};
  };

  void init_bindings_snapshot(nb::module_ &m) {
    m.attr("_SNAPSHOT_TBLOB_ALIGN") = static_cast<uint64_t>(MAG_SNAP_TENSOR_BLOB_ALIGN);
    nb::class_<snapshot_stream_writer>(m, "SnapshotStreamWriter")
      .def("__init__", [](snapshot_stream_writer *self, const std::string &filename, const std::string &meta, uint64_t data_len) -> void {
        nb::gil_scoped_release nogil;
        std::lock_guard lock {get_global_mutex()};
        new (self) snapshot_stream_writer {filename, meta, data_len};
      }, "filename"_a, "meta"_a, "data_len"_a)
      .def_prop_ro("is_open", &snapshot_stream_writer::is_open)
      .def("write", [](snapshot_stream_writer &self, nb::object chunk) -> void {
          Py_buffer view {};
          if (PyObject_GetBuffer(chunk.ptr(), &view, PyBUF_SIMPLE) != 0)
            throw nb::type_error("chunk must support the buffer protocol");
          on_scope_exit release([&]() -> void {
            PyBuffer_Release(&view);
          });
          nb::gil_scoped_release nogil {};
          std::lock_guard lock {get_global_mutex()};
          self.write(view.buf, static_cast<uint64_t>(view.len));
      }, "chunk"_a)
      .def(
        "write_tensor",
        [](snapshot_stream_writer &self, const tensor_wrapper &tensor) -> void {
          if (!mag_tensor_is_contiguous(*tensor)) throw nb::value_error("tensor must be contiguous");
          if (!mag_tensor_is_cpu(*tensor)) throw nb::value_error("tensor must reside on CPU");
          const void *data = reinterpret_cast<const void *>(mag_tensor_data_ptr(*tensor));
          auto nbytes = static_cast<uint64_t>(mag_tensor_numbytes(*tensor));
          nb::gil_scoped_release nogil {};
          std::lock_guard lock {get_global_mutex()};
          self.write(data, nbytes);
      }, "tensor"_a)
      .def("close", [](snapshot_stream_writer &self) -> void {
        nb::gil_scoped_release nogil {};
        std::lock_guard lock {get_global_mutex()};
        self.close();
      })
      .def("abort", [](snapshot_stream_writer &self) -> void {
        nb::gil_scoped_release nogil {};
        std::lock_guard lock {get_global_mutex()};
        self.abort();
      })
      .def("__enter__", [](snapshot_stream_writer &self) -> snapshot_stream_writer & {
        if (!self.is_open()) throw std::runtime_error {"SnapshotStreamWriter is closed"};
        return self;
      }, nb::rv_policy::reference_internal)
      .def("__exit__", [](snapshot_stream_writer &self, nb::handle exc_type, nb::handle, nb::handle) -> bool {
        bool has_exc = !exc_type.is_none();
        nb::gil_scoped_release nogil {};
        std::lock_guard lock {get_global_mutex()};
        if (has_exc) self.abort();
        else self.close();
        return false;
      }, "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none());
  }
}
