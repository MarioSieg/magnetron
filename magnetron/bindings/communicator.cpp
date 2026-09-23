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

#include <utility>

namespace mag::bindings {
  namespace {
    class communicator_wrapper final {
    public:
      communicator_wrapper(uint32_t rank, uint32_t size, const std::string &backend) {
        mag_error_t err {};
        throw_if_error(mag_comm_init(&err, &m_comm, rank, size, backend.c_str(), nullptr), err);
      }
      communicator_wrapper(const communicator_wrapper &) = delete;
      communicator_wrapper &operator=(const communicator_wrapper &) = delete;
      communicator_wrapper(communicator_wrapper &&rhs) noexcept : m_comm {std::exchange(rhs.m_comm, nullptr)} {}
      communicator_wrapper &operator=(communicator_wrapper &&rhs) noexcept {
        if (this == &rhs) return *this;
        destroy();
        m_comm = std::exchange(rhs.m_comm, nullptr);
        return *this;
      }
      ~communicator_wrapper() noexcept { destroy(); }

      void destroy() noexcept {
        if (auto *comm = std::exchange(m_comm, nullptr))
          mag_comm_destroy(comm);
      }

      [[nodiscard]] bool is_alive() const noexcept { return m_comm != nullptr; }
      [[nodiscard]] uint32_t rank() const { return mag_comm_rank(require()); }
      [[nodiscard]] uint32_t size() const { return mag_comm_size(require()); }
      [[nodiscard]] const char *backend() const { return mag_comm_backend_name(require()); }

      mag_communicator_t *require() const {
        if (!m_comm) throw std::runtime_error {"Communicator is destroyed"};
        return m_comm;
      }

      void require_rank(uint32_t rank, const char *what) const {
        uint32_t n = size();
        if (rank >= n) {
          std::ostringstream ss {};
          ss << what << " rank " << rank << " is out of range for communicator size " << n;
          throw nb::index_error(ss.str().c_str());
        }
      }

      template <typename F>
      void invoke(F &&fn) const {
        mag_communicator_t *comm = require();
        mag_error_t err {};
        throw_if_error(call_without_gil([&]() noexcept -> mag_status_t { return fn(&err, comm); }), err);
      }

    private:
      mag_communicator_t *m_comm {};
    };

    [[nodiscard]] std::vector<size_t> parse_counts(nb::handle h, size_t expected, const char *what) {
      std::vector<int64_t> raw = parse_i64_list_handle(h, what);
      if (raw.size() != expected) {
        std::ostringstream ss {};
        ss << what << " must have one entry per rank (" << expected << "), got " << raw.size();
        throw nb::value_error(ss.str().c_str());
      }
      std::vector<size_t> out {};
      out.reserve(raw.size());
      for (int64_t v : raw) {
        if (v < 0) {
          std::ostringstream ss {};
          ss << what << " must be non-negative, got " << v;
          throw nb::value_error(ss.str().c_str());
        }
        out.emplace_back(static_cast<size_t>(v));
      }
      return out;
    }
  }

  void init_bindings_distributed(nb::module_ &m) {
    auto distributed = m.def_submodule("_distributed", "Distributed communicators and tensor collectives.");

    nb::enum_<mag_reduce_op_t>(distributed, "ReduceOp", "Reduction applied by reduce collectives.")
      .value("SUM", MAG_REDUCE_SUM)
      .value("AVG", MAG_REDUCE_AVG)
      .value("PROD", MAG_REDUCE_PROD)
      .value("MIN", MAG_REDUCE_MIN)
      .value("MAX", MAG_REDUCE_MAX)
      .value("AND", MAG_REDUCE_AND)
      .value("OR", MAG_REDUCE_OR)
      .value("XOR", MAG_REDUCE_XOR);

    nb::class_<communicator_wrapper>(distributed, "Communicator", "Handle to a distributed communicator spanning a set of ranks.")
      .def("__init__", [](communicator_wrapper *self, uint32_t rank, uint32_t size, const std::string &backend) -> void {
        if (size == 0) throw nb::value_error("size must be > 0");
        if (rank >= size) {
          std::ostringstream ss {};
          ss << "rank " << rank << " is out of range for size " << size;
          throw nb::value_error(ss.str().c_str());
        }
        nb::gil_scoped_release nogil {};
        new (self) communicator_wrapper {rank, size, backend};
      }, "rank"_a, "size"_a, "backend"_a, "Create a communicator for the given rank within a group of the given size using the named backend.")
      .def_prop_ro("rank", &communicator_wrapper::rank, "Rank of the current process in the communicator.")
      .def_prop_ro("size", &communicator_wrapper::size, "Total number of ranks in the communicator.")
      .def_prop_ro("backend", [](const communicator_wrapper &self) -> nb::str {
        return nb::str {self.backend()};
      }, "Name of the backend that implements the collectives.")
      .def_prop_ro("is_alive", &communicator_wrapper::is_alive, "Whether the communicator has not been destroyed yet.")
      .def("destroy", &communicator_wrapper::destroy, "Release the communicator. Subsequent collective calls raise.")
      .def("barrier", [](const communicator_wrapper &self) -> void {
        self.invoke([](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_barrier(err, comm);
        });
      }, "Synchronize all ranks in the communicator.")
      .def("broadcast_", [](const communicator_wrapper &self, const tensor_wrapper &tensor, uint32_t root) -> void {
        self.require_rank(root, "root");
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_broadcast(err, comm, *tensor, root);
        });
      }, "tensor"_a, "root"_a = 0, "Broadcast the tensor in-place from the root rank to all other ranks.")
      .def("reduce_", [](const communicator_wrapper &self, const tensor_wrapper &tensor, uint32_t root, mag_reduce_op_t op) -> void {
        self.require_rank(root, "root");
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_reduce(err, comm, *tensor, root, op);
        });
      }, "tensor"_a, "root"_a = 0, "op"_a = MAG_REDUCE_SUM, "Reduce the tensor in-place across all ranks; only the root rank receives the result.")
      .def("all_reduce_", [](const communicator_wrapper &self, const tensor_wrapper &tensor, mag_reduce_op_t op) -> void {
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_all_reduce(err, comm, *tensor, op);
        });
      }, "tensor"_a, "op"_a = MAG_REDUCE_SUM, "Reduce the tensor in-place across all ranks; every rank receives the result.")
      .def("all_gather", [](const communicator_wrapper &self, const tensor_wrapper &out, const tensor_wrapper &in) -> void {
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_all_gather(err, comm, *out, *in);
        });
      }, "out"_a, "input"_a, "Gather the input tensor of every rank into the output tensor on every rank.")
      .def("reduce_scatter", [](const communicator_wrapper &self, const tensor_wrapper &out, const tensor_wrapper &in, mag_reduce_op_t op) -> void {
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_reduce_scatter(err, comm, *out, *in, op);
        });
      }, "out"_a, "input"_a, "op"_a = MAG_REDUCE_SUM, "Reduce the input tensor across all ranks and scatter one equal chunk of the result to each rank's output tensor.")
      .def("all_to_all", [](const communicator_wrapper &self, const tensor_wrapper &out, const tensor_wrapper &in) -> void {
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_all_to_all(err, comm, *out, *in);
        });
      }, "out"_a, "input"_a, "Exchange equal-sized chunks of the input tensor between all ranks into the output tensor.")
      .def("all_to_all_v", [](
        const communicator_wrapper &self,
        const tensor_wrapper &out,
        const tensor_wrapper &in,
        nb::handle send_counts,
        nb::handle send_offsets,
        nb::handle recv_counts,
        nb::handle recv_offsets
      ) -> void {
        size_t n = self.size();
        std::vector<size_t> sc = parse_counts(send_counts, n, "send_counts");
        std::vector<size_t> so = parse_counts(send_offsets, n, "send_offsets");
        std::vector<size_t> rc = parse_counts(recv_counts, n, "recv_counts");
        std::vector<size_t> ro = parse_counts(recv_offsets, n, "recv_offsets");
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_all_to_all_v(err, comm, *out, *in, sc.data(), so.data(), rc.data(), ro.data());
        });
      }, "out"_a, "input"_a, "send_counts"_a, "send_offsets"_a, "recv_counts"_a, "recv_offsets"_a,
      "Exchange variable-sized chunks between all ranks. Each count and offset sequence has one element-count entry per rank.")
      .def("send", [](const communicator_wrapper &self, const tensor_wrapper &tensor, uint32_t dst) -> void {
        self.require_rank(dst, "destination");
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_send(err, comm, *tensor, dst);
        });
      }, "tensor"_a, "dst"_a, "Send the tensor to the destination rank.")
      .def("recv", [](const communicator_wrapper &self, const tensor_wrapper &tensor, uint32_t src) -> void {
        self.require_rank(src, "source");
        self.invoke([&](mag_error_t *err, mag_communicator_t *comm) noexcept -> mag_status_t {
          return mag_comm_recv(err, comm, *tensor, src);
        });
      }, "tensor"_a, "src"_a, "Receive into the tensor from the source rank.")
      .def("__enter__", [](communicator_wrapper &self) -> communicator_wrapper & {
        self.require();
        return self;
      }, nb::rv_policy::reference_internal)
      .def("__exit__", [](communicator_wrapper &self, nb::handle, nb::handle, nb::handle) -> bool {
        self.destroy();
        return false;
      }, "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none())
      .def("__repr__", [](const communicator_wrapper &self) -> nb::str {
        if (!self.is_alive()) return nb::str {"Communicator(destroyed)"};
        return nb::str("Communicator(rank={}, size={}, backend={!r})").format(self.rank(), self.size(), self.backend());
      });
  }
}
