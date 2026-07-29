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
  class process_group_wrapper final {
  public:
    process_group_wrapper() = default;
    process_group_wrapper(
      const char *master_addr,
      uint16_t master_port,
      uint32_t rank,
      uint32_t world_size
    ) {
      mag_error_t err {};
      throw_if_error(mag_pgroup_init_tcp(&err, &m_pg, master_addr, master_port, rank, world_size), err);
    }
    process_group_wrapper(const process_group_wrapper &) = delete;
    process_group_wrapper &operator=(const process_group_wrapper &) = delete;
    process_group_wrapper(process_group_wrapper &&other) noexcept : m_pg{other.m_pg} {
      other.m_pg = nullptr;
    }
    process_group_wrapper &operator=(process_group_wrapper &&other) noexcept {
      if (this != &other) {
        reset();
        m_pg = other.m_pg;
        other.m_pg = nullptr;
      }
      return *this;
    }
    ~process_group_wrapper() {
      reset();
    }
    void reset() noexcept {
      if (m_pg) {
        mag_pgroup_destroy(m_pg);
        m_pg = nullptr;
      }
    }
    constexpr mag_process_group_t *operator * () const noexcept { return m_pg; }

  private:
    mag_process_group_t *m_pg = nullptr;
  };

  void init_bindings_distributed(nb::module_ &m) {
    auto distributed = m.def_submodule(
      "distributed",
      "Distributed process groups and tensor collectives."
    );
    nb::class_<process_group_wrapper>(distributed, "ProcessGroup")
      .def(
        "__init__", [](process_group_wrapper *self, const char *master_addr, uint16_t master_port, uint32_t rank, uint32_t world_size) {
          new (self) process_group_wrapper {
            master_addr,
            master_port,
            rank,
            world_size
          };
        },
        "master_addr"_a,
        "master_port"_a,
        "rank"_a,
        "world_size"_a,
        "Create a TCP process group."
      )
      .def_prop_ro("rank", [](const process_group_wrapper &self) noexcept -> uint32_t {
        return mag_pgroup_rank(*self);
      }, "Rank of the current process in the group.")
      .def_prop_ro("world_size", [](const process_group_wrapper &self) noexcept -> uint32_t {
        return mag_pgroup_world_size(*self);
      }, "Total number of processes in the group.")
      .def("barrier", [](process_group_wrapper &self) {
        mag_error_t err {};
        throw_if_error(mag_pgroup_barrier(&err, *self), err);
      }, "Synchronize all processes in the group.")
      .def("broadcast_", [](process_group_wrapper &self, const tensor_wrapper &tensor) -> void {
        mag_error_t err {};
        throw_if_error(mag_pgroup_broadcast_(&err, *self, *tensor), err);
      }, "tensor"_a, "Broadcast a tensor from rank 0 to all other ranks in the group.")
      .def("all_reduce_sum_", [](process_group_wrapper &self, const tensor_wrapper &tensor) -> void {
        mag_error_t err {};
        throw_if_error(mag_pgroup_all_reduce_sum_(&err, *self, *tensor), err);
      }, "tensor"_a, "Perform an in-place all-reduce sum on the tensor across all ranks in the group.")
      .def("__repr__", [](const process_group_wrapper &self) -> nb::str {
        return nb::str("ProcessGroup(rank={}, world_size={})") .format(mag_pgroup_rank(*self), mag_pgroup_world_size(*self));
      });
  }
}
