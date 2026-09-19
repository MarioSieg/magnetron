/*
** Turning a fused chain into a CUDA kernel.
**
** The graph core hands over says nothing about how it should run. The CPU backend answers by writing
** C and calling the host compiler; this one writes CUDA C and calls NVRTC, loads the PTX through the
** driver API and launches it on the device's own stream. Neither answer is visible from core, which
** is the point: the same graph, lowered twice, by two backends that share no code.
**
** A chain that cannot be lowered - an unsupported dtype, no NVRTC, a compile that fails - is
** declined, and core replays the operators one at a time. That path is what ran here before this
** file existed.
*/

#ifndef MAG_CUDA_FUSION_CUH
#define MAG_CUDA_FUSION_CUH

#include "mag_cuda_prelude.cuh"

#include <core/mag_fuse_graph.h>

namespace mag {
  /* Lower, cache and launch the chain the command carries. Declines by returning an error. */
  [[nodiscard]] extern mag_status_t fused_op(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);

  /* Unload every module this device compiled. Called when the device goes away. */
  extern void fused_cache_shutdown(int ordinal) noexcept;
}

#endif
