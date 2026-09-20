/*
** The unfused operations recorded by a fusion region, and the passes that select kernels.
**
** This graph records operator dependencies before any kernel is formed. Its device field is an
** opaque identity used only to keep one kernel on one device. Neither the graph nor its passes
** know how a backend executes the selected kernels.
*/

#ifndef MAG_FUSE_TRACE_H
#define MAG_FUSE_TRACE_H

#include "mag_fuse_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_FUSE_TAPE_MAX 48
#define MAG_FUSE_NO_PRODUCER (-1)
#define MAG_FUSE_NO_GROUP UINT8_MAX

typedef struct mag_fuse_trace_node_t {
  mag_opcode_t op;
  uint8_t num_in;
  uint8_t dtype;
  int64_t numel;
  const void *device;
  mag_tensor_t *in[3]; /* Tensor leaves and runtime values, owned by the capture tape. */
  mag_tensor_t *out;
  int16_t producer[3]; /* Earlier node, or MAG_FUSE_NO_PRODUCER for an external input. */
  bool observed;       /* The caller or a backward needs this result in storage. */
} mag_fuse_trace_node_t;

typedef struct mag_fuse_trace_t {
  uint8_t len;
  mag_fuse_trace_node_t nodes[MAG_FUSE_TAPE_MAX];
} mag_fuse_trace_t;

typedef struct mag_fuse_group_t {
  uint8_t len;
  bool fused; /* False for a pure operator that must run through ordinary backend dispatch. */
  uint8_t nodes[MAG_FUSE_TAPE_MAX]; /* Topological order within this execution group. */
} mag_fuse_group_t;

typedef struct mag_fuse_plan_t {
  uint8_t num_groups;
  uint8_t group_of[MAG_FUSE_TAPE_MAX]; /* MAG_FUSE_NO_GROUP for dead operations. */
  uint8_t order[MAG_FUSE_TAPE_MAX];    /* Topological order of the selected kernels. */
  bool live[MAG_FUSE_TAPE_MAX];
  bool store[MAG_FUSE_TAPE_MAX];       /* Materialize at an observable or group boundary. */
  mag_fuse_group_t groups[MAG_FUSE_TAPE_MAX];
} mag_fuse_plan_t;

/*
** Shared graph passes: backward liveness, fusion partitioning, boundary stores, and scheduling.
** Compatible independent operations may join a group even when another device or shape was
** recorded between them. A dependency cycle between proposed groups prevents that merge.
*/
extern MAG_EXPORT bool mag_fuse_trace_plan(const mag_fuse_trace_t *trace, mag_fuse_plan_t *plan);

#ifdef __cplusplus
}
#endif

#endif
