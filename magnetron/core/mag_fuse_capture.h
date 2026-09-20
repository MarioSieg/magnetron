/*
** Capturing a chain of operators instead of running them one at a time.
**
** Fusing a chain means seeing the whole chain before any of it runs, and eager execution never
** offers that view: by the time an operator is dispatched, its predecessor has already produced a
** tensor. So inside a fusion region the dispatcher records a fusible operator rather than submitting
** it, and marks its output pending - a tensor that exists, with storage allocated, but whose bytes
** have not been written yet.
**
** Everything else is about making that lie invisible. Reading a pending result or writing storage a
** pending chain still reads forces the chain to run first. The data-pointer accessors cover direct
** reads and writes, and dispatch checks the outputs of operators that cannot join the chain.
**
** The chain is re-recorded on every entry to the region rather than cached. A recorded tape would
** need a guard proving the control flow that produced it has not changed since; tracing afresh is
** its own guard, and a different path simply produces a different chain.
*/

#ifndef MAG_FUSE_CAPTURE_H
#define MAG_FUSE_CAPTURE_H

#include "mag_fuse_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
** Chain length before a flush is forced.
**
** Larger than the number of buffers a graph can bind, because most links in a long chain consume a
** value the previous link produced and never need a buffer of their own.
*/
#define MAG_FUSE_TAPE_MAX 48

/*
** Below this, recording a chain costs more than running the operators where they stand.
**
** Building the graph, walking the tape and submitting the chain is a fixed cost per chain, and on a
** small tensor the operators it replaces are already cheaper than that. Measured on an Apple M3
** against a two and four deep multiply-add chain: fusion loses at 1024 through 8192 elements
** (0.68x to 0.85x) and wins from 16384 upward.
**
** Treat the exact figure as provisional. The eager path it was measured against threads any operator
** above ten thousand elements, which costs about 12us of fan-out per operator and is far too eager -
** so part of what fusion appears to win just above that line is really eager paying for threading it
** should not have used. This wants re-deriving once those thresholds are themselves measured.
*/
#define MAG_FUSE_MIN_ELEMS 16384

/*
** Regions nest, and only leaving the outermost one runs the chain. Nesting is common by accident -
** a helper that opens a region called from code that already did - and flushing at every exit would
** chop chains at boundaries the author never intended to draw.
*/
extern MAG_EXPORT void mag_fuse_region_begin(mag_context_t *ctx);
extern MAG_EXPORT mag_status_t mag_fuse_region_end(mag_error_t *err, mag_context_t *ctx);
extern MAG_EXPORT bool mag_fuse_region_active(const mag_context_t *ctx);

/*
** Offer an operator to the chain.
**
** Sets *captured when the operator joined, which means the caller must not submit it. A false
** *captured is the ordinary outcome for anything the chain cannot take, and is not an error.
*/
extern mag_status_t mag_fuse_capture(
  mag_error_t *err,
  mag_context_t *ctx,
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out,
  bool *captured
);

/* Run whatever is on the tape and clear it. Safe to call when there is nothing pending. */
extern MAG_EXPORT mag_status_t mag_fuse_flush(mag_error_t *err, mag_context_t *ctx);

/* True when a pending chain still reads bytes the tensor may overwrite. The mutable data-pointer
   accessor uses this for writes that bypass mag_dispatch, such as copy_raw_. */
extern bool mag_fuse_tape_reads_storage(const mag_tensor_t *tensor);

/*
** How many chains have run, how many operators went into them, and how many results never had to be
** written at all.
*/
extern MAG_EXPORT void mag_fuse_stats(mag_context_t *ctx, uint64_t *out_chains, uint64_t *out_ops_fused, uint64_t *out_elided);

extern MAG_COLDPROC void mag_fuse_tape_shutdown(mag_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
