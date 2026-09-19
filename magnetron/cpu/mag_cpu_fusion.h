/*
** Turning a fused chain into compiled code.
**
** The interpreter next door already runs any chain, and correctly. What it cannot do is stop paying
** for itself: a dispatch per instruction per tile, operands reached through pointers it cannot prove
** anything about, and no way for the compiler to see that the whole chain is one expression. Writing
** the chain out as C and handing it to the host toolchain removes all three at once, because the
** result is the loop somebody would have written by hand.
**
** This is a CPU answer to a question core asked in the abstract. Core never learns that a compiler
** was involved, and a backend that lowers the same graph to PTX or a Metal library is doing the same
** job by different means.
**
** Compilation can always fail - no toolchain on the machine, a read-only cache directory, a compiler
** that does not like the flags - and none of that is an error. The chain simply runs interpreted,
** which is why the interpreter is not a fallback bolted on afterwards but the thing this has to
** prove itself against.
*/

#ifndef MAG_CPU_FUSION_H
#define MAG_CPU_FUSION_H

#include <core/mag_fuse_graph.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
** What a compiled chain looks like from the outside.
**
** Deliberately the same shape as the slice the interpreter runs, so the kernel picks one or the
** other and is otherwise identical. Immediates arrive here rather than being written into the code,
** so two chains differing only in a constant share one compiled kernel.
*/
typedef void (*mag_cpu_fused_fn_t)(void *const *bufs, const double *imms, int64_t begin, int64_t end);

typedef struct mag_cpu_fuse_cache_t mag_cpu_fuse_cache_t;

/* The C source for a graph. Caller frees with mag_cpu_fuse_source_free. NULL if the graph cannot be written. */
extern char *mag_cpu_fuse_codegen(const mag_fuse_graph_t *g);
extern void mag_cpu_fuse_source_free(char *src);

/*
** The compiled kernel for a graph, compiling it if this is the first time.
**
** NULL means run it interpreted. Once a compile has failed for want of a toolchain this stops
** trying, because the answer will not change and each attempt costs a process spawn.
*/
extern mag_cpu_fused_fn_t mag_cpu_fuse_resolve(mag_cpu_fuse_cache_t **cache, const mag_fuse_graph_t *g);

extern void mag_cpu_fuse_cache_stats(const mag_cpu_fuse_cache_t *cache, uint64_t *out_hits, uint64_t *out_compiles);
extern void mag_cpu_fuse_cache_destroy(mag_cpu_fuse_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif
