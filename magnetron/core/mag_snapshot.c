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

#include "mag_alloc.h"
#include "mag_mmap.h"
#include "mag_romap.h"
#include "mag_tensor.h"

#include <time.h>

#include "mag_context.h"
#include "../cpu/mag_cpu.h"


/*
** File Format:
** ================= Full File Overview =================
**
** +----------------------+ <- Magic: "MAG!"
** | File Header          |
** +----------------------+ <- Section Marker: "SRP!"
** | String Pool          |
** +----------------------+ <- Section Marker: "MDT!
** | Metadata Map         |
** +----------------------+ <- Section Marker: "DSC!"
** | Tensor Header Map    |
** +----------------------+ <- Section Marker: "BUF!"
** | Tensor Data          |
** +----------------------+
*/

#define mag_snap_verify(expr, action) \
if (mag_unlikely(!(expr))) { \
    mag_log_error("Error reading/writing snapshot file: " #expr); \
    action; \
}

#define mag_snap_pack4_ne(a,b,c,d) ((((d)&255)<<24)+(((c)&255)<<16)+(((b)&255)<<8)+((a)&255))

#define MAG_SNAP_MAX_STRLEN 0xffff
#define MAG_SNAP_MAX_RANK 64
#define MAG_SNAP_MAX_STR_POOL_BLOB_SIZE (128ull<<20) /* 128 MiB */
#define MAG_SNAP_MAX_OFFSETS 0xffff
#define MAG_SNAPSHOT_META_MAP_DEFAULT_CAP 32
#define MAG_SNAP_FILE_MAGIC mag_snap_pack4_ne('M', 'A', 'G', '!')
#define MAG_SNAP_SECTION_STR_POOL mag_snap_pack4_ne('S', 'R', 'P', '!')
#define MAG_SNAP_SECTION_META_DATA mag_snap_pack4_ne('M', 'D', 'T', '!')
#define MAG_SNAP_SECTION_TENSOR_DESC mag_snap_pack4_ne('D', 'S', 'C', '!')
#define MAG_SNAP_SECTION_TENSOR_DATA mag_snap_pack4_ne('B', 'U', 'F', '!')
#define MAG_SNAP_SECTION_MARKERS_COUNT 4 /* File magic is not included, belongs to file header */
#define MAG_SNAP_TBUF_ALIGN 16 /* Every tensor buffer start address must be aligned to this */

#ifdef MAG_BIG_ENDIAN
/*
** If some annoying host is BE, support could be added by byte-wapping tensor buffer elements with COW mmap.
** Not yet done at the moment.
** Only the data section requires handling for BE, the headers and metadata already do endinaess swapping
*/
#error "Big endian is not supported at the moment"
#endif

typedef struct mag_mmap_owner_t {
    MAG_RC_INJECT_HEADER;
    mag_mapped_file_t file;
} mag_mmap_owner_t;

static void mag_mmap_owner_dtor(void *self) {
    mag_mmap_owner_t *o = self;
    mag_unmap_file(&o->file);
    (*mag_alloc)(o, 0, 0);
}

static mag_mmap_owner_t *mag_mmap_owner_open(const char *path) {
    mag_mmap_owner_t *o = (*mag_alloc)(NULL, sizeof(*o), 0);
    memset(o, 0, sizeof(*o));
    if (!mag_map_file(&o->file, path, 0, MAG_MAP_READ)) {
        (*mag_alloc)(o, 0, 0);
        return NULL;
    }
    mag_rc_init_object(o, &mag_mmap_owner_dtor);
    return o;
}

typedef enum mag_mem_stream_flags_t {
    MAG_MEM_STREAM_FLAGS_NONE = 0,
    MAG_MEM_STREAM_FLAGS_WRITE = 1<<1
} mag_mem_stream_flags_t;

typedef struct mag_mem_stream_t {
    uint8_t *base;
    uint8_t *pos;
    uint8_t *end;
    mag_mem_stream_flags_t flags;
} mag_mem_stream_t;

static bool mag_stream_from_mapped_file(mag_mem_stream_t *s, mag_mmap_owner_t *owner, bool write) {
    memset(s, 0, sizeof(*s));
    mag_snap_verify(owner && owner->file.map && owner->file.fs, return false);
    s->base = s->pos = owner->file.map;
    s->end = s->base + owner->file.fs;
    if (write) s->flags |= MAG_MEM_STREAM_FLAGS_WRITE;
    return true;
}

static void mag_stream_close(mag_mem_stream_t *stream) {
    if (!stream) return;
    memset(stream, 0, sizeof(*stream));
}

static bool mag_stream_mmap_file_w(mag_mem_stream_t *stream, mag_mapped_file_t *map, const char *path, size_t size) {
    memset(stream, 0, sizeof(*stream));
    mag_snap_verify(path != NULL && *path, return false);
    mag_snap_verify(size > 0, return false);
    mag_snap_verify(mag_map_file(map, path, size, MAG_MAP_WRITE), return false);
    stream->base = stream->pos = map->map;
    stream->end = stream->base + map->fs;
    stream->flags |= MAG_MEM_STREAM_FLAGS_WRITE;
    return true;
}

static size_t mag_stream_needle(const mag_mem_stream_t *stream) { return (size_t)(stream->pos - stream->base); }
static size_t mag_stream_remaining(const mag_mem_stream_t *stream) { return (size_t)(stream->end - stream->pos); }

static bool mag_stream_wu32_le(mag_mem_stream_t *stream, uint32_t val) {
    mag_snap_verify((size_t)(stream->end - stream->pos) >= sizeof(val), return false);
    mag_snap_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap32(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru32_le(mag_mem_stream_t *stream, uint32_t *val) {
    mag_snap_verify(val != NULL, return false);
    mag_snap_verify((size_t)(stream->end - stream->pos) >= sizeof(*val), return false);
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap32(*val);
    #endif
    return true;
}

static bool mag_stream_wu64_le(mag_mem_stream_t *stream, uint64_t val) {
    mag_snap_verify((size_t)(stream->end - stream->pos) >= sizeof(val), return false);
    mag_snap_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap64(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru64_le(mag_mem_stream_t *stream, uint64_t *val) {
    mag_snap_verify(val != NULL, return false);
    mag_snap_verify((size_t)(stream->end - stream->pos) >= sizeof(*val), return false);
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap64(*val);
    #endif
    return true;
}

static bool mag_stream_wstr(mag_mem_stream_t *stream, const uint8_t *str) {
    mag_snap_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    size_t len = strlen((const char *)str);
    mag_snap_verify(len <= MAG_SNAP_MAX_STRLEN && len <= UINT32_MAX, return false);
    mag_snap_verify(mag_utf8_validate(str, len), return false);
    mag_snap_verify(mag_stream_wu32_le(stream, (uint32_t)len), return false);
    mag_snap_verify((size_t)(stream->end - stream->pos) >= len, return false);
    memcpy(stream->pos, str, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rstr(mag_mem_stream_t *stream, uint8_t **out_str) {
    uint32_t len = 0;
    mag_snap_verify(mag_stream_ru32_le(stream, &len), return false);
    mag_snap_verify(len <= MAG_SNAP_MAX_STRLEN, return false);
    mag_snap_verify((size_t)(stream->end - stream->pos) >= len, return false);
    uint8_t *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(str, stream->pos, len);
    str[len] = '\0';
    mag_snap_verify(mag_utf8_validate(str, len), return false);
    stream->pos += len;
    *out_str = str;
    return true;
}

static bool mag_stream_wbuf(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_snap_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_snap_verify(buf != NULL || len == 0, return false);
    mag_snap_verify(len <= UINT32_MAX, return false);
    mag_snap_verify(mag_stream_wu32_le(stream, (uint32_t)len), return false);
    mag_snap_verify((size_t)(stream->end - stream->pos) >= len, return false);
    if (len) {
        memcpy(stream->pos, buf, len);
        stream->pos += len;
    }
    return true;
}

static bool mag_stream_wbytes(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_snap_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_snap_verify((size_t)(stream->end - stream->pos) >= len, return false);
    if (len) memcpy(stream->pos, buf, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rbytes_view(mag_mem_stream_t *s, const uint8_t **out, size_t len) {
    mag_snap_verify(out != NULL, return false);
    mag_snap_verify((size_t)(s->end - s->pos) >= len, return false);
    *out = s->pos;
    s->pos += len;
    return true;
}

/*
** Contains the file header structure.
** Not directly written to file due to possible packing issues
** De/serialization is done manually.
*/
typedef struct mag_file_header_t {
    uint32_t magic;
    uint32_t version;
    uint64_t timestamp; /* 64-bit Unix epoch */
    uint32_t checksum;
    uint32_t aux;
    uint32_t metadata_map_len;
    uint32_t tensor_header_count;
} mag_file_header_t;

#define MAG_FILE_HEADER_SIZE (4+4+8+4+4+4+4) /* We don't rely on struct packing */
mag_static_assert(sizeof(mag_file_header_t) % 4 == 0);
mag_static_assert(sizeof(mag_file_header_t) == MAG_FILE_HEADER_SIZE);

static bool mag_file_header_serialize(const mag_file_header_t *header, mag_mem_stream_t *stream, uint8_t **u32_chk_patch_needle) {
    mag_snap_verify(header->magic == MAG_SNAP_FILE_MAGIC, return false);
    mag_snap_verify(mag_stream_wu32_le(stream, header->magic), return false);
    mag_snap_verify(header->version == MAG_SNAPSHOT_VERSION, return false); /* Reading older versions is supporting, writing is not */
    mag_snap_verify(mag_stream_wu32_le(stream, header->version), return false);
    mag_snap_verify(mag_stream_wu64_le(stream, header->timestamp), return false);
    *u32_chk_patch_needle = stream->pos; /* Needle where the checksum is overwritten later */
    mag_snap_verify(mag_stream_wu32_le(stream, header->checksum), return false);
    mag_snap_verify(mag_stream_wu32_le(stream, header->aux), return false);
    mag_snap_verify(mag_stream_wu32_le(stream, header->metadata_map_len), return false);
    mag_snap_verify(mag_stream_wu32_le(stream, header->tensor_header_count), return false);
    return true;
}

static bool mag_file_header_deserialize(mag_file_header_t *header, mag_mem_stream_t *stream) {
    mag_snap_verify(mag_stream_ru32_le(stream, &header->magic), return false);
    mag_snap_verify(header->magic == MAG_SNAP_FILE_MAGIC, return false);
    mag_snap_verify(mag_stream_ru32_le(stream, &header->version), return false);
    mag_snap_verify(header->version <= MAG_SNAPSHOT_VERSION, return false);
    mag_snap_verify(mag_stream_ru64_le(stream, &header->timestamp), return false);
    mag_snap_verify(mag_stream_ru32_le(stream, &header->checksum), return false);
    mag_snap_verify(mag_stream_ru32_le(stream, &header->aux), return false);
    mag_snap_verify(mag_stream_ru32_le(stream, &header->metadata_map_len), return false);
    mag_snap_verify(mag_stream_ru32_le(stream, &header->tensor_header_count), return false);
    return true;
}

static uint32_t mag_pack4xu8_le(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return (uint32_t)a|((uint32_t)b<<8)|((uint32_t)c<<16)|((uint32_t)d<<24);
}

static void mag_unpack4xu8_le(uint32_t packed, uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *d) {
    *a = (uint8_t)packed;
    *b = (uint8_t)(packed>>8);
    *c = (uint8_t)(packed>>16);
    *d = (uint8_t)(packed>>24);
}

typedef struct mag_tensor_desc_t {
    uint8_t rank; /* 0..MAG_SNAP_MAX_RANK */
    mag_dtype_t dtype;
    uint8_t aux0;
    uint8_t aux1;
    uint32_t key_id;
    uint64_t numel;
    uint64_t offset;
    uint64_t shape[MAG_SNAP_MAX_RANK];
} mag_tensor_desc_t;
#define MAG_TENSOR_DESC_SIZE(rank) (4+4+8+8 + 8*(rank))

static bool mag_tensor_desc_serialize(const mag_tensor_desc_t *desc, mag_mem_stream_t *stream) {
    mag_snap_verify(mag_stream_wu32_le(stream, mag_pack4xu8_le(desc->rank, desc->dtype, desc->aux0, desc->aux1)), return false);
    mag_snap_verify(mag_stream_wu32_le(stream, desc->key_id), return false);
    mag_snap_verify(mag_stream_wu64_le(stream, desc->numel), return false);
    mag_snap_verify(mag_stream_wu64_le(stream, desc->offset), return false);
    for (uint8_t i=0; i < desc->rank; ++i)
        mag_snap_verify(mag_stream_wu64_le(stream, desc->shape[i]), return false);
    return true;
}

static bool mag_tensor_desc_deserialize(mag_tensor_desc_t *desc, mag_mem_stream_t *stream, uint32_t pool_len) {
    uint32_t packed = 0;
    mag_snap_verify(mag_stream_ru32_le(stream, &packed), return false);
    uint8_t dtype;
    mag_unpack4xu8_le(packed, &desc->rank, &dtype, &desc->aux0, &desc->aux1);
    mag_snap_verify(desc->rank < MAG_SNAP_MAX_RANK, return false);
    mag_snap_verify(dtype < MAG_DTYPE__NUM, return false);
    desc->dtype = dtype;
    mag_snap_verify(mag_stream_ru32_le(stream, &desc->key_id), return false);
    mag_snap_verify(desc->key_id < pool_len, return false);
    mag_snap_verify(mag_stream_ru64_le(stream, &desc->numel), return false);
    mag_snap_verify(desc->numel > 0 && desc->numel <= INT64_MAX, return false);
    mag_snap_verify(mag_stream_ru64_le(stream, &desc->offset), return false);     /* TODO: verify offset */
    int64_t numel_prod = 1;
    for (uint8_t i=0; i < desc->rank; ++i) {
        uint64_t dim=0;
        mag_snap_verify(mag_stream_ru64_le(stream, &dim), return false);
        mag_snap_verify(dim <= INT64_MAX, return false);
        mag_snap_verify(!mag_mulov64(dim, numel_prod, &numel_prod), return false);
        desc->shape[i] = dim;
    }
    mag_snap_verify(numel_prod <= INT64_MAX && numel_prod == desc->numel, return false);
    return true;
}

typedef struct mag_pool_record_t {
    const uint8_t *ptr;
    uint32_t len;
} mag_pool_record_t;

typedef struct mag_string_pool_t {
    mag_map_t map;
    mag_pool_record_t *records;
    size_t len;
    size_t cap;
} mag_string_pool_t;

static void mag_pool_init(mag_string_pool_t *pool) {
    memset(pool, 0, sizeof(*pool));
    mag_map_init(&pool->map, 256, true); /* TODO: we don't want this */
}

static void mag_pool_free(mag_string_pool_t *pool) {
    mag_map_free(&pool->map);
    (*mag_alloc)(pool->records, 0, 0);
    memset(pool, 0, sizeof(*pool));
}

static bool mag_pool_intern(mag_string_pool_t *pool, const uint8_t *buf, size_t len, uint32_t *out_id) {
    mag_snap_verify(buf && len && len < UINT32_MAX, return false);
    mag_snap_verify(mag_utf8_validate(buf, len), return false);
    void *found = mag_map_lookup(&pool->map, buf, len);
    if (found) {
        *out_id = (uint32_t)(uintptr_t)found-1;  /* unbias */
        return true;
    }
    mag_snap_verify(pool->len < UINT32_MAX, return false);
    *out_id = pool->len++;
    if (pool->len > pool->cap) {
        size_t cap = pool->cap ? pool->cap : 32;
        while (cap < pool->len) cap <<= 1;
        pool->records = (*mag_alloc)(pool->records, cap*sizeof(*pool->records), 0);
        pool->cap = cap;
    }
    mag_map_insert_if_absent(&pool->map, buf, len, (void *)(uintptr_t)(1+*out_id)); /* bias by 1 to distinguish from NULL */
    const uint8_t *owned = mag_map_lookup_key_ptr(&pool->map, buf, len);
    mag_snap_verify(owned, return false);
    mag_pool_record_t *rec = pool->records+*out_id;
    rec->ptr = owned;
    rec->len = (uint32_t)len;
    return true;
}

static bool mag_pool_serialize(const mag_string_pool_t *pool, mag_mem_stream_t *stream) {
    mag_snap_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_snap_verify(pool && pool->len <= UINT32_MAX, return false);
    mag_snap_verify(mag_stream_wu32_le(stream, (uint32_t)pool->len), return false);
    mag_snap_verify(mag_stream_wu32_le(stream, 0), return false); /* offsets[0] = 0, for monotonically and clean O(1) offsets */
    uint32_t offs = 0;
    for (size_t i=0; i < pool->len; ++i) { /* Offset array */
        mag_pool_record_t *rec = pool->records+i;
        mag_snap_verify((rec->ptr || !rec->len) && rec->len <= UINT32_MAX, return false);
        mag_snap_verify(UINT32_MAX-offs >= rec->len, return false);
        offs += rec->len;
        mag_snap_verify(mag_stream_wu32_le(stream, offs), return false);
    }
    for (size_t i=0; i < pool->len; ++i) { /* String blob */
        mag_pool_record_t *rec = pool->records+i;
        mag_snap_verify(mag_stream_wbytes(stream, rec->ptr, rec->len), return false);
    }
    return true;
}

static bool mag_pool_deserialize(mag_string_pool_t *pool, mag_mem_stream_t *stream) {
    mag_pool_free(pool);
    mag_pool_init(pool);
    mag_assert2(pool->len == 0); /*Pool must be fresh */
    uint32_t count = 0;
    mag_snap_verify(mag_stream_ru32_le(stream, &count), return false);
    size_t num_offsets = (size_t)count+1;
    mag_snap_verify(num_offsets <= MAG_SNAP_MAX_OFFSETS, return false);
    uint32_t *offs = (*mag_alloc)(NULL, num_offsets*sizeof(*offs), 0);
    for (size_t i=0; i < num_offsets; ++i) /* Read in offsets */
        mag_snap_verify(mag_stream_ru32_le(stream, offs+i), goto fail);
    mag_snap_verify(*offs == 0, goto fail);
    for (size_t i=1; i < num_offsets; ++i)
        mag_snap_verify(offs[i] >= offs[i-1], goto fail); /* Monotonic verify */
    uint32_t blob_size = offs[count];
    mag_snap_verify(blob_size <= MAG_SNAP_MAX_STR_POOL_BLOB_SIZE, goto fail);
    const uint8_t *blob = NULL;
    mag_snap_verify(mag_stream_rbytes_view(stream, &blob, blob_size), goto fail);
    for (uint32_t id=0; id < count; ++id) {
        uint32_t a = offs[id];
        uint32_t b = offs[id+1];
        mag_snap_verify(a <= b && b <= blob_size, goto fail);
        const uint8_t *str = blob+a;
        uint32_t delta = b-a;
        mag_snap_verify(delta, goto fail);
        mag_snap_verify(mag_utf8_validate(str, delta), goto fail);
        uint32_t len = 0;
        mag_snap_verify(mag_pool_intern(pool, str, delta, &len), goto fail);
        mag_snap_verify(len == id, goto fail);
    }
    (*mag_alloc)(offs, 0, 0);
    return true;
fail:
    (*mag_alloc)(offs, 0, 0);
    mag_pool_free(pool);
    mag_pool_init(pool);
    return false;
}

static size_t mag_pool_compute_size(mag_string_pool_t *pool) {
    size_t nb = sizeof(uint32_t); /* Count */
    nb += sizeof(uint32_t)*(pool->len+1); /* Offsets */
    for (size_t i=0; i < pool->len; ++i)
        nb += pool->records[i].len; /* Bytes */
    return nb;
}

static bool mag_pool_find_id(mag_string_pool_t *pool, const uint8_t *buf, size_t len, uint32_t *out_id) {
    mag_snap_verify(pool && buf && len && out_id, return false);
    void *found = mag_map_lookup(&pool->map, buf, len);
    if (!found) return false;
    *out_id = (uint32_t)(uintptr_t)found-1; /* unbias */
    return true;
}

struct mag_snapshot_t {
    mag_context_t *ctx;
    mag_string_pool_t str_pool;
    mag_map_t tensor_map;
    mag_mem_stream_t stream;
    mag_mmap_owner_t *mmap_owner;
    size_t nb_total;
    size_t nb_meta;
    size_t nb_storage;
};

static size_t mag_snaprage_compute_tensor_sizes(mag_map_t *tmap) {
    size_t nb = 0, iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(tmap, &iter, &len, &val)) {  /* Tensors */
        mag_tensor_t *tensor = val;
        nb += MAG_TENSOR_DESC_SIZE(tensor->coords.rank);
        nb += mag_tensor_numbytes(tensor);
    }
    return nb;
}

static size_t mag_snaprage_compute_size(mag_snapshot_t *snap) {
    size_t nb = 0;
    nb += MAG_FILE_HEADER_SIZE; /* File Header */
    nb += 4*MAG_SNAP_SECTION_MARKERS_COUNT; /* Markers */
    nb += mag_pool_compute_size(&snap->str_pool);
    nb += mag_snaprage_compute_tensor_sizes(&snap->tensor_map);
    return nb;
}

mag_snapshot_t *mag_snapshot_new(mag_context_t *ctx) {
    mag_snapshot_t *snap = (*mag_alloc)(NULL, sizeof(*snap), 0);
    memset(snap, 0, sizeof(*snap));
    snap->ctx = ctx;
    mag_pool_init(&snap->str_pool);
    mag_map_init(&snap->tensor_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, true);
    return snap;
}

void mag_snapshot_free(mag_snapshot_t *snap) {
    mag_pool_free(&snap->str_pool);
    size_t iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(&snap->tensor_map, &iter, &len, &val)) /* Free cloned metadata records */
        if (val) mag_tensor_decref(val);
    if (snap->mmap_owner)
        mag_rc_decref(snap->mmap_owner);
    mag_map_free(&snap->tensor_map);
    memset(snap, 0, sizeof(*snap));
    (*mag_alloc)(snap, 0, 0);
}

static bool mag_snapshot_insert_tensor_by_id(mag_snapshot_t *snap, uint32_t key_id, mag_tensor_t *tensor) {
    mag_snap_verify(snap && tensor, return false);
    mag_snap_verify(key_id < snap->str_pool.len, return false);
    if (mag_unlikely(mag_map_lookup(&snap->tensor_map, &key_id, sizeof(key_id)))) return false; /* Already exists */
    mag_map_insert_if_absent(&snap->tensor_map, &key_id, sizeof(key_id), tensor);
    mag_tensor_incref(tensor);
    return true;
}

extern mag_status_t mag_tensor_init(mag_tensor_t **out, mag_context_t *ctx, mag_storage_buffer_t *storage, mag_dtype_t type, int64_t rank, const int64_t *shape);

static void mag_cpu_storage_dtor(void *self) {
    mag_storage_buffer_t *buf = self;
    mag_context_t *ctx = buf->ctx;
    mag_assert(ctx->telemetry.num_alive_storages > 0, "double freed storage");
    --ctx->telemetry.num_alive_storages;
    if (buf->flags & MAG_STORAGE_FLAG_BORROWED) {
        mag_mmap_owner_t *owner = (mag_mmap_owner_t *)(uintptr_t)buf->aux.impl;
        if (owner) mag_rc_decref(owner);
    } else {
        (*mag_alloc)((void *)buf->base, 0, MAG_CPU_BUF_ALIGN);
    }
    mag_slab_free(&ctx->storage_slab, buf);
}

static void mag_cpu_borrow_storage(
    mag_device_t *cpu,
    mag_storage_buffer_t **out,
    const void *ptr,
    size_t size,
    mag_dtype_t dtype,
    mag_mmap_owner_t *owner /* retained */
) {
    mag_context_t *ctx = cpu->ctx;
    mag_storage_buffer_t *buf = mag_slab_alloc(&ctx->storage_slab);
    *buf = (mag_storage_buffer_t){
        .ctx = ctx,
        .aux = {0},
        .flags = MAG_STORAGE_FLAG_BORROWED,
        .base = (uintptr_t)ptr,
        .size = size,
        .alignment = 1,
        .dtype = dtype,
        .granularity = mag_type_trait(dtype)->size,
        .device = cpu,
    };
    buf->flags &= ~MAG_STORAGE_FLAG_ACCESS_W;
    buf->aux.impl = owner;
    mag_rc_incref(owner);
    mag_rc_init_object(buf, &mag_cpu_storage_dtor);
    ++ctx->telemetry.num_alive_storages;
    *out = buf;
}

mag_snapshot_t *mag_snapshot_deserialize(mag_context_t *ctx, const char *filename) {
    mag_snap_verify(filename && *filename, return false);
    const char *ext = strrchr(filename, '.'); /* check that the file extension is .mag */
    mag_snap_verify(ext != NULL && strcmp(ext, ".mag") == 0, return false);
    mag_snap_verify(ctx->device->id.type == MAG_BACKEND_TYPE_CPU, return false); /* Only CPU deserialization is supported at the moment */

    mag_tensor_desc_t *stable = NULL;
    mag_snapshot_t *snap = mag_snapshot_new(ctx);
    mag_mem_stream_t *stream = &snap->stream;
    snap->mmap_owner = mag_mmap_owner_open(filename);
    mag_snap_verify(snap->mmap_owner != NULL, goto error);
    mag_snap_verify(mag_stream_from_mapped_file(stream, snap->mmap_owner, false), goto error);
    mag_snap_verify(mag_stream_remaining(stream) >= MAG_FILE_HEADER_SIZE + 4*MAG_SNAP_SECTION_MARKERS_COUNT, goto error); /* We must at minimum have enough bytes for an empty file */

    snap->nb_total = mag_stream_remaining(stream);
    size_t marker = mag_stream_needle(stream);

    /* File header */
    mag_file_header_t header = {0};
    mag_snap_verify(mag_file_header_deserialize(&header, stream), goto error)
    mag_assert2(mag_stream_needle(stream)-marker == MAG_FILE_HEADER_SIZE); /* Verify exact file header bytes written */

    /* String pool */
    marker = mag_stream_needle(stream);
    uint32_t section_marker = 0;
    mag_snap_verify(mag_stream_ru32_le(stream, &section_marker), goto error);
    mag_snap_verify(section_marker == MAG_SNAP_SECTION_STR_POOL, goto error);
    mag_snap_verify(mag_pool_deserialize(&snap->str_pool, stream), goto error);
    mag_assert2(mag_stream_needle(stream)-marker == 4+mag_pool_compute_size(&snap->str_pool)); /* Verify exact section marker + pool bytes written */

    mag_snap_verify(mag_stream_ru32_le(stream, &section_marker), goto error);
    mag_snap_verify(section_marker == MAG_SNAP_SECTION_META_DATA, goto error);
    /* TODO: metadata */

    size_t nt = header.tensor_header_count;
    stable = (*mag_alloc)(NULL, nt*sizeof(*stable), 0);

    mag_snap_verify(mag_stream_ru32_le(stream, &section_marker), goto error);
    mag_snap_verify(section_marker == MAG_SNAP_SECTION_TENSOR_DESC, goto error);
    for (uint32_t i=0; i < nt; ++i) {
        mag_tensor_desc_t *desc = stable+i;
        mag_snap_verify(mag_tensor_desc_deserialize(desc, stream, snap->str_pool.len), goto error);
    }
    /* Read data */
    mag_snap_verify(mag_stream_ru32_le(stream, &section_marker), goto error);
    mag_snap_verify(section_marker == MAG_SNAP_SECTION_TENSOR_DATA, goto error);

    snap->nb_meta = mag_stream_needle(stream); /* Everything up to here is metadata */

    uint64_t offs=0;
    for (size_t i=0; i < nt; ++i) {
        const mag_tensor_desc_t *desc = stable+i;
        uint64_t offset = desc->offset;
        int64_t shape[MAG_SNAP_MAX_RANK];
        for (uint8_t j=0; j < desc->rank && j < sizeof(shape)/sizeof(*shape); ++j)
            shape[j] = (int64_t)desc->shape[j];
        size_t nbytes = (size_t)desc->numel*(size_t)mag_type_trait(desc->dtype)->size;
        mag_snap_verify(offset == offs, goto error); /* Verify offset */
        const uint8_t *blob = NULL;
        mag_snap_verify(mag_stream_rbytes_view(stream, &blob, nbytes), goto error);
        mag_storage_buffer_t *storage = NULL;
        mag_cpu_borrow_storage(ctx->device, &storage, blob, nbytes, desc->dtype, snap->mmap_owner);
        mag_tensor_t *tensor = NULL;
        mag_snap_verify(mag_isok(mag_tensor_init(&tensor, ctx, storage, desc->dtype, desc->rank, shape)), goto error);
        mag_snap_verify(mag_snapshot_insert_tensor_by_id(snap, desc->key_id, tensor), mag_tensor_decref(tensor); goto error);
        mag_tensor_decref(tensor); /* Decref as the snapshot now holds a reference */
        mag_rc_decref(storage); /* Decref as the snapshot now holds a reference */
        offs += nbytes;
    }

    snap->nb_storage = mag_stream_needle(stream) - snap->nb_meta;
    mag_snap_verify(snap->nb_total == snap->nb_meta + snap->nb_storage, goto error);

    (*mag_alloc)(stable, 0, 0);
    return snap;
    error:
        if (stable) (*mag_alloc)(stable, 0, 0);
        mag_snapshot_free(snap);
        return NULL;
}

bool mag_snapshot_serialize(mag_snapshot_t *snap, const char *filename) {
    mag_snap_verify(filename && *filename, return false);
    const char *ext = strrchr(filename, '.'); /* check that the file extension is .mag */
    mag_snap_verify(ext != NULL && strcmp(ext, ".mag") == 0, return false);
    mag_snap_verify(snap->tensor_map.nitems <= UINT32_MAX, return false);
    mag_mem_stream_t stream = {0};
    mag_mapped_file_t map = {0};
    mag_snap_verify(mag_stream_mmap_file_w(&stream, &map, filename, mag_snaprage_compute_size(snap)), return false);
    mag_file_header_t header = (mag_file_header_t) {
        .magic = MAG_SNAP_FILE_MAGIC,
        .version = MAG_SNAPSHOT_VERSION,
        .timestamp = time(NULL),
        .checksum = 0,
        .aux = 0,
        .metadata_map_len = 0,
        .tensor_header_count = snap->tensor_map.nitems
    };
    mag_tensor_t **stable = NULL;
    size_t marker = 0;

    /* File header */
    marker = mag_stream_needle(&stream);
    uint8_t *u32_chk_patch_needle; /* Where to patch the checksum */
    mag_snap_verify(mag_file_header_serialize(&header, &stream, &u32_chk_patch_needle), goto error);
    const uint8_t *chk_start = u32_chk_patch_needle+sizeof(uint32_t); /* Checksum start region, excluding checksum field itself */
    mag_assert2(mag_stream_needle(&stream)-marker == MAG_FILE_HEADER_SIZE); /* Verify exact file header bytes written */

    /* String pool */
    marker = mag_stream_needle(&stream);
    mag_snap_verify(mag_stream_wu32_le(&stream, MAG_SNAP_SECTION_STR_POOL), goto error); /* Section marker */
    mag_snap_verify(mag_pool_serialize(&snap->str_pool, &stream), goto error);
    mag_assert2(mag_stream_needle(&stream)-marker == 4+mag_pool_compute_size(&snap->str_pool)); /* Verify exact section marker + pool bytes written */

    mag_snap_verify(mag_stream_wu32_le(&stream, MAG_SNAP_SECTION_META_DATA), goto error); /* TODO: Meta data marker */

    stable = (*mag_alloc)(NULL, snap->tensor_map.nitems*sizeof(*stable), 0);
    uint64_t offs = 0;
    size_t iter = 0, klen = 0; /* Write tensor headers */
    void *key = NULL, *val = NULL;
    size_t k;
    mag_snap_verify(mag_stream_wu32_le(&stream, MAG_SNAP_SECTION_TENSOR_DESC), goto error); /* Tensor desc marker */
    for (k=0; k < snap->tensor_map.nitems && (key = mag_map_next(&snap->tensor_map, &iter, &klen, &val)); ++k) {  /* Tensor descriptors */
        mag_assert2(klen == sizeof(uint32_t));
        uint32_t key_id = *(const uint32_t *)key;
        mag_tensor_t *tensor = val;
        mag_tensor_desc_t desc = {
            .rank = tensor->coords.rank,
            .dtype = tensor->dtype,
            .aux0 = 0,
            .aux1 = 0,
            .key_id = key_id,
            .numel = tensor->numel,
            .offset = offs,
            .shape = {}
        };
        mag_snap_verify(tensor->coords.rank >= 0 && tensor->coords.rank <= MAG_SNAP_MAX_RANK,  goto error);
        for (int64_t i=0; i < tensor->coords.rank; ++i) {
            mag_assert2(tensor->coords.shape[i] >= 0);
            desc.shape[i] = (uint64_t)tensor->coords.shape[i];
        }
        marker = mag_stream_needle(&stream);
        mag_snap_verify(mag_tensor_desc_serialize(&desc, &stream), goto error);
        mag_assert2(mag_stream_needle(&stream)-marker == MAG_TENSOR_DESC_SIZE(tensor->coords.rank));
        offs += mag_tensor_numbytes(tensor);
        stable[k] = tensor;
    }
    mag_assert2(k == snap->tensor_map.nitems);

    /* Compute checksum of metadata before data section starts */
    const uint8_t *chk_end = stream.pos;
    mag_device_t *dvc_interface;
    mag_backend_registry_get_by_device_id(snap->ctx->backend_registry, &dvc_interface, &MAG_DEVICE_ID_CPU);
    mag_assert2(dvc_interface);
    mag_cpu_device_t *dvc_impl = dvc_interface->impl;
    mag_assert2(dvc_impl);
    uint32_t (*vcrc32c)(const void *, size_t) = dvc_impl->kernels.crc32c; /* Get SIMD CRC32C from specializations */
    mag_assert2(vcrc32c);
    size_t chk_delta = chk_end-chk_start;
    mag_assert2(chk_delta > 0 && chk_end < stream.end);
    uint32_t crc32c = (*vcrc32c)((const void *)chk_start, chk_end-chk_start);
    #ifdef MAG_BIG_ENDIAN
        crc32c = mag_bswap32(crc32c);
    #endif
    memcpy(u32_chk_patch_needle, &crc32c, sizeof(crc32c));

    /* Tensor data section */
    marker = mag_stream_needle(&stream);
    mag_snap_verify(mag_stream_wu32_le(&stream, MAG_SNAP_SECTION_TENSOR_DATA), goto error); /* Data section marker */
    size_t nb_dat_total = 0;
    for (size_t i=0; i < snap->tensor_map.nitems; ++i) { /* Tensor data */
        mag_tensor_t *tensor = stable[i];
        mag_snap_verify(tensor->storage->device->id.type == MAG_BACKEND_TYPE_CPU, goto error); /* Tensor must live on CPU */
        mag_contiguous(&tensor, tensor); /* Make contiguous to allow the 1:1 copy into mmap destination region */
        size_t nb = mag_tensor_numbytes(tensor);
        nb_dat_total += nb;
        mag_snap_verify(mag_stream_wbytes(&stream, (const void *)mag_tensor_data_ptr(tensor), nb), mag_tensor_decref(tensor); goto error);
        mag_tensor_decref(tensor);
    }
    mag_assert2(mag_stream_needle(&stream)-marker == 4+nb_dat_total); /* Data section marker + total bytes */
    mag_assert2(mag_stream_needle(&stream) == stream.end-stream.base); /* All pre-estimated bytes must be written, down to the last crumb of cookie */
    mag_stream_close(&stream);
    mag_unmap_file(&map);
    (*mag_alloc)(stable, 0, 0);
    return true;
    error:
    mag_stream_close(&stream);
    mag_unmap_file(&map);
    if (stable) (*mag_alloc)(stable, 0, 0);
    return false;
}

mag_tensor_t *mag_snapshot_get_tensor(mag_snapshot_t *snap, const char *key) {
    mag_snap_verify(snap && key && *key, return NULL);
    uint32_t key_id = 0;
    if (mag_unlikely(!mag_pool_find_id(&snap->str_pool, (const uint8_t*)key, strlen(key), &key_id))) return NULL;
    mag_tensor_t *found = mag_map_lookup(&snap->tensor_map, &key_id, sizeof(key_id));
    if (found) mag_tensor_incref(found);
    return found;
}

bool mag_snapshot_put_tensor(mag_snapshot_t *snap, const char *key, mag_tensor_t *tensor) {
    mag_snap_verify(key && *key && tensor, return false;)
    uint32_t key_id = 0;
    if (mag_unlikely(!mag_pool_intern(&snap->str_pool, (const uint8_t *)key, strlen(key), &key_id))) return false;
    return mag_snapshot_insert_tensor_by_id(snap, key_id, tensor);
}

size_t mag_snapshot_get_num_tensors(mag_snapshot_t *snap) {
    return snap->tensor_map.nitems;
}

const char **mag_snapshot_get_tensor_keys(mag_snapshot_t *snap, size_t *out_num_keys) {
    if (!out_num_keys) return NULL;
    *out_num_keys = 0;
    if (!snap) return NULL;
    size_t n = snap->tensor_map.nitems;
    if (mag_unlikely(!n)) return NULL;
    char **keys = (*mag_alloc)(NULL, n*sizeof(*keys), 0);
    if (mag_unlikely(!keys)) return NULL;
    size_t iter = 0, klen = 0, idx = 0;
    void *keyp = NULL, *valp = NULL;
    while ((keyp = mag_map_next(&snap->tensor_map, &iter, &klen, &valp))) {
        if (mag_unlikely(klen != sizeof(uint32_t))) goto fail;
        const uint32_t key_id = *(const uint32_t *)keyp;
        if (mag_unlikely(key_id >= snap->str_pool.len)) goto fail;
        const mag_pool_record_t *rec = snap->str_pool.records + key_id;
        if (!rec->ptr || rec->len == 0) goto fail;
        char *name = (*mag_alloc)(NULL, (size_t)rec->len + 1, 0);
        if (mag_unlikely(!name)) goto fail;
        memcpy(name, rec->ptr, rec->len);
        name[rec->len] = '\0';
        keys[idx++] = name;
        if (mag_unlikely(idx > n)) goto fail;
    }
    if (mag_unlikely(idx != n)) goto fail;
    *out_num_keys = n;
    return (const char **)keys;
    fail:
        for (size_t i=0; i < idx; ++i)
            (*mag_alloc)(keys[i], 0, 0);
    (*mag_alloc)(keys, 0, 0);
    *out_num_keys = 0;
    return NULL;
}

void mag_snapshot_free_tensor_keys(const char **keys, size_t num_keys) {
    if (!keys) return;
    for (size_t i=0; i < num_keys; ++i)
        (*mag_alloc)((void *)keys[i], 0, 0);
    (*mag_alloc)((void *)keys, 0, 0);
}

MAG_COLDPROC void mag_snapshot_print_info(mag_snapshot_t *snap) {
    const mag_string_pool_t *pool = &snap->str_pool;
    printf("--- String Pool ---\n");
    printf("Entries: %zu\n", pool->len);
    double size = 0.0;
    const char *unit = "";
    mag_humanize_memory_size(mag_pool_compute_size((mag_string_pool_t *)pool), &size, &unit);
    printf("Size: %.03f%s\n", size, unit);
    for (size_t i = 0; i < pool->len; ++i) {
        const mag_pool_record_t *rec = pool->records+i;
        if (!rec->ptr || !rec->len) continue;
        printf("\t[%zu] Len: %u, Val: \"", i, rec->len);
        printf("%.*s", (int)rec->len, (const char *)rec->ptr);
        printf("\"\n");
    }
    printf("--- Tensors ---\n");
    printf("Entries: %zu\n", snap->tensor_map.nitems);
    size_t iter = 0, klen = 0;
    void *keyp = NULL, *valp = NULL;
    for (size_t slot=0; (keyp = mag_map_next(&snap->tensor_map, &iter, &klen, &valp)); ++slot) {
        mag_tensor_t *tensor = valp;
        uint32_t key_id = 0;
        if (klen == sizeof(uint32_t)) key_id = *(const uint32_t *)keyp;
        const char *name = NULL; /* NOT null terminated, must use length! */
        uint32_t name_len = 0;
        if (klen == sizeof(uint32_t) && key_id < pool->len) {
            const mag_pool_record_t *rec = pool->records + key_id;
            if (rec->ptr && rec->len) {
                name = (const char *)rec->ptr;
                name_len = rec->len;
            }
        }
        if (!name || !name_len) {
            name = "?";
            name_len = sizeof("?")-1;
        }
        char shape[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&shape, &tensor->coords.shape, tensor->coords.rank);
        mag_humanize_memory_size(mag_tensor_numbytes(tensor), &size, &unit);
        printf("\t[%zu] Name: \"%.*s\", Shape: %s, Type: %s, Size: %.01f%s\n", slot, (int)name_len, name, shape, mag_type_trait(tensor->dtype)->name, size, unit);
    }
    printf("--- Stats ---\n");
    mag_humanize_memory_size(snap->nb_meta, &size, &unit);
    printf("\tMetadata Size: %.03f%s (%.01f%%)\n", size, unit, snap->nb_meta ? 100.0*(double)snap->nb_meta / (double)snap->nb_total : 0.0);
    mag_humanize_memory_size(snap->nb_storage, &size, &unit);
    printf("\tStorage Size: %.03f%s (%.01f%%)\n", size, unit, snap->nb_total ? 100.0*(double)snap->nb_storage / (double)snap->nb_total : 0.0);
    mag_humanize_memory_size(snap->nb_total, &size, &unit);
    printf("\tTotal File Size: %.03f%s\n", size, unit);
    printf("-------------------\n");
}
