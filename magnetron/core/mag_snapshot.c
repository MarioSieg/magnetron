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

#include "mag_snapshot.h"
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
** +----------------------+
** | File Header          |
** +----------------------+
** | String Pool          |
** +----------------------+
** | Metadata Map         |
** +----------------------+
** | Tensor Header Map    |
** +----------------------+
** | Tensor Data          |
** +----------------------+
**
** ================= File Header =================
** +----------------------+
** | Magic : u32          |
** +----------------------+
** | Version : u32        |
** +----------------------+
** | Checksum: u32        | <- CRC32 Castagnoli checksum includes all following metadata (expect tensor data section), expect the prevous fields.
** +----------------------+ <- From now on all metadata until the tensor data section is checksummed.
** | Aux: u32             | <- Reserved for future use.
** +----------------------+
** | String Pool Len : u32|
** +----------------------+
** | Metadata Map Len: u32|
** +----------------------+
** | Tensor Headers: u32  |
** +----------------------+
**
*/

#define mag_sto_verify(expr, action) \
if (mag_unlikely(!(expr))) { \
    mag_log_error("Error reading/writing snapshot file: " #expr); \
    action; \
}

#define MAG_STO_MAX_STRLEN 0xffff
#define MAG_STO_MAX_RANK 64
#define MAG_STO_MAX_STR_POOL_BLOB_SIZE (128ull<<20) /* 128 MiB */
#define MAG_STO_MAX_OFFSETS 0xffff

#define mag_sto_pack4_ne(a,b,c,d) ((((d)&255)<<24)+(((c)&255)<<16)+(((b)&255)<<8)+((a)&255))
#define MAG_STO_FILE_MAGIC mag_sto_pack4_ne('M', 'A', 'G', '!')
#define MAG_STO_SECTION_STR_POOL mag_sto_pack4_ne('S', 'R', 'P', '!')
#define MAG_STO_SECTION_META_DATA mag_sto_pack4_ne('M', 'D', 'T', '!')
#define MAG_STO_SECTION_TENSOR_DESC mag_sto_pack4_ne('D', 'S', 'C', '!')
#define MAG_STO_SECTION_TENSOR_DATA mag_sto_pack4_ne('B', 'U', 'F', '!')
#define MAG_STO_SECTION_MARKERS_COUNT 4 /* File magic is not included, belongs to file header */

#ifdef MAG_BIG_ENDIAN
/*
** If some annoying host is BE, support could be added by byte-wapping tensor buffer elements with COW mmap.
** Not yet done at the moment.
** Only the data section requires handling for BE, the headers and metadata already do endinaess swapping
*/
#error "Big endian is not supported at the moment"
#endif

typedef enum mag_mem_stream_flags_t {
    MAG_MEM_STREAM_FLAGS_NONE = 0,
    MAG_MEM_STREAM_FLAGS_ISFILE = 1<<0,
    MAG_MEM_STREAM_FLAGS_WRITE = 1<<1
} mag_mem_stream_flags_t;

typedef struct mag_mem_stream_t {
    uint8_t *base;
    uint8_t *pos;
    uint8_t *end;
    mag_mem_stream_flags_t flags;
    mag_mapped_file_t file;
} mag_mem_stream_t;

static bool mag_stream_mmap_file_r(mag_mem_stream_t *stream, const char *path) {
    memset(stream, 0, sizeof(*stream));
    mag_sto_verify(path != NULL && *path, return false);
    mag_sto_verify(mag_map_file(&stream->file, path, 0, MAG_MAP_READ), return false);
    stream->base = stream->pos = stream->file.map;
    stream->end = stream->base + stream->file.fs;
    stream->flags |= MAG_MEM_STREAM_FLAGS_ISFILE;
    return true;
}

static bool mag_stream_mmap_file_w(mag_mem_stream_t *stream, const char *path, size_t size) {
    memset(stream, 0, sizeof(*stream));
    mag_sto_verify(path != NULL && *path, return false);
    mag_sto_verify(size > 0, return false);
    mag_sto_verify(mag_map_file(&stream->file, path, size, MAG_MAP_WRITE), return false);
    stream->base = stream->pos = stream->file.map;
    stream->end = stream->base + stream->file.fs;
    stream->flags |= MAG_MEM_STREAM_FLAGS_ISFILE|MAG_MEM_STREAM_FLAGS_WRITE;
    return true;
}

static void mag_stream_close(mag_mem_stream_t *stream) {
    if (!stream) return;
    if (stream->flags & MAG_MEM_STREAM_FLAGS_ISFILE)
        mag_unmap_file(&stream->file);
    memset(stream, 0, sizeof(*stream));
}

static size_t mag_stream_needle(const mag_mem_stream_t *stream) { return (size_t)(stream->pos - stream->base); }
static size_t mag_stream_remaining(const mag_mem_stream_t *stream) { return (size_t)(stream->end - stream->pos); }

static bool mag_stream_wu32_le(mag_mem_stream_t *stream, uint32_t val) {
    mag_sto_verify((size_t)(stream->end - stream->pos) >= sizeof(val), return false);
    mag_sto_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap32(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru32_le(mag_mem_stream_t *stream, uint32_t *val) {
    mag_sto_verify(val != NULL, return false);
    mag_sto_verify((size_t)(stream->end - stream->pos) >= sizeof(*val), return false);
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap32(*val);
    #endif
    return true;
}

static bool mag_stream_wu64_le(mag_mem_stream_t *stream, uint64_t val) {
    mag_sto_verify((size_t)(stream->end - stream->pos) >= sizeof(val), return false);
    mag_sto_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap64(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru64_le(mag_mem_stream_t *stream, uint64_t *val) {
    mag_sto_verify(val != NULL, return false);
    mag_sto_verify((size_t)(stream->end - stream->pos) >= sizeof(*val), return false);
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap64(*val);
    #endif
    return true;
}

static bool mag_stream_wstr(mag_mem_stream_t *stream, const uint8_t *str) {
    mag_sto_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    size_t len = strlen((const char *)str);
    mag_sto_verify(len <= MAG_STO_MAX_STRLEN && len <= UINT32_MAX, return false);
    mag_sto_verify(mag_utf8_validate(str, len), return false);
    mag_sto_verify(mag_stream_wu32_le(stream, (uint32_t)len), return false);
    mag_sto_verify((size_t)(stream->end - stream->pos) >= len, return false);
    memcpy(stream->pos, str, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rstr(mag_mem_stream_t *stream, uint8_t **out_str) {
    uint32_t len = 0;
    mag_sto_verify(mag_stream_ru32_le(stream, &len), return false);
    mag_sto_verify(len <= MAG_STO_MAX_STRLEN, return false);
    mag_sto_verify((size_t)(stream->end - stream->pos) >= len, return false);
    uint8_t *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(str, stream->pos, len);
    str[len] = '\0';
    mag_sto_verify(mag_utf8_validate(str, len), return false);
    stream->pos += len;
    *out_str = str;
    return true;
}

static bool mag_stream_wbuf(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_sto_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_sto_verify(buf != NULL || len == 0, return false);
    mag_sto_verify(len <= UINT32_MAX, return false);
    mag_sto_verify(mag_stream_wu32_le(stream, (uint32_t)len), return false);
    mag_sto_verify((size_t)(stream->end - stream->pos) >= len, return false);
    if (len) {
        memcpy(stream->pos, buf, len);
        stream->pos += len;
    }
    return true;
}

static bool mag_stream_wbytes(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_sto_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_sto_verify((size_t)(stream->end - stream->pos) >= len, return false);
    if (len) memcpy(stream->pos, buf, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rbytes_view(mag_mem_stream_t *s, const uint8_t **out, size_t len) {
    mag_sto_verify(out != NULL, return false);
    mag_sto_verify((size_t)(s->end - s->pos) >= len, return false);
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
    mag_sto_verify(header->magic == MAG_STO_FILE_MAGIC, return false);
    mag_sto_verify(mag_stream_wu32_le(stream, header->magic), return false);
    mag_sto_verify(header->version == MAG_SNAPSHOT_VERSION, return false); /* Reading older versions is supporting, writing is not */
    mag_sto_verify(mag_stream_wu32_le(stream, header->version), return false);
    mag_sto_verify(mag_stream_wu64_le(stream, header->timestamp), return false);
    *u32_chk_patch_needle = stream->pos; /* Needle where the checksum is overwritten later */
    mag_sto_verify(mag_stream_wu32_le(stream, header->checksum), return false);
    mag_sto_verify(mag_stream_wu32_le(stream, header->aux), return false);
    mag_sto_verify(mag_stream_wu32_le(stream, header->metadata_map_len), return false);
    mag_sto_verify(mag_stream_wu32_le(stream, header->tensor_header_count), return false);
    return true;
}

static bool mag_file_header_deserialize(mag_file_header_t *header, mag_mem_stream_t *stream) {
    mag_sto_verify(mag_stream_ru32_le(stream, &header->magic), return false);
    mag_sto_verify(header->magic == MAG_STO_FILE_MAGIC, return false);
    mag_sto_verify(mag_stream_ru32_le(stream, &header->version), return false);
    mag_sto_verify(header->version <= MAG_SNAPSHOT_VERSION, return false);
    mag_sto_verify(mag_stream_ru64_le(stream, &header->timestamp), return false);
    mag_sto_verify(mag_stream_ru32_le(stream, &header->checksum), return false);
    mag_sto_verify(mag_stream_ru32_le(stream, &header->aux), return false);
    mag_sto_verify(mag_stream_ru32_le(stream, &header->metadata_map_len), return false);
    mag_sto_verify(mag_stream_ru32_le(stream, &header->tensor_header_count), return false);
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
    uint8_t rank; /* 0..MAG_STO_MAX_RANK */
    mag_dtype_t dtype;
    uint8_t aux0;
    uint8_t aux1;
    uint32_t key_id;
    uint64_t numel;
    uint64_t offset;
    uint64_t shape[MAG_STO_MAX_RANK];
} mag_tensor_desc_t;
#define MAG_TENSOR_DESC_SIZE(rank) (4+4+8+8 + 8*(rank))

static bool mag_tensor_desc_serialize(const mag_tensor_desc_t *desc, mag_mem_stream_t *stream) {
    mag_sto_verify(mag_stream_wu32_le(stream, mag_pack4xu8_le(desc->rank, desc->dtype, desc->aux0, desc->aux1)), return false);
    mag_sto_verify(mag_stream_wu32_le(stream, desc->key_id), return false);
    mag_sto_verify(mag_stream_wu64_le(stream, desc->numel), return false);
    mag_sto_verify(mag_stream_wu64_le(stream, desc->offset), return false);
    for (uint8_t i=0; i < desc->rank; ++i)
        mag_sto_verify(mag_stream_wu64_le(stream, desc->shape[i]), return false);
    return true;
}

static bool mag_tensor_desc_deserialize(mag_tensor_desc_t *desc, mag_mem_stream_t *stream) {
    uint32_t packed = 0;
    mag_sto_verify(mag_stream_ru32_le(stream, &packed), return false);
    uint8_t dtype;
    mag_unpack4xu8_le(packed, &desc->rank, &dtype, &desc->aux0, &desc->aux1);
    mag_sto_verify(desc->rank < MAG_STO_MAX_RANK, return false);
    mag_sto_verify(dtype < MAG_DTYPE__NUM, return false);
    desc->dtype = dtype;
    mag_sto_verify(mag_stream_ru32_le(stream, &desc->key_id), return false);
    mag_sto_verify(desc->key_id != 0, return false); /* TODO: key id */
    mag_sto_verify(mag_stream_ru64_le(stream, &desc->numel), return false);
    mag_sto_verify(desc->numel > 0 && desc->numel <= INT64_MAX, return false);
    mag_sto_verify(mag_stream_ru64_le(stream, &desc->offset), return false);     /* TODO: verify offset */
    int64_t numel_prod = 1;
    for (uint8_t i=0; i < desc->rank; ++i) {
        uint64_t dim=0;
        mag_sto_verify(mag_stream_ru64_le(stream, &dim), return false);
        mag_sto_verify(dim <= INT64_MAX, return false);
        mag_sto_verify(!mag_mulov64(dim, numel_prod, &numel_prod), return false);
        desc->shape[i] = dim;
    }
    mag_sto_verify(numel_prod <= INT64_MAX && numel_prod == desc->numel, return false);
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
    mag_sto_verify(buf && len && len < UINT32_MAX, return false);
    mag_sto_verify(mag_utf8_validate(buf, len), return false);
    void *found = mag_map_lookup(&pool->map, buf, len);
    if (found) {
        *out_id = (uint32_t)(uintptr_t)found;
        return true;
    }
    mag_sto_verify(pool->len < UINT32_MAX, return false);
    *out_id = pool->len++;
    if (pool->len > pool->cap) {
        size_t cap = pool->cap ? pool->cap : 32;
        while (cap < pool->len) cap <<= 1;
        pool->records = (*mag_alloc)(pool->records, cap*sizeof(*pool->records), 0);
        pool->cap = cap;
    }
    mag_map_insert(&pool->map, buf, len, (void *)(uintptr_t)*out_id);
    const uint8_t *owned = mag_map_lookup_key_ptr(&pool->map, buf, len);
    mag_sto_verify(owned, return false);
    mag_pool_record_t *rec = pool->records+*out_id;
    rec->ptr = owned;
    rec->len = (uint32_t)len;
    return true;
}

static bool mag_pool_serialize(const mag_string_pool_t *pool, mag_mem_stream_t *stream) {
    mag_sto_verify(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_sto_verify(pool && pool->len <= UINT32_MAX, return false);
    mag_sto_verify(mag_stream_wu32_le(stream, (uint32_t)pool->len), return false);
    mag_sto_verify(mag_stream_wu32_le(stream, 0), return false); /* offsets[0] = 0, for monotonically and clean O(1) offsets */
    uint32_t offs = 0;
    for (size_t i=0; i < pool->len; ++i) { /* Offset array */
        mag_pool_record_t *rec = pool->records+i;
        mag_sto_verify((rec->ptr || !rec->len) && rec->len <= UINT32_MAX, return false);
        mag_sto_verify(UINT32_MAX-offs >= rec->len, return false);
        offs += rec->len;
        mag_sto_verify(mag_stream_wu32_le(stream, offs), return false);
    }
    for (size_t i=0; i < pool->len; ++i) { /* String blob */
        mag_pool_record_t *rec = pool->records+i;
        mag_sto_verify(mag_stream_wbytes(stream, rec->ptr, rec->len), return false);
    }
    return true;
}

static bool mag_pool_deserialize(mag_string_pool_t *pool, mag_mem_stream_t *stream) {
    mag_pool_free(pool);
    mag_pool_init(pool);
    mag_assert2(pool->len == 0); /*Pool must be fresh */
    uint32_t count = 0;
    mag_sto_verify(mag_stream_ru32_le(stream, &count), return false);
    size_t num_offsets = (size_t)count+1;
    mag_sto_verify(num_offsets <= MAG_STO_MAX_OFFSETS, return false);
    uint32_t *offs = (*mag_alloc)(NULL, num_offsets*sizeof(*offs), 0);
    for (size_t i=0; i < num_offsets; ++i) /* Read in offsets */
        mag_sto_verify(mag_stream_ru32_le(stream, offs+i), goto fail);
    mag_sto_verify(*offs == 0, goto fail);
    for (size_t i=1; i < num_offsets; ++i)
        mag_sto_verify(offs[i] >= offs[i-1], goto fail); /* Monotonic verify */
    uint32_t blob_size = offs[count];
    mag_sto_verify(blob_size <= MAG_STO_MAX_STR_POOL_BLOB_SIZE, goto fail);
    const uint8_t *blob = NULL;
    mag_sto_verify(mag_stream_rbytes_view(stream, &blob, blob_size), goto fail);
    for (uint32_t id=0; id < count; ++id) {
        uint32_t a = offs[id];
        uint32_t b = offs[id+1];
        mag_sto_verify(a <= b && b <= blob_size, goto fail);
        const uint8_t *str = blob+a;
        uint32_t delta = b-a;
        mag_sto_verify(delta, goto fail);
        mag_sto_verify(mag_utf8_validate(str, delta), goto fail);
        uint32_t len = 0;
        mag_sto_verify(mag_pool_intern(pool, str, delta, &len), goto fail);
        mag_sto_verify(len == id, goto fail);
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

struct mag_snapshot_t {
    mag_context_t *ctx;
    mag_string_pool_t str_pool;
    mag_map_t tensor_map;
    mag_mem_stream_t stream;
    bool owns_stream; /* (Reading mode) If true, memory from within the stream is referenced and must kept alive until snapshot_free. */
};

static size_t mag_storage_compute_tensor_sizes(mag_map_t *tmap) {
    size_t nb = 0, iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(tmap, &iter, &len, &val)) {  /* Tensors */
        mag_tensor_t *tensor = val;
        nb += MAG_TENSOR_DESC_SIZE(tensor->coords.rank);
        nb += mag_tensor_numbytes(tensor);
    }
    return nb;
}

static size_t mag_storage_compute_size(mag_snapshot_t *snap) {
    size_t nb = 0;
    nb += MAG_FILE_HEADER_SIZE; /* File Header */
    nb += 4*MAG_STO_SECTION_MARKERS_COUNT; /* Markers */
    nb += mag_pool_compute_size(&snap->str_pool);
    nb += mag_storage_compute_tensor_sizes(&snap->tensor_map);
    return nb;
}

mag_snapshot_t *mag_snapshot_new(mag_context_t *ctx) {
    mag_snapshot_t *snap = (*mag_alloc)(NULL, sizeof(*snap), 0);
    memset(snap, 0, sizeof(*snap));
    snap->ctx = ctx;
    mag_pool_init(&snap->str_pool);
    mag_map_init(&snap->tensor_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, false);
    return snap;
}

void mag_snapshot_free(mag_snapshot_t *snap) {
    mag_pool_free(&snap->str_pool);
    size_t iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(&snap->tensor_map, &iter, &len, &val)) /* Free cloned metadata records */
        mag_tensor_decref(val);
    if (snap->owns_stream) /* If we own the stream, close it. TODO: what if tensors are still alive which reference the mem?! */
        mag_stream_close(&snap->stream);
    mag_map_free(&snap->tensor_map);
    memset(snap, 0, sizeof(*snap));
    (*mag_alloc)(snap, 0, 0);
}

mag_snapshot_t *mag_snapshot_deserialize(mag_context_t *ctx, const char *filename) {
    mag_sto_verify(filename && *filename, return false);
    const char *ext = strrchr(filename, '.'); /* check that the file extension is .mag */
    mag_sto_verify(ext != NULL && strcmp(ext, ".mag") == 0, return false);
    mag_snapshot_t *snap = mag_snapshot_new(ctx);
    mag_mem_stream_t *stream = &snap->stream;
    mag_sto_verify(mag_stream_mmap_file_r(stream, filename), return false);
    snap->owns_stream = true; /* We need to free the stream later as we reference memory from it now */
    mag_sto_verify(mag_stream_remaining(stream) >= MAG_FILE_HEADER_SIZE + 4*MAG_STO_SECTION_MARKERS_COUNT, goto error); /* We must at minimum have enough bytes for an empty file */

    size_t marker = mag_stream_needle(stream);
    /* File header */
    mag_file_header_t header = {0};
    mag_sto_verify(mag_file_header_deserialize(&header, stream), goto error)
    mag_assert2(mag_stream_needle(stream)-marker == MAG_FILE_HEADER_SIZE); /* Verify exact file header bytes written */

    /* String pool */
    marker = mag_stream_needle(stream);
    uint32_t section_marker = 0;
    mag_sto_verify(mag_stream_ru32_le(stream, &section_marker), goto error);
    mag_sto_verify(section_marker == MAG_STO_SECTION_STR_POOL, goto error);
    mag_sto_verify(mag_pool_deserialize(&snap->str_pool, stream), goto error);
    mag_assert2(mag_stream_needle(stream)-marker == 4+mag_pool_compute_size(&snap->str_pool)); /* Verify exact section marker + pool bytes written */

    /*uint64_t offs=0;*/
    for (uint32_t i=0; i < header.tensor_header_count; ++i) {
        mag_tensor_desc_t desc = {0};
        mag_sto_verify(mag_tensor_desc_deserialize(&desc, stream), goto error);
        mag_tensor_t *tensor = NULL;
        int64_t shape[MAG_STO_MAX_RANK];
        for (uint8_t j=0; j < desc.rank && j < sizeof(shape)/sizeof(*shape); ++j)
            shape[j] = (int64_t)desc.shape[j];
        mag_sto_verify(mag_empty(&tensor, ctx, desc.dtype, desc.rank, shape), goto error); /*  */
        /*mag_snapshot_put_tensor() TODO: lookup key and insert tensor */
    }

    /* Read data */

    return snap;
    error:
        mag_snapshot_free(snap);
        return NULL;
}

bool mag_snapshot_serialize(mag_snapshot_t *snap, const char *filename) {
    mag_sto_verify(filename && *filename, return false);
    const char *ext = strrchr(filename, '.'); /* check that the file extension is .mag */
    mag_sto_verify(ext != NULL && strcmp(ext, ".mag") == 0, return false);
    mag_sto_verify(snap->tensor_map.nitems <= UINT32_MAX, return false);
    mag_mem_stream_t stream;
    mag_sto_verify(mag_stream_mmap_file_w(&stream, filename, mag_storage_compute_size(snap)), return false);
    mag_file_header_t header = (mag_file_header_t) {
        .magic = MAG_STO_FILE_MAGIC,
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
    mag_sto_verify(mag_file_header_serialize(&header, &stream, &u32_chk_patch_needle), goto error);
    const uint8_t *chk_start = u32_chk_patch_needle+sizeof(uint32_t); /* Checksum start region, excluding checksum field itself */
    mag_assert2(mag_stream_needle(&stream)-marker == MAG_FILE_HEADER_SIZE); /* Verify exact file header bytes written */

    /* String pool */
    marker = mag_stream_needle(&stream);
    mag_sto_verify(mag_stream_wu32_le(&stream, MAG_STO_SECTION_STR_POOL), goto error); /* Section marker */
    mag_sto_verify(mag_pool_serialize(&snap->str_pool, &stream), goto error);
    mag_assert2(mag_stream_needle(&stream)-marker == 4+mag_pool_compute_size(&snap->str_pool)); /* Verify exact section marker + pool bytes written */

    mag_sto_verify(mag_stream_wu32_le(&stream, MAG_STO_SECTION_META_DATA), goto error); /* TODO: Meta data marker */

    stable = (*mag_alloc)(NULL, snap->tensor_map.nitems*sizeof(*stable), 0);
    uint64_t offs = 0;
    size_t iter = 0, len = 0; /* Write tensor headers */
    void *val = NULL;
    size_t k;
    mag_sto_verify(mag_stream_wu32_le(&stream, MAG_STO_SECTION_TENSOR_DESC),goto error); /* Tensor desc marker */
    for (k=0; k < snap->tensor_map.nitems && mag_map_next(&snap->tensor_map, &iter, &len, &val); ++k) {  /* Tensor descriptors */
        mag_tensor_t *tensor = val;
        mag_tensor_desc_t desc = {
            .rank = tensor->coords.rank,
            .dtype = tensor->dtype,
            .aux0 = 0,
            .aux1 = 0,
            .key_id = 0, /* TODO */
            .numel = tensor->numel,
            .offset = offs,
            .shape = {}
        };
        mag_sto_verify(tensor->coords.rank >= 0 && tensor->coords.rank <= MAG_STO_MAX_RANK,  goto error);
        for (int64_t i=0; i < tensor->coords.rank; ++i) {
            mag_assert2(tensor->coords.shape[i] >= 0);
            desc.shape[i] = (uint64_t)tensor->coords.shape[i];
        }
        marker = mag_stream_needle(&stream);
        mag_sto_verify(mag_tensor_desc_serialize(&desc, &stream), goto error);
        mag_assert2(mag_stream_needle(&stream)-marker == MAG_TENSOR_DESC_SIZE(tensor->coords.rank));
        offs += mag_tensor_numbytes(tensor);
        stable[k] = tensor;
    }
    mag_assert2(k == snap->tensor_map.nitems);

    /* Compute checksum of metadata before data section starts */
    const uint8_t *chk_end = stream.pos;
    mag_device_t *dvc_interface;
    mag_backend_registry_get_by_device_id(snap->ctx->backend_registry, &dvc_interface, "cpu:0");
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
    mag_sto_verify(mag_stream_wu32_le(&stream, MAG_STO_SECTION_TENSOR_DATA), goto error); /* Data section marker */
    size_t nb_dat_total = 0;
    for (size_t i=0; i < snap->tensor_map.nitems; ++i) { /* Tensor data */
        mag_tensor_t *tensor = stable[i];
        mag_sto_verify(mag_device_is(tensor->storage->device, "cpu"), goto error); /* Tensor must live on CPU */
        mag_contiguous(&tensor, tensor); /* Make contiguous to allow the 1:1 copy into mmap destination region */
        size_t nb = mag_tensor_numbytes(tensor);
        nb_dat_total += nb;
        mag_sto_verify(mag_stream_wbytes(&stream, (const void *)mag_tensor_data_ptr(tensor), nb), mag_tensor_decref(tensor); goto error);
        mag_tensor_decref(tensor);
    }
    mag_assert2(mag_stream_needle(&stream)-marker == 4+nb_dat_total); /* Data section marker + total bytes */
    mag_assert2(mag_stream_needle(&stream) == stream.end-stream.base); /* All pre-estimated bytes must be written, down to the last crumb of cookie */
    mag_stream_close(&stream);
    (*mag_alloc)(stable, 0, 0);
    return true;
    error:
    mag_stream_close(&stream);
    if (stable) (*mag_alloc)(stable, 0, 0);
    return false;
}

mag_tensor_t *mag_snapshot_get_tensor(mag_snapshot_t *snap, const char *key) {
    mag_sto_verify(snap && key && *key, return false;)
    uint32_t key_id = 0;
    mag_sto_verify(mag_pool_intern(&snap->str_pool, (const uint8_t *)key, strlen(key), &key_id), return false;)
    return mag_map_lookup(&snap->tensor_map, &key_id, sizeof(key_id));
}

bool mag_snapshot_put_tensor(mag_snapshot_t *snap, const char *key, mag_tensor_t *tensor) {
    mag_sto_verify(key && *key && tensor, return false;)
    uint32_t key_id = 0;
    mag_sto_verify(mag_pool_intern(&snap->str_pool, (const uint8_t *)key, strlen(key), &key_id), return false;)
    const mag_pool_record_t *rec = snap->str_pool.records+key_id;
    mag_map_insert(&snap->tensor_map, rec->ptr, rec->len, tensor);
    mag_tensor_incref(tensor);
    return true;
}

MAG_COLDPROC void mag_snapshot_print_info(mag_snapshot_t *snap) {
    const mag_string_pool_t *pool = &snap->str_pool;
    printf("--- String Pool ---\n");
    printf("count: %zu\n", pool->len);
    printf("serialized_bytes: %zu\n", mag_pool_compute_size((mag_string_pool_t *)pool));
    for (size_t i = 0; i < pool->len; ++i) {
        const mag_pool_record_t *rec = pool->records+i;
        if (!rec->ptr || !rec->len) continue;
        printf("[%zu] len=%u  \"", i, rec->len);
        printf("%.*s", (int)rec->len, (const char *)rec->ptr);
        printf("\"\n");
    }
    printf("-------------------\n");
}
