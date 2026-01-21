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

#define mag_storage_contract(expr, action) \
if (mag_unlikely(!(expr))) { \
    mag_log_error("Error reading/writing snapshot file: " #expr); \
    action; \
}

#define MAG_STORAGE_MAX_STRING_LENGTH 0xffff
#define mag_storage_magic(a,b,c,d) ((((d)&255)<<24)+(((c)&255)<<16)+(((b)&255)<<8)+((a)&255))
#define MAG_STO_FILE_MAGIC mag_storage_magic('M', 'A', 'G', '!')
#define MAG_STO_SECTION_STR_POOL mag_storage_magic('S', 'R', 'P', '!')
#define MAG_STO_SECTION_META_DATA mag_storage_magic('M', 'D', 'T', '!')
#define MAG_STO_SECTION_TENSOR_DESC mag_storage_magic('D', 'S', 'C', '!')
#define MAG_STO_SECTION_TENSOR_DATA mag_storage_magic('B', 'U', 'F', '!')
#define MAG_STO_SECTION_MARKERS_COUNT 4 /* File magic is not included, belongs to file header */
#define MAG_STO_MAX_RANK 64

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
    mag_storage_contract(path != NULL && *path, return false);
    mag_storage_contract(mag_map_file(&stream->file, path, 0, MAG_MAP_READ), return false);
    stream->base = stream->pos = stream->file.map;
    stream->end = stream->base + stream->file.fs;
    stream->flags |= MAG_MEM_STREAM_FLAGS_ISFILE;
    return true;
}

static bool mag_stream_mmap_file_w(mag_mem_stream_t *stream, const char *path, size_t size) {
    memset(stream, 0, sizeof(*stream));
    mag_storage_contract(path != NULL && *path, return false);
    mag_storage_contract(size > 0, return false);
    mag_storage_contract(mag_map_file(&stream->file, path, size, MAG_MAP_WRITE), return false);
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

static bool mag_stream_wu32_le(mag_mem_stream_t *stream, uint32_t val) {
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(val), return false);
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap32(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru32_le(mag_mem_stream_t *stream, uint32_t *val) {
    mag_storage_contract(val != NULL, return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(*val), return false);
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap32(*val);
    #endif
    return true;
}

static bool mag_stream_wu64_le(mag_mem_stream_t *stream, uint64_t val) {
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(val), return false);
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap64(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru64_le(mag_mem_stream_t *stream, uint64_t *val) {
    mag_storage_contract(val != NULL, return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(*val), return false);
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap64(*val);
    #endif
    return true;
}

static bool mag_stream_wstr(mag_mem_stream_t *stream, const uint8_t *str) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    size_t len = strlen((const char *)str);
    mag_storage_contract(len <= MAG_STORAGE_MAX_STRING_LENGTH && len <= UINT32_MAX, return false);
    mag_storage_contract(mag_utf8_validate(str, len), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, (uint32_t)len), return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len, return false);
    memcpy(stream->pos, str, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rstr(mag_mem_stream_t *stream, uint8_t **out_str) {
    uint32_t len = 0;
    mag_storage_contract(mag_stream_ru32_le(stream, &len), return false);
    mag_storage_contract(len <= MAG_STORAGE_MAX_STRING_LENGTH, return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len, return false);
    uint8_t *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(str, stream->pos, len);
    str[len] = '\0';
    mag_storage_contract(mag_utf8_validate(str, len), return false);
    stream->pos += len;
    *out_str = str;
    return true;
}

static bool mag_stream_wbuf(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_storage_contract(buf != NULL || len == 0, return false);
    mag_storage_contract(len <= UINT32_MAX, return false);
    mag_storage_contract(mag_stream_wu32_le(stream, (uint32_t)len), return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len, return false);
    if (len) {
        memcpy(stream->pos, buf, len);
        stream->pos += len;
    }
    return true;
}

static bool mag_stream_wbytes(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len, return false);
    if (len) memcpy(stream->pos, buf, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rbuf_alloc(mag_mem_stream_t *stream, uint8_t **out_buf, uint32_t *out_len) {
    mag_storage_contract(out_buf != NULL, return false);
    mag_storage_contract(out_len != NULL, return false);
    uint32_t len = 0;
    mag_storage_contract(mag_stream_ru32_le(stream, &len), return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len, return false);
    uint8_t *buf = NULL;
    if (len) {
        buf = (*mag_alloc)(NULL, (size_t)len, 0);
        mag_storage_contract(buf != NULL, return false);
        memcpy(buf, stream->pos, len);
    }
    stream->pos += len;
    *out_buf = buf;
    *out_len = len;
    return true;
}

static bool mag_stream_rbuf_view(mag_mem_stream_t *stream, const uint8_t **out_buf, uint32_t *out_len) {
    mag_storage_contract(out_buf != NULL, return false);
    mag_storage_contract(out_len != NULL, return false);
    uint32_t len = 0;
    mag_storage_contract(mag_stream_ru32_le(stream, &len), return false);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len, return false);
    const uint8_t *buf = stream->pos;
    stream->pos += len;
    *out_buf = buf;
    *out_len = len;
    return true;
}

/*
** Contains the file header structure.
** Not directly written to file due to possible packing issues
** De/serialization is done manually.
*/
typedef struct mag_file_header_state_t {
    uint32_t magic;
    uint32_t version;
    uint32_t checksum;
    uint32_t aux;
    uint32_t metadata_map_len;
    uint32_t tensor_header_count;
} mag_file_header_state_t;

#define MAG_FILE_HEADER_SIZE (4+4+4+4+4+4) /* We don't rely on struct packing */
mag_static_assert(sizeof(mag_file_header_state_t) % 4 == 0);
mag_static_assert(sizeof(mag_file_header_state_t) == MAG_FILE_HEADER_SIZE);

static bool mag_file_header_serialize(const mag_file_header_state_t *header, mag_mem_stream_t *stream) {
    mag_storage_contract(mag_stream_wu32_le(stream, header->magic), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, header->version), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, header->checksum), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, header->aux), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, header->metadata_map_len), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, header->tensor_header_count), return false);
    return true;
}

static bool mag_file_header_deserialize(mag_file_header_state_t *out_header, mag_mem_stream_t *stream) {
    mag_storage_contract(mag_stream_ru32_le(stream, &out_header->magic), return false);
    mag_storage_contract(mag_stream_ru32_le(stream, &out_header->version), return false);
    mag_storage_contract(mag_stream_ru32_le(stream, &out_header->checksum), return false);
    mag_storage_contract(mag_stream_ru32_le(stream, &out_header->aux), return false);
    mag_storage_contract(mag_stream_ru32_le(stream, &out_header->metadata_map_len), return false);
    mag_storage_contract(mag_stream_ru32_le(stream, &out_header->tensor_header_count), return false);
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
    mag_storage_contract(mag_stream_wu32_le(stream, mag_pack4xu8_le(desc->rank, desc->dtype, desc->aux0, desc->aux1)), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, desc->key_id), return false);
    mag_storage_contract(mag_stream_wu64_le(stream, desc->numel), return false);
    mag_storage_contract(mag_stream_wu64_le(stream, desc->offset), return false);
    for (int i=0; i < desc->rank; ++i)
        mag_storage_contract(mag_stream_wu64_le(stream, desc->shape[i]), return false);
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
    mag_map_init(&pool->map, 256, true);
}

static void mag_pool_free(mag_string_pool_t *pool) {
    mag_map_free(&pool->map);
    (*mag_alloc)(pool->records, 0, 0);
    memset(pool, 0, sizeof(*pool));
}

static bool mag_pool_intern(mag_string_pool_t *pool, const uint8_t *buf, size_t len, uint32_t *out_id) {
    mag_storage_contract(buf && len && len < UINT32_MAX, return false);
    mag_storage_contract(mag_utf8_validate(buf, len), return false);
    void *found = mag_map_lookup(&pool->map, buf, len);
    if (found) {
        *out_id = (uint32_t)(uintptr_t)found;
        return true;
    }
    mag_storage_contract(pool->len < UINT32_MAX, return false);
    *out_id = pool->len++;
    if (pool->len > pool->cap) {
        size_t cap = pool->cap ? pool->cap : 32;
        while (cap < pool->len) cap <<= 1;
        pool->records = (*mag_alloc)(pool->records, cap*sizeof(*pool->records), 0);
        pool->cap = cap;
    }
    mag_map_insert(&pool->map, buf, len, (void *)(uintptr_t)*out_id);
    const uint8_t *owned = mag_map_lookup_key_ptr(&pool->map, buf, len);
    mag_storage_contract(owned, return false);
    mag_pool_record_t *rec = pool->records+*out_id;
    rec->ptr = owned;
    rec->len = (uint32_t)len;
    return true;
}

static bool mag_pool_serialize(const mag_string_pool_t *pool, mag_mem_stream_t *stream) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE, return false);
    mag_storage_contract(pool && pool->len <= UINT32_MAX, return false);
    mag_storage_contract(mag_stream_wu32_le(stream, (uint32_t)pool->len), return false);
    mag_storage_contract(mag_stream_wu32_le(stream, 0), return false); /* offsets[0] = 0, for monotonically and clean O(1) offsets */
    uint32_t offs = 0;
    for (size_t i=0; i < pool->len; ++i) { /* Offset array */
        mag_pool_record_t *rec = pool->records+i;
        mag_storage_contract((rec->ptr || !rec->len) && rec->len <= UINT32_MAX, return false);
        mag_storage_contract(UINT32_MAX-offs >= rec->len, return false);
        offs += rec->len;
        mag_storage_contract(mag_stream_wu32_le(stream, offs), return false);
    }
    for (size_t i=0; i < pool->len; ++i) { /* String blob */
        mag_pool_record_t *rec = pool->records+i;
        mag_storage_contract(mag_stream_wbytes(stream, rec->ptr, rec->len), return false);
    }
    return true;
}

struct mag_snapshot_t {
    mag_context_t *ctx;
    mag_string_pool_t str_pool;
    mag_map_t tensor_map;
};

mag_snapshot_t *mag_snapshot_new(mag_context_t *ctx) {
    mag_snapshot_t *snap = (*mag_alloc)(NULL, sizeof(*snap), 0);
    memset(snap, 0, sizeof(*snap));
    snap->ctx = ctx;
    mag_pool_init(&snap->str_pool);
    mag_map_init(&snap->tensor_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, false);
    return snap;
}

static size_t mag_storage_estimate_size(mag_snapshot_t *snap) {
    size_t nb = 0;
    nb += MAG_FILE_HEADER_SIZE; /* File Header */
    nb += 4*MAG_STO_SECTION_MARKERS_COUNT; /* Markers */
    /* String pool */
    nb += sizeof(uint32_t); /* Count */
    nb += sizeof(uint32_t)*(snap->str_pool.len+1); /* Offsets */
    for (size_t i=0; i < snap->str_pool.len; ++i)
        nb += snap->str_pool.records[i].len; /* Bytes */
    size_t iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(&snap->tensor_map, &iter, &len, &val)) {  /* Tensors */
        mag_tensor_t *tensor = val;
        nb += MAG_TENSOR_DESC_SIZE(tensor->coords.rank);
        nb += mag_tensor_numbytes(tensor);
    }
    return nb;
}

bool mag_snapshot_save(mag_snapshot_t *snap, const char *filename) {
    mag_storage_contract(filename && *filename, return false);
    const char *ext = strrchr(filename, '.');     /* check that the file extension is .mag */
    mag_storage_contract(ext != NULL && strcmp(ext, ".mag") == 0, return false);
    mag_mem_stream_t stream;
    mag_storage_contract(mag_stream_mmap_file_w(&stream, filename, mag_storage_estimate_size(snap)), return false);
    mag_file_header_state_t header = (mag_file_header_state_t) {
        .magic = MAG_STO_FILE_MAGIC,
        .version = 1,
        .checksum = 0,
        .aux = 0,
        .metadata_map_len = 0,
        .tensor_header_count = snap->tensor_map.nitems
    };
    mag_tensor_t **stable = NULL;
    mag_storage_contract(mag_file_header_serialize(&header, &stream),  goto error); /* 1. File header */
    mag_storage_contract(mag_stream_wu32_le(&stream, MAG_STO_SECTION_STR_POOL),  goto error); /* 2. String pool marker */
    mag_storage_contract(mag_pool_serialize(&snap->str_pool, &stream),  goto error); /* 3. String pool */
    mag_storage_contract(mag_stream_wu32_le(&stream, MAG_STO_SECTION_META_DATA),  goto error); /* 4. Meta data marker */
    stable = (*mag_alloc)(NULL, snap->tensor_map.nitems*sizeof(*stable), 0);
    uint64_t offs = 0;
    size_t iter = 0, len = 0; /* Write tensor headers */
    void *val = NULL;
    size_t k;
    mag_storage_contract(mag_stream_wu32_le(&stream, MAG_STO_SECTION_TENSOR_DESC),goto error); /* 5. Tensor desc marker */
    for (k=0; k < snap->tensor_map.nitems && mag_map_next(&snap->tensor_map, &iter, &len, &val); ++k) {  /* 6. Tensor descriptors */
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
        mag_storage_contract(tensor->coords.rank >= 0 && tensor->coords.rank <= MAG_STO_MAX_RANK,  goto error);
        for (int64_t i=0; i < tensor->coords.rank; ++i) {
            mag_assert2(tensor->coords.shape[i] >= 0);
            desc.shape[i] = (uint64_t)tensor->coords.shape[i];
        }
        mag_storage_contract(mag_tensor_desc_serialize(&desc, &stream), goto error);
        offs += mag_tensor_numbytes(tensor);
        stable[k] = tensor;
    }
    mag_assert2(k == snap->tensor_map.nitems);
    mag_storage_contract(mag_stream_wu32_le(&stream, MAG_STO_SECTION_TENSOR_DATA), goto error); /* 7. Data section marker */
    for (size_t i=0; i < snap->tensor_map.nitems; ++i) { /* Tensor data */
        mag_tensor_t *tensor = stable[i];
        mag_storage_contract(mag_device_is(tensor->storage->device, "cpu"), goto error); /* 8. Tensor must live on CPU */
        mag_storage_contract(mag_stream_wbytes(&stream, (const void *)mag_tensor_data_ptr(tensor), mag_tensor_numbytes(tensor)), goto error);
    }
    mag_assert2((size_t)(stream.pos - stream.base) <= (size_t)(stream.end - stream.base));
    mag_stream_close(&stream);
    (*mag_alloc)(stable, 0, 0);
    return true;
    error:
    mag_stream_close(&stream);
    if (stable) (*mag_alloc)(stable, 0, 0);
    return false;
}

mag_tensor_t *mag_snapshot_get_tensor(mag_snapshot_t *snap, const char *key) {
    mag_storage_contract(snap && key && *key, return false;)
    uint32_t key_id = 0;
    mag_storage_contract(mag_pool_intern(&snap->str_pool, (const uint8_t *)key, strlen(key), &key_id), return false;)
    return mag_map_lookup(&snap->tensor_map, &key_id, sizeof(key_id));
}

bool mag_snapshot_put_tensor(mag_snapshot_t *snap, const char *key, mag_tensor_t *tensor) {
    mag_storage_contract(key && *key && tensor, return false;)
    uint32_t key_id = 0;
    mag_storage_contract(mag_pool_intern(&snap->str_pool, (const uint8_t *)key, strlen(key), &key_id), return false;)
    const mag_pool_record_t *rec = snap->str_pool.records+key_id;
    mag_map_insert(&snap->tensor_map, rec->ptr, rec->len, tensor);
    mag_tensor_incref(tensor);
    return true;
}

void mag_snapshot_free(mag_snapshot_t *snap) {
    mag_pool_free(&snap->str_pool);
    size_t iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(&snap->tensor_map, &iter, &len, &val)) /* Free cloned metadata records */
        mag_tensor_decref(val);
    mag_map_free(&snap->tensor_map);
    memset(snap, 0, sizeof(*snap));
    (*mag_alloc)(snap, 0, 0);
}
