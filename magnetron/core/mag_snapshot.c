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

#define MAG_STORAGE_MAX_STRING_LENGTH 0xffff
#define mag_storage_contract(expr) do { if (mag_unlikely(!(expr))) return false; } while (0)
#define mag_storage_contract_do(expr, action) do { if (mag_unlikely(!(expr))) { action; } } while (0)
#define mag_sto_magic4(a,b,c,d) ((((d)&255)<<24) + (((c)&255)<<16) + (((b)&255)<<8) + ((a)&255))
#define MAG_STO_FILE_MAGIC mag_sto_magic4('M', 'A', 'G', '!')

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
    mag_storage_contract(path != NULL && *path);
    mag_storage_contract(mag_map_file(&stream->file, path, 0, MAG_MAP_READ));
    stream->base = stream->pos = stream->file.map;
    stream->end = stream->base + stream->file.fs;
    stream->flags |= MAG_MEM_STREAM_FLAGS_ISFILE;
    return true;
}

static bool mag_stream_mmap_file_w(mag_mem_stream_t *stream, const char *path, size_t size) {
    memset(stream, 0, sizeof(*stream));
    mag_storage_contract(path != NULL && *path);
    mag_storage_contract(size > 0);
    mag_storage_contract(mag_map_file(&stream->file, path, size, MAG_MAP_WRITE));
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
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(val));
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap32(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru32_le(mag_mem_stream_t *stream, uint32_t *val) {
    mag_storage_contract(val != NULL);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(*val));
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap32(*val);
    #endif
    return true;
}

static bool mag_stream_wu64_le(mag_mem_stream_t *stream, uint64_t val) {
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(val));
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    #ifdef MAG_BIG_ENDIAN
        val = mag_bswap64(val);
    #endif
    memcpy(stream->pos, &val, sizeof(val));
    stream->pos += sizeof(val);
    return true;
}

static bool mag_stream_ru64_le(mag_mem_stream_t *stream, uint64_t *val) {
    mag_storage_contract(val != NULL);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= sizeof(*val));
    memcpy(val, stream->pos, sizeof(*val));
    stream->pos += sizeof(*val);
    #ifdef MAG_BIG_ENDIAN
        *val = mag_bswap64(*val);
    #endif
    return true;
}

static bool mag_stream_wstr(mag_mem_stream_t *stream, const uint8_t *str) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    size_t len = strlen((const char *)str);
    mag_storage_contract(len <= MAG_STORAGE_MAX_STRING_LENGTH && len <= UINT32_MAX);
    mag_storage_contract(mag_utf8_validate(str, len));
    mag_storage_contract(mag_stream_wu32_le(stream, (uint32_t)len));
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len);
    memcpy(stream->pos, str, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rstr(mag_mem_stream_t *stream, uint8_t **out_str) {
    uint32_t len = 0;
    mag_storage_contract(mag_stream_ru32_le(stream, &len));
    mag_storage_contract(len <= MAG_STORAGE_MAX_STRING_LENGTH);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len);
    uint8_t *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(str, stream->pos, len);
    str[len] = '\0';
    mag_storage_contract(mag_utf8_validate(str, len));
    stream->pos += len;
    *out_str = str;
    return true;
}

static bool mag_stream_wbuf(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    mag_storage_contract(buf != NULL || len == 0);
    mag_storage_contract(len <= UINT32_MAX);
    mag_storage_contract(mag_stream_wu32_le(stream, (uint32_t)len));
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len);
    if (len) {
        memcpy(stream->pos, buf, len);
        stream->pos += len;
    }
    return true;
}

static bool mag_stream_wbytes(mag_mem_stream_t *stream, const void *buf, size_t len) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len);
    if (len) memcpy(stream->pos, buf, len);
    stream->pos += len;
    return true;
}

static bool mag_stream_rbuf_alloc(mag_mem_stream_t *stream, uint8_t **out_buf, uint32_t *out_len) {
    mag_storage_contract(out_buf != NULL);
    mag_storage_contract(out_len != NULL);
    uint32_t len = 0;
    mag_storage_contract(mag_stream_ru32_le(stream, &len));
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len);
    uint8_t *buf = NULL;
    if (len) {
        buf = (*mag_alloc)(NULL, (size_t)len, 0);
        mag_storage_contract(buf != NULL);
        memcpy(buf, stream->pos, len);
    }
    stream->pos += len;
    *out_buf = buf;
    *out_len = len;
    return true;
}

static bool mag_stream_rbuf_view(mag_mem_stream_t *stream, const uint8_t **out_buf, uint32_t *out_len) {
    mag_storage_contract(out_buf != NULL);
    mag_storage_contract(out_len != NULL);
    uint32_t len = 0;
    mag_storage_contract(mag_stream_ru32_le(stream, &len));
    mag_storage_contract((size_t)(stream->end - stream->pos) >= len);
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
    uint32_t string_pool_len;
    uint32_t metadata_map_len;
    uint32_t tensor_header_count;
} mag_file_header_state_t;

#define MAG_FILE_HEADER_SIZE (4+4+4+4+4+4+4) /* We don't rely on struct packing */
mag_static_assert(sizeof(mag_file_header_state_t) % 4 == 0);
mag_static_assert(sizeof(mag_file_header_state_t) == MAG_FILE_HEADER_SIZE);

static bool mag_file_header_serialize(const mag_file_header_state_t *header, mag_mem_stream_t *ms) {
    mag_storage_contract(mag_stream_wu32_le(ms, header->magic));
    mag_storage_contract(mag_stream_wu32_le(ms, header->version));
    mag_storage_contract(mag_stream_wu32_le(ms, header->checksum));
    mag_storage_contract(mag_stream_wu32_le(ms, header->aux));
    mag_storage_contract(mag_stream_wu32_le(ms, header->string_pool_len));
    mag_storage_contract(mag_stream_wu32_le(ms, header->metadata_map_len));
    mag_storage_contract(mag_stream_wu32_le(ms, header->tensor_header_count));
    return true;
}

static bool mag_file_header_deserialize(mag_file_header_state_t *out_header, mag_mem_stream_t *ms) {
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->magic));
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->version));
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->checksum));
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->aux));
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->string_pool_len));
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->metadata_map_len));
    mag_storage_contract(mag_stream_ru32_le(ms, &out_header->tensor_header_count));
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
    mag_storage_contract(buf && len && len < UINT32_MAX);
    mag_storage_contract(mag_utf8_validate(buf, len));
    void *found = mag_map_lookup(&pool->map, buf, len);
    if (found) {
        *out_id = (uint32_t)(uintptr_t)found;
        return true;
    }
    mag_storage_contract(pool->len < UINT32_MAX);
    *out_id = pool->len++;
    if (pool->len > pool->cap) {
        size_t cap = pool->cap ? pool->cap : 32;
        while (cap < pool->len) cap <<= 1;
        pool->records = (*mag_alloc)(pool->records, cap*sizeof(*pool->records), 0);
        pool->cap = cap;
    }
    mag_map_insert(&pool->map, buf, len, (void *)(uintptr_t)*out_id);
    const uint8_t *owned = mag_map_lookup_key_ptr(&pool->map, buf, len);
    mag_storage_contract(owned);
    mag_pool_record_t *rec = pool->records+*out_id;
    rec->ptr = owned;
    rec->len = (uint32_t)len;
    return true;
}

static bool mag_pool_serialize(const mag_string_pool_t *pool, mag_mem_stream_t *stream) {
    mag_storage_contract(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    mag_storage_contract(pool && pool->len <= UINT32_MAX);
    mag_storage_contract(mag_stream_wu32_le(stream, (uint32_t)pool->len));
    mag_storage_contract(mag_stream_wu32_le(stream, 0)); /* offsets[0] = 0, for monotonically and clean O(1) offsets */
    uint32_t offs = 0;
    for (size_t i=0; i < pool->len; ++i) { /* Offset array */
        mag_pool_record_t *rec = pool->records+i;
        mag_storage_contract((rec->ptr || !rec->len) && rec->len <= UINT32_MAX);
        mag_storage_contract(UINT32_MAX-offs >= rec->len);
        offs += rec->len;
        mag_storage_contract(mag_stream_wu32_le(stream, offs));
    }
    for (size_t i=0; i < pool->len; ++i) { /* String blob */
        mag_pool_record_t *rec = pool->records+i;
        mag_storage_contract(mag_stream_wbytes(stream, rec->ptr, rec->len));
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
    mag_map_init(&snap->tensor_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, true);
    return snap;
}

static size_t mag_storage_estimate_size(mag_snapshot_t *snap) {
    size_t nb = 0;
    nb += MAG_FILE_HEADER_SIZE; /* File Header */
    /* String pool */
    nb += sizeof(uint32_t); /* Count */
    nb += sizeof(uint32_t)*(snap->str_pool.len+1); /* Offsets */
    for (size_t i=0; i < snap->str_pool.len; ++i)
        nb += snap->str_pool.records[i].len; /* Bytes */
    return nb;
}

bool mag_snapshot_save(mag_snapshot_t *snap, const char *filename) {
    mag_storage_contract(filename && *filename);
    const char *ext = strrchr(filename, '.');     /* check that the file extension is .mag */
    mag_storage_contract(ext != NULL && strcmp(ext, ".mag") == 0);
    mag_mem_stream_t stream;
    mag_storage_contract(mag_stream_mmap_file_w(&stream, filename, mag_storage_estimate_size(snap)));
    mag_file_header_state_t header = (mag_file_header_state_t) {
        .magic = MAG_STO_FILE_MAGIC,
        .version = 1,
        .checksum = 0,
        .aux = 0,
        .string_pool_len = 0,
        .metadata_map_len = 0,
        .tensor_header_count = 0
    };
    mag_storage_contract_do(mag_file_header_serialize(&header, &stream), goto error);
    mag_storage_contract_do(mag_pool_serialize(&snap->str_pool, &stream), goto error);
    return true;
    error:
    mag_stream_close(&stream);
    return false;
}

void mag_snapshot_put_tensor(mag_snapshot_t *snap, mag_tensor_t *tensor) {

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
