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

static bool mag_mem_stream_mmap_file_r(mag_mem_stream_t *ms, const char *path) {
    memset(ms, 0, sizeof(*ms));
    mag_storage_contract(path != NULL && *path);
    mag_storage_contract(mag_map_file(&ms->file, path, 0, MAG_MAP_READ));
    ms->base = ms->pos = ms->file.map;
    ms->end = ms->base + ms->file.fs;
    ms->flags |= MAG_MEM_STREAM_FLAGS_ISFILE;
    return true;
}

static bool mag_mem_stream_mmap_file_w(mag_mem_stream_t *ms, const char *path, size_t size) {
    memset(ms, 0, sizeof(*ms));
    mag_storage_contract(path != NULL && *path);
    mag_storage_contract(size > 0);
    mag_storage_contract(mag_map_file(&ms->file, path, size, MAG_MAP_WRITE));
    ms->base = ms->pos = ms->file.map;
    ms->end = ms->base + ms->file.fs;
    ms->flags |= MAG_MEM_STREAM_FLAGS_ISFILE|MAG_MEM_STREAM_FLAGS_WRITE;
    return true;
}

static void mag_mem_stream_close(mag_mem_stream_t *ms) {
    if (!ms) return;
    if (ms->flags & MAG_MEM_STREAM_FLAGS_ISFILE)
        mag_unmap_file(&ms->file);
    memset(ms, 0, sizeof(*ms));
}

static bool mag_ms_wu32_le(mag_mem_stream_t *ms, uint32_t val) {
    mag_storage_contract((size_t)(ms->end - ms->pos) >= sizeof(val));
    mag_storage_contract(ms->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    ms->pos[0] = (uint8_t)val;
    ms->pos[1] = (uint8_t)(val>>8);
    ms->pos[2] = (uint8_t)(val>>16);
    ms->pos[3] = (uint8_t)(val>>24);
    ms->pos += sizeof(val);
    return true;
}

static bool mag_ms_ru32_le(mag_mem_stream_t *ms, uint32_t *out) {
    mag_storage_contract(out != NULL);
    mag_storage_contract((size_t)(ms->end - ms->pos) >= sizeof(*out));
    *out = ((uint32_t)ms->pos[0]) |
        ((uint32_t)ms->pos[1]<<8) |
        ((uint32_t)ms->pos[2]<<16) |
        ((uint32_t)ms->pos[3]<<24);
    ms->pos += sizeof(*out);
    return true;
}

static bool mag_ms_wu64_le(mag_mem_stream_t *ms, uint64_t v) {
    mag_storage_contract((size_t)(ms->end - ms->pos) >= sizeof(v));
    mag_storage_contract(ms->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    ms->pos[0] = (uint8_t)v;
    ms->pos[1] = (uint8_t)(v>>8);
    ms->pos[2] = (uint8_t)(v>>16);
    ms->pos[3] = (uint8_t)(v>>24);
    ms->pos[4] = (uint8_t)(v>>32);
    ms->pos[5] = (uint8_t)(v>>40);
    ms->pos[6] = (uint8_t)(v>>48);
    ms->pos[7] = (uint8_t)(v>>56);
    ms->pos += sizeof(v);
    return true;
}

static bool mag_ms_ru64_le(mag_mem_stream_t *ms, uint64_t *out) {
    mag_storage_contract(out != NULL);
    mag_storage_contract((size_t)(ms->end - ms->pos) >= sizeof(*out));
    *out = ((uint64_t)ms->pos[0]) |
        ((uint64_t)ms->pos[1]<<8) |
        ((uint64_t)ms->pos[2]<<16) |
        ((uint64_t)ms->pos[3]<<24) |
        ((uint64_t)ms->pos[4]<<32) |
        ((uint64_t)ms->pos[5]<<40) |
        ((uint64_t)ms->pos[6]<<48) |
        ((uint64_t)ms->pos[7]<<56);
    ms->pos += sizeof(*out);
    return true;
}

static bool mag_ms_wstr(mag_mem_stream_t *ms, const char *str) {
    mag_storage_contract(ms->flags & MAG_MEM_STREAM_FLAGS_WRITE);
    size_t len = strlen(str);
    mag_storage_contract(len <= MAG_STORAGE_MAX_STRING_LENGTH && len <= UINT32_MAX);
    mag_storage_contract(mag_utf8_validate(str, len));
    mag_storage_contract(mag_ms_wu32_le(ms, (uint32_t)len));
    mag_storage_contract((size_t)(ms->end - ms->pos) >= len);
    memcpy(ms->pos, str, len);
    ms->pos += len;
    return true;
}

static bool mag_ms_rstr(mag_mem_stream_t *ms, char **out_str) {
    uint32_t len = 0;
    mag_storage_contract(mag_ms_ru32_le(ms, &len));
    mag_storage_contract(len <= MAG_STORAGE_MAX_STRING_LENGTH);
    mag_storage_contract((size_t)(ms->end - ms->pos) >= len);
    char *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(str, ms->pos, len);
    str[len] = '\0';
    mag_storage_contract(mag_utf8_validate(str, len));
    ms->pos += len;
    *out_str = str;
    return true;
}

/* Contains the file header structure.
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

static bool mag_file_header_write(mag_mem_stream_t *ms, const mag_file_header_state_t *header) {
    mag_storage_contract(mag_ms_wu32_le(ms, header->magic));
    mag_storage_contract(mag_ms_wu32_le(ms, header->version));
    mag_storage_contract(mag_ms_wu32_le(ms, header->checksum));
    mag_storage_contract(mag_ms_wu32_le(ms, header->aux));
    mag_storage_contract(mag_ms_wu32_le(ms, header->string_pool_len));
    mag_storage_contract(mag_ms_wu32_le(ms, header->metadata_map_len));
    mag_storage_contract(mag_ms_wu32_le(ms, header->tensor_header_count));
    return true;
}

static bool mag_file_header_read(mag_mem_stream_t *ms, mag_file_header_state_t *out_header) {
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->magic));
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->version));
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->checksum));
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->aux));
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->string_pool_len));
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->metadata_map_len));
    mag_storage_contract(mag_ms_ru32_le(ms, &out_header->tensor_header_count));
    return true;
}

struct mag_snapshot_t {
    mag_context_t *ctx;
    mag_map_t metadata_map;
};

mag_snapshot_t *mag_snapshot_new(mag_context_t *ctx) {
    mag_snapshot_t *snap = (*mag_alloc)(NULL, sizeof(*snap), 0);
    memset(snap, 0, sizeof(*snap));
    snap->ctx = ctx;
    mag_map_init(&snap->metadata_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, true);
    return snap;
}

void mag_snapshot_metadata_insert(mag_snapshot_t *snap, const char *key, mag_scalar_t record) {
    mag_scalar_t *cloned_val = (*mag_alloc)(NULL, sizeof(*cloned_val), 0);
    *cloned_val = record;
    mag_map_insert(&snap->metadata_map, key, strlen(key), cloned_val);
}

const mag_scalar_t *mag_snapshot_metadata_lookup(mag_snapshot_t *snap, const char *key) {
    return mag_map_lookup(&snap->metadata_map, key, strlen(key));
}

void mag_snapshot_metadata_erase(mag_snapshot_t *snap, const char *key) {
    mag_map_erase(&snap->metadata_map, key, strlen(key));
}

static size_t mag_storage_estimate_size(mag_snapshot_t *snap) {
    size_t nb = 0;
    nb += MAG_FILE_HEADER_SIZE; /* File Header */
    return nb;
}

bool mag_snapshot_save(mag_snapshot_t *snap, const char *filename) {
    const char *ext = strrchr(filename, '.');     /* check that the file extension is .mag */
    mag_storage_contract(ext != NULL && strcmp(ext, ".mag") == 0);
    mag_mem_stream_t stream;
    mag_storage_contract(mag_mem_stream_mmap_file_w(&stream, filename, mag_storage_estimate_size(snap)));
    mag_file_header_state_t header = (mag_file_header_state_t) {
        .magic = MAG_STO_FILE_MAGIC,
        .version = 1,
        .checksum = 0,
        .aux = 0,
        .string_pool_len = 0,
        .metadata_map_len = 0,
        .tensor_header_count = 0
    };
    mag_storage_contract_do(mag_file_header_write(&stream, &header), goto error);
    return true;
    error:
    mag_mem_stream_close(&stream);
    return false;
}

void mag_snapshot_free(mag_snapshot_t *snap) {
    size_t iter = 0, len = 0;
    void *val = NULL;
    while (mag_map_next(&snap->metadata_map, &iter, &len, &val)) /* Free cloned metadata records */
        (*mag_alloc)(val, 0, 0);
    mag_map_free(&snap->metadata_map);
    memset(snap, 0, sizeof(*snap));
    (*mag_alloc)(snap, 0, 0);
}
