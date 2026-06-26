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

#include "mag_alloc.h"
#include "mag_mmap.h"
#include "mag_romap.h"
#include "mag_tensor.h"

#include <time.h>

#include "mag_context.h"
#include "../cpu/mag_cpu.h"

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
#define MAG_SNAP_TBUF_ALIGN MAG_CPU_BUF_ALIGN /* Every tensor buffer start address must be aligned to this */
mag_static_assert(MAG_SNAP_TBUF_ALIGN == 16);

#define mag_snap_alignup(x, al) (((x)+(al)-1)&~((al)-1))

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
MAG_RC_OBJECT_IS_VALID(mag_mmap_owner_t);

static mag_status_t mag_mmap_owner_dtor(void *self) {
  mag_mmap_owner_t *o = self;
  mag_unmap_file(&o->file);
  (*mag_alloc)(o, 0, 0);
  return MAG_STATUS_OK;
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

static mag_status_t mag_stream_from_mapped_file(mag_error_t *err, mag_mem_stream_t *s, mag_mmap_owner_t *owner, bool write) {
  memset(s, 0, sizeof(*s));
  if (mag_unlikely(!(owner && owner->file.map && owner->file.fs))) {
    return mag_set_error(err, MAG_STATUS_ERR_FAILED_TO_MAP_FILE, "snapshot: invalid memory-mapped file owner.");
  }
  s->base = s->pos = owner->file.map;
  s->end = s->base + owner->file.fs;
  if (write) s->flags|=MAG_MEM_STREAM_FLAGS_WRITE;
  return MAG_STATUS_OK;
}

static void mag_stream_close(mag_mem_stream_t *stream) {
  if (!stream) return;
  memset(stream, 0, sizeof(*stream));
}

static mag_status_t mag_stream_mmap_file_w(mag_error_t *err, mag_mem_stream_t *stream, mag_mapped_file_t *map, const char *path, size_t size) {
  memset(stream, 0, sizeof(*stream));
  if (mag_unlikely(!(path != NULL && *path))) {
    return mag_set_error(err, MAG_STATUS_ERR_FAILED_TO_MAP_FILE, "snapshot: file path for memory mapping must not be empty.");
  }
  if (mag_unlikely(!(size > 0))) {
    return mag_set_error(err, MAG_STATUS_ERR_FAILED_TO_MAP_FILE, "snapshot: file size for memory mapping must be > 0.");
  }
  if (mag_unlikely(!mag_map_file(map, path, size, MAG_MAP_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_FAILED_TO_MAP_FILE, "snapshot: failed to memory-map file '%s'.", path);
  }
  stream->base = stream->pos = map->map;
  stream->end = stream->base + map->fs;
  stream->flags|=MAG_MEM_STREAM_FLAGS_WRITE;
  return MAG_STATUS_OK;
}

static size_t mag_stream_needle(const mag_mem_stream_t *stream) { return (size_t)(stream->pos - stream->base); }
static size_t mag_stream_remaining(const mag_mem_stream_t *stream) { return (size_t)(stream->end - stream->pos); }

static mag_status_t mag_stream_wu32_le(mag_error_t *err, mag_mem_stream_t *stream, uint32_t val) {
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= sizeof(val)))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient capacity to write a uint32.");
  }
  if (mag_unlikely(!(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream is read-only.");
  }
  #ifdef MAG_BIG_ENDIAN
    val = mag_bswap32(val);
  #endif
  memcpy(stream->pos, &val, sizeof(val));
  stream->pos += sizeof(val);
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_ru32_le(mag_error_t *err, mag_mem_stream_t *stream, uint32_t *val) {
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= sizeof(*val)))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient data to read a uint32.");
  }
  memcpy(val, stream->pos, sizeof(*val));
  stream->pos += sizeof(*val);
  #ifdef MAG_BIG_ENDIAN
    *val = mag_bswap32(*val);
  #endif
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_wu64_le(mag_error_t *err, mag_mem_stream_t *stream, uint64_t val) {
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= sizeof(val)))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient capacity to write a uint64.");
  }
  if (mag_unlikely(!(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream is read-only.");
  }
  #ifdef MAG_BIG_ENDIAN
    val = mag_bswap64(val);
  #endif
  memcpy(stream->pos, &val, sizeof(val));
  stream->pos += sizeof(val);
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_ru64_le(mag_error_t *err, mag_mem_stream_t *stream, uint64_t *val) {
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= sizeof(*val)))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient data to read a uint64.");
  }
  memcpy(val, stream->pos, sizeof(*val));
  stream->pos += sizeof(*val);
  #ifdef MAG_BIG_ENDIAN
    *val = mag_bswap64(*val);
  #endif
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_wstr(mag_error_t *err, mag_mem_stream_t *stream, const uint8_t *str) {
  if (mag_unlikely(!(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream is read-only.");
  }
  size_t len = strlen((const char *)str);
  if (mag_unlikely(!(len <= MAG_SNAP_MAX_STRLEN && len <= UINT32_MAX))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string length %zu exceeds the maximum of %u.", len, MAG_SNAP_MAX_STRLEN);
  }
  if (mag_unlikely(!mag_utf8_validate(str, len))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string contains invalid UTF-8.");
  }
  {
    mag_status_t s = mag_stream_wu32_le(err, stream, (uint32_t)len);
    if (mag_unlikely(s != MAG_STATUS_OK)) return s;
  }
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= len))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient capacity to write string data.");
  }
  memcpy(stream->pos, str, len);
  stream->pos += len;
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_rstr(mag_error_t *err, mag_mem_stream_t *stream, uint8_t **out_str) {
  uint32_t len = 0;
  {
    mag_status_t s = mag_stream_ru32_le(err, stream, &len);
    if (mag_unlikely(s != MAG_STATUS_OK)) return s;
  }
  if (mag_unlikely(!(len <= MAG_SNAP_MAX_STRLEN))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string length %u in stream exceeds the maximum of %u.", len, MAG_SNAP_MAX_STRLEN);
  }
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= len))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient data to read the string.");
  }
  uint8_t *str = (*mag_alloc)(NULL, len+1, 0);
  memcpy(str, stream->pos, len);
  str[len] = '\0';
  if (mag_unlikely(!mag_utf8_validate(str, len))) {
    (*mag_alloc)(str, 0, 0);
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string in stream contains invalid UTF-8.");
  }
  stream->pos += len;
  *out_str = str;
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_wbuf(mag_error_t *err, mag_mem_stream_t *stream, const void *buf, size_t len) {
  if (mag_unlikely(!(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream is read-only.");
  }
  if (mag_unlikely(!(buf != NULL || len == 0))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: buffer pointer must not be NULL when length is non-zero.");
  }
  if (mag_unlikely(!(len <= UINT32_MAX))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: buffer size %zu exceeds the maximum of %u.", len, UINT32_MAX);
  }
  {
    mag_status_t s = mag_stream_wu32_le(err, stream, (uint32_t)len);
    if (mag_unlikely(s != MAG_STATUS_OK)) return s;
  }
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= len))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient capacity to write buffer data.");
  }
  if (len) {
    memcpy(stream->pos, buf, len);
    stream->pos += len;
  }
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_wbytes(mag_error_t *err, mag_mem_stream_t *stream, const void *buf, size_t len) {
  if (mag_unlikely(!(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream is read-only.");
  }
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= len))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient capacity to write %zu bytes.", len);
  }
  if (mag_unlikely(!(buf != NULL || len == 0))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: buffer pointer must not be NULL when length is non-zero.");
  }
  if (len) memcpy(stream->pos, buf, len);
  stream->pos += len;
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_rbytes_view(mag_error_t *err, mag_mem_stream_t *stream, const uint8_t **out, size_t len) {
  if (mag_unlikely(!((size_t)(stream->end - stream->pos) >= len))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream has insufficient data to read %zu bytes.", len);
  }
  *out = stream->pos;
  stream->pos += len;
  return MAG_STATUS_OK;
}

static mag_status_t mag_stream_wzeros(mag_error_t *err, mag_mem_stream_t *stream, size_t n) {
  static const uint8_t z[MAG_SNAP_TBUF_ALIGN] = {0};
  while (n) {
    size_t k = n < sizeof(z) ? n : sizeof(z);
    {
      mag_status_t s = mag_stream_wbytes(err, stream, z, k);
      if (mag_unlikely(s != MAG_STATUS_OK)) return s;
    }
    n -= k;
  }
  return MAG_STATUS_OK;
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

#define MAG_FILE_HEADER_SIZE (4+4+8+4+4+4+4) /* We don't rely on struct packing 🐈 */
mag_static_assert(!(sizeof(mag_file_header_t)&3));
mag_static_assert(sizeof(mag_file_header_t) == MAG_FILE_HEADER_SIZE);

static mag_status_t mag_file_header_serialize(mag_error_t *err, const mag_file_header_t *header, mag_mem_stream_t *stream, uint8_t **u32_chk_patch_needle) {
  if (mag_unlikely(!(header->magic == MAG_SNAP_FILE_MAGIC))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid file magic (got 0x%x, expected 0x%x).", header->magic, MAG_SNAP_FILE_MAGIC);
  }
  mag_status_t status = MAG_STATUS_OK;
  status = mag_stream_wu32_le(err, stream, header->magic);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  if (mag_unlikely(!(header->version == MAG_SNAPSHOT_VERSION))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: unsupported version (got %u, expected %u).", header->version, MAG_SNAPSHOT_VERSION);
  }
  status = mag_stream_wu32_le(err, stream, header->version);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu64_le(err, stream, header->timestamp);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  *u32_chk_patch_needle = stream->pos; /* Needle where the checksum is overwritten later */
  status = mag_stream_wu32_le(err, stream, header->checksum);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu32_le(err, stream, header->aux);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu32_le(err, stream, header->metadata_map_len);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu32_le(err, stream, header->tensor_header_count);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  return MAG_STATUS_OK;
}

static mag_status_t mag_file_header_deserialize(mag_error_t *err, mag_file_header_t *header, mag_mem_stream_t *stream) {
  mag_status_t status = MAG_STATUS_OK;
  status = mag_stream_ru32_le(err, stream, &header->magic);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  if (mag_unlikely(!(header->magic == MAG_SNAP_FILE_MAGIC))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid file magic (got 0x%x, expected 0x%x).", header->magic, MAG_SNAP_FILE_MAGIC);
  }
  status = mag_stream_ru32_le(err, stream, &header->version);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  /* Cleanly handle if file version is too new or too old. Right now we don't support backwards compat (: */
  if (mag_unlikely(header->version != MAG_SNAPSHOT_VERSION)) {
    if (header->version < MAG_SNAPSHOT_VERSION) mag_log_error("Snapshot file version %u is older than library version %u; recreate the file or downgrade Magnetron.", header->version, MAG_SNAPSHOT_VERSION);
    else mag_log_error("Snapshot file version %u is newer than library version %u; upgrade Magnetron to read this file.", header->version, MAG_SNAPSHOT_VERSION);
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: unsupported version (got %u, expected %u).", header->version, MAG_SNAPSHOT_VERSION);
  }
  status = mag_stream_ru64_le(err, stream, &header->timestamp);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_ru32_le(err, stream, &header->checksum);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_ru32_le(err, stream, &header->aux);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_ru32_le(err, stream, &header->metadata_map_len);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_ru32_le(err, stream, &header->tensor_header_count);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  return MAG_STATUS_OK;
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

static mag_status_t mag_tensor_desc_serialize(mag_error_t *err, const mag_tensor_desc_t *desc, mag_mem_stream_t *stream) {
  mag_status_t status = MAG_STATUS_OK;
  status = mag_stream_wu32_le(err, stream, mag_pack4xu8_le(desc->rank, desc->dtype, desc->aux0, desc->aux1));
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu32_le(err, stream, desc->key_id);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu64_le(err, stream, desc->numel);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu64_le(err, stream, desc->offset);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  for (uint8_t i=0; i < desc->rank; ++i) {
    status = mag_stream_wu64_le(err, stream, desc->shape[i]);
    if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  }
  return MAG_STATUS_OK;
}

static mag_status_t mag_tensor_desc_deserialize(mag_error_t *err, mag_tensor_desc_t *desc, mag_mem_stream_t *stream, uint32_t pool_len) {
  mag_status_t status = MAG_STATUS_OK;
  uint32_t packed = 0;
  status = mag_stream_ru32_le(err, stream, &packed);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  uint8_t dtype;
  mag_unpack4xu8_le(packed, &desc->rank, &dtype, &desc->aux0, &desc->aux1);
  if (mag_unlikely(!(desc->rank < MAG_SNAP_MAX_RANK))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid tensor rank %u (maximum is %u).", desc->rank, MAG_SNAP_MAX_RANK);
  }
  if (mag_unlikely(!(dtype < MAG_DTYPE__NUM))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid tensor dtype %d.", (int)dtype);
  }
  desc->dtype = dtype;
  status = mag_stream_ru32_le(err, stream, &desc->key_id);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  if (mag_unlikely(!(desc->key_id < pool_len))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: tensor key id %u is out of range (pool size %u).", desc->key_id, pool_len);
  }
  status = mag_stream_ru64_le(err, stream, &desc->numel);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  if (mag_unlikely(!(desc->numel > 0 && desc->numel <= INT64_MAX))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid tensor numel %zu.", (size_t)desc->numel);
  }
  status = mag_stream_ru64_le(err, stream, &desc->offset);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  if (mag_unlikely(desc->offset & (MAG_CPU_BUF_ALIGN-1))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: tensor data offset %zu is not aligned to %u bytes.", (size_t)desc->offset, MAG_CPU_BUF_ALIGN);
  }
  int64_t prod = 1;
  for (uint8_t i=0; i < desc->rank; ++i) {
    uint64_t dim=0;
    status = mag_stream_ru64_le(err, stream, &dim);
    if (mag_unlikely(status != MAG_STATUS_OK)) return status;
    if (mag_unlikely(!(dim <= INT64_MAX))) {
      return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid tensor dimension size %zu.", (size_t)dim);
    }
    if (mag_unlikely(mag_mulov64(dim, prod, &prod))) {
      return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: tensor element count overflowed at dim %u (size %zu).", (unsigned)i, (size_t)dim);
    }
    desc->shape[i] = dim;
  }
  if (mag_unlikely(!(prod <= INT64_MAX && prod == desc->numel))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: tensor numel %zu does not match the product of its dimensions (%zu).", (size_t)desc->numel, (size_t)prod);
  }
  return MAG_STATUS_OK;
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
  if (mag_unlikely(!(buf && len && len < UINT32_MAX))) return false;
  if (mag_unlikely(!mag_utf8_validate(buf, len))) return false;
  void *found = mag_map_lookup(&pool->map, buf, len);
  if (found) {
    *out_id = (uint32_t)(uintptr_t)found-1;  /* unbias */
    return true;
  }
  if (mag_unlikely(!(pool->len < UINT32_MAX))) return false;
  *out_id = pool->len++;
  if (pool->len > pool->cap) {
    size_t cap = pool->cap ? pool->cap : 32;
    while (cap < pool->len) cap <<= 1;
    pool->records = (*mag_alloc)(pool->records, cap*sizeof(*pool->records), 0);
    pool->cap = cap;
  }
  mag_map_insert_if_absent(&pool->map, buf, len, (void *)(uintptr_t)(1+*out_id)/*🐱*/); /* bias by 1 to distinguish from NULL */
  const uint8_t *owned = mag_map_lookup_key_ptr(&pool->map, buf, len);
  if (mag_unlikely(!owned)) return false;
  mag_pool_record_t *rec = pool->records+*out_id;
  rec->ptr = owned;
  rec->len = (uint32_t)len;
  return true;
}

static mag_status_t mag_pool_serialize(mag_error_t *err, const mag_string_pool_t *pool, mag_mem_stream_t *stream) {
  if (mag_unlikely(!(stream->flags & MAG_MEM_STREAM_FLAGS_WRITE))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: stream is read-only.");
  }
  if (mag_unlikely(!(pool && pool->len <= UINT32_MAX))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: string pool size %zu exceeds the maximum of %u.", pool ? pool->len : 0, UINT32_MAX);
  }
  mag_status_t status = MAG_STATUS_OK;
  status = mag_stream_wu32_le(err, stream, (uint32_t)pool->len);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  status = mag_stream_wu32_le(err, stream, 0); /* offsets[0] = 0, for monotonically and clean O(1) offsets */
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  uint32_t offs = 0;
  for (size_t i=0; i < pool->len; ++i) { /* Offset array */
    mag_pool_record_t *rec = pool->records+i;
    if (mag_unlikely(!((rec->ptr || !rec->len) && rec->len <= UINT32_MAX))) {
      return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid string pool record at index %zu.", i);
    }
    if (mag_unlikely(!(UINT32_MAX-offs >= rec->len))) {
      return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: string pool blob size overflowed at index %zu (offset %u + length %u > %u).", i, offs, rec->len, UINT32_MAX);
    }
    offs += rec->len;
    status = mag_stream_wu32_le(err, stream, offs);
    if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  }
  for (size_t i=0; i < pool->len; ++i) { /* String blob */
    mag_pool_record_t *rec = pool->records+i;
    status = mag_stream_wbytes(err, stream, rec->ptr, rec->len);
    if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  }
  return MAG_STATUS_OK;
}

static mag_status_t mag_pool_deserialize(mag_error_t *err, mag_string_pool_t *pool, mag_mem_stream_t *stream) {
  mag_pool_free(pool);
  mag_pool_init(pool);
  mag_assert2(pool->len == 0); /*Pool must be fresh */
  mag_status_t status = MAG_STATUS_OK;
  uint32_t *offs = NULL;
  uint32_t count = 0;
  status = mag_stream_ru32_le(err, stream, &count);
  if (mag_unlikely(status != MAG_STATUS_OK)) return status;
  size_t num_offsets = (size_t)count+1;
  if (mag_unlikely(!(num_offsets <= MAG_SNAP_MAX_OFFSETS))) {
    return mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool contains %u entries (maximum is %u).", count, MAG_SNAP_MAX_OFFSETS-1);
  }
  offs = (*mag_alloc)(NULL, num_offsets*sizeof(*offs), 0);
  for (size_t i=0; i < num_offsets; ++i) {     /* Read in offsets */
    status = mag_stream_ru32_le(err, stream, offs+i);
    if (mag_unlikely(status != MAG_STATUS_OK)) {
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: failed to read string pool offset at index %zu.", i);
      goto cleanup;
    }
  }
  if (mag_unlikely(!(*offs == 0))) {
    status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool's first offset must be 0.");
    goto cleanup;
  }
  for (size_t i=1; i < num_offsets; ++i) { /* Verify that offsets are monotonically increasing */
    if (mag_unlikely(!(offs[i] >= offs[i-1]))) {
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool offsets are not monotonic at index %zu (%u < %u).", i, offs[i], offs[i-1]);
      goto cleanup;
    }
  }
  {
    uint32_t blob_size = offs[count];
    if (mag_unlikely(!(blob_size <= MAG_SNAP_MAX_STR_POOL_BLOB_SIZE))) {
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool blob size %u exceeds the maximum of %zu.", blob_size, (size_t)MAG_SNAP_MAX_STR_POOL_BLOB_SIZE);
      goto cleanup;
    }
    const uint8_t *blob = NULL;
    status = mag_stream_rbytes_view(err, stream, &blob, blob_size);
    if (mag_unlikely(status != MAG_STATUS_OK)) {
      status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: failed to read string pool blob from stream.");
      goto cleanup;
    }
    for (uint32_t id=0; id < count; ++id) {
      uint32_t a = offs[id];
      uint32_t b = offs[id+1];
      if (mag_unlikely(!(a <= b && b <= blob_size))) {
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: invalid string pool offsets for id %u: [%u, %u) within blob of size %u.", id, a, b, blob_size);
        goto cleanup;
      }
      const uint8_t *str = blob+a;
      uint32_t delta = b-a;
      if (mag_unlikely(!delta)) {
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool contains an empty entry at id %u.", id);
        goto cleanup;
      }
      if (mag_unlikely(!mag_utf8_validate(str, delta))) {
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool entry at id %u contains invalid UTF-8.", id);
        goto cleanup;
      }
      uint32_t len = 0;
      if (mag_unlikely(!mag_pool_intern(pool, str, delta, &len))) {
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: failed to intern string pool entry %u.", id);
        goto cleanup;
      }
      if (mag_unlikely(!(len == id))) {
        status = mag_set_error(err, MAG_STATUS_ERR_STREAM_IO_ERROR, "snapshot: string pool id mismatch (expected %u, got %u).", id, len);
        goto cleanup;
      }
    }
  }
  (*mag_alloc)(offs, 0, 0);
  return MAG_STATUS_OK;
cleanup:
  (*mag_alloc)(offs, 0, 0);
  mag_pool_free(pool);
  mag_pool_init(pool);
  return status;
}

static size_t mag_pool_compute_size(mag_string_pool_t *pool) {
  size_t nb = sizeof(uint32_t); /* Count */
  nb += sizeof(uint32_t)*(pool->len+1); /* Offsets */
  for (size_t i=0; i < pool->len; ++i)
    nb += pool->records[i].len; /* Bytes */
  return nb;
}

static bool mag_pool_find_id(mag_string_pool_t *pool, const uint8_t *buf, size_t len, uint32_t *out_id) {
  if (mag_unlikely(!(pool && buf && len && out_id))) return false;
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

static size_t mag_snap_compute_tensor_desc_size(mag_map_t *tmap) {
  size_t nb = 0, iter = 0, len = 0;
  void *val = NULL;
  while (mag_map_next(tmap, &iter, &len, &val)) {
    mag_tensor_t *tensor = val;
    nb += MAG_TENSOR_DESC_SIZE(tensor->coords.rank);
  }
  return nb;
}

static size_t mag_snap_compute_tensor_data_size(mag_map_t *tmap) {
  size_t nb = 0, iter = 0, len = 0;
  void *val = NULL;
  size_t al = MAG_SNAP_TBUF_ALIGN-1;
  while (mag_map_next(tmap, &iter, &len, &val)) {
    mag_tensor_t *tensor = val;
    nb = (nb+al)&~al;
    nb += mag_tensor_numbytes(tensor);
  }
  return nb;
}

static size_t mag_snap_compute_size(mag_snapshot_t *snap) {
  size_t meta = 0;
  meta += MAG_FILE_HEADER_SIZE;
  meta += 4; /* SRP! */
  meta += mag_pool_compute_size(&snap->str_pool);
  meta += 4; /* MDT! */
  meta += 4; /* DSC! */
  meta += mag_snap_compute_tensor_desc_size(&snap->tensor_map);
  meta += 4; /* BUF! */
  size_t al = MAG_SNAP_TBUF_ALIGN-1;
  size_t db_pad = ((meta+al)&~al) - meta;
  return meta+db_pad+mag_snap_compute_tensor_data_size(&snap->tensor_map);
}

mag_status_t mag_snapshot_new(mag_error_t *err, mag_snapshot_t **out_snap, mag_context_t *ctx) {
  mag_snapshot_t *snap = (*mag_alloc)(NULL, sizeof(*snap), 0);
  memset(snap, 0, sizeof(*snap));
  snap->ctx = ctx;
  mag_pool_init(&snap->str_pool);
  mag_map_init(&snap->tensor_map, MAG_SNAPSHOT_META_MAP_DEFAULT_CAP, true);
  *out_snap = snap;
  return MAG_STATUS_OK;
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
  if (mag_unlikely(!(snap && tensor))) return false;
  if (mag_unlikely(!(key_id < snap->str_pool.len))) return false;
  if (mag_unlikely(mag_map_lookup(&snap->tensor_map, &key_id, sizeof(key_id)))) return false; /* Already exists */
  mag_map_insert_if_absent(&snap->tensor_map, &key_id, sizeof(key_id), tensor);
  mag_tensor_incref(tensor);
  return true;
}

static void mag_snapshot_mmap_borrow_release(void *usr) {
  if (usr) mag_rc_decref(usr);
}

mag_status_t mag_snapshot_deserialize(mag_error_t *err, mag_snapshot_t **out_snap, mag_context_t *ctx, const char *filename) {
  if (mag_unlikely(!(filename && *filename))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: file path must not be empty.");
  }
  const char *ext = strrchr(filename, '.'); /* check that the file extension is .mag */
  if (mag_unlikely(!(ext != NULL && strcmp(ext, ".mag") == 0))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: file '%s' must have a '.mag' extension.", filename);
  }

  mag_tensor_desc_t *stable = NULL;
  mag_snapshot_t *snap = NULL;
  {
    mag_status_t s = mag_snapshot_new(err, &snap, ctx);
    if (mag_unlikely(s != MAG_STATUS_OK)) return s;
  }
  mag_mem_stream_t *stream = &snap->stream;
  snap->mmap_owner = mag_mmap_owner_open(filename);
  if (mag_unlikely(!(snap->mmap_owner != NULL))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to open file '%s'.", filename);
  }
  {
    mag_status_t s = mag_stream_from_mapped_file(err, stream, snap->mmap_owner, false);
    if (mag_unlikely(s != MAG_STATUS_OK)) return s;
  }

  mag_status_t status = MAG_STATUS_OK;

  if (mag_unlikely(!(mag_stream_remaining(stream) >= MAG_FILE_HEADER_SIZE + 4*MAG_SNAP_SECTION_MARKERS_COUNT))) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: file '%s' is truncated (size %zu < required %d).", filename, mag_stream_remaining(stream), MAG_FILE_HEADER_SIZE + 4*MAG_SNAP_SECTION_MARKERS_COUNT);
    goto cleanup;
  }

  snap->nb_total = mag_stream_remaining(stream);
  size_t marker = mag_stream_needle(stream);

  /* File header */
  mag_file_header_t header = {0};
  status = mag_file_header_deserialize(err, &header, stream);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read the file header from '%s'.", filename);
    goto cleanup;
  }
  mag_assert2(mag_stream_needle(stream)-marker == MAG_FILE_HEADER_SIZE); /* Verify exact file header bytes written */

  /* String pool */
  marker = mag_stream_needle(stream);
  uint32_t section_marker = 0;
  status = mag_stream_ru32_le(err, stream, &section_marker);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read the string pool section marker in '%s'.", filename);
    goto cleanup;
  }
  if (mag_unlikely(!(section_marker == MAG_SNAP_SECTION_STR_POOL))) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid string pool section marker in '%s' (got 0x%08x, expected 0x%08x).", filename, section_marker, MAG_SNAP_SECTION_STR_POOL);
    goto cleanup;
  }
  status = mag_pool_deserialize(err, &snap->str_pool, stream);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read the string pool from '%s'.", filename);
    goto cleanup;
  }
  mag_assert2(mag_stream_needle(stream)-marker == 4+mag_pool_compute_size(&snap->str_pool)); /* Verify exact section marker + pool bytes written */

  status = mag_stream_ru32_le(err, stream, &section_marker);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read the metadata section marker in '%s'.", filename);
    goto cleanup;
  }
  if (mag_unlikely(!(section_marker == MAG_SNAP_SECTION_META_DATA))) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid metadata section marker in '%s' (got 0x%08x, expected 0x%08x).", filename, section_marker, MAG_SNAP_SECTION_META_DATA);
    goto cleanup;
  }
  /* TODO: metadata */

  size_t nt = header.tensor_header_count;
  stable = (*mag_alloc)(NULL, nt*sizeof(*stable), 0);

  status = mag_stream_ru32_le(err, stream, &section_marker);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read the tensor descriptor section marker in '%s'.", filename);
    goto cleanup;
  }
  if (mag_unlikely(!(section_marker == MAG_SNAP_SECTION_TENSOR_DESC))) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid tensor descriptor section marker in '%s' (got 0x%08x, expected 0x%08x).", filename, section_marker, MAG_SNAP_SECTION_TENSOR_DESC);
    goto cleanup;
  }
  for (uint32_t i=0; i < nt; ++i) {
    mag_tensor_desc_t *desc = stable+i;
    status = mag_tensor_desc_deserialize(err, desc, stream, snap->str_pool.len);
    if (mag_unlikely(status != MAG_STATUS_OK)) {
      status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read tensor descriptor %u from '%s'.", i, filename);
      goto cleanup;
    }
  }
  /* Read data */
  status = mag_stream_ru32_le(err, stream, &section_marker);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read the tensor data section marker in '%s'.", filename);
    goto cleanup;
  }
  if (mag_unlikely(!(section_marker == MAG_SNAP_SECTION_TENSOR_DATA))) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: invalid tensor data section marker in '%s' (got 0x%08x, expected 0x%08x).", filename, section_marker, MAG_SNAP_SECTION_TENSOR_DATA);
    goto cleanup;
  }
  {
    size_t db = mag_stream_needle(stream);
    size_t al = MAG_SNAP_TBUF_ALIGN-1;
    size_t db_al = (db+al)&~al;
    const uint8_t *ignored = NULL;
    status = mag_stream_rbytes_view(err, stream, &ignored, db_al - db);
    if (mag_unlikely(status != MAG_STATUS_OK)) {
      status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read tensor data section padding in '%s'.", filename);
      goto cleanup;
    }
    snap->nb_meta = mag_stream_needle(stream); /* Everything up to here is metadata */
    mag_device_t *cpu_device=NULL;
    if (mag_unlikely(!mag_backend_registry_get_backend_and_device_by_id(snap->ctx->backend_registry, mag_device(CPU, 0), NULL, &cpu_device))) {
      status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: cannot deserialize '%s' (CPU backend is not available).", filename);
      goto cleanup;
    }

    uint64_t offset=0;
    mag_tensor_t *tensor = NULL;
    for (size_t i=0; i < nt; ++i) {
      const mag_tensor_desc_t *desc = stable+i;
      uint64_t delta = desc->offset;
      size_t elsize = mag_type_trait(desc->dtype)->size;
      size_t numel = (size_t)desc->numel;
      int64_t shape[MAG_SNAP_MAX_RANK];
      for (uint8_t j=0; j < desc->rank && j < sizeof(shape)/sizeof(*shape); ++j) shape[j] = (int64_t)desc->shape[j];
      size_t nb = numel*elsize;
      const uint8_t *blob = NULL;
      mag_assert2(((int64_t)delta-(int64_t)offset)>=0);
      uint64_t pad = delta-offset;
      ignored = NULL;
      status = mag_stream_rbytes_view(err, stream, &ignored, pad);
      if (mag_unlikely(status != MAG_STATUS_OK)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read tensor padding in '%s'.", filename);
        goto cleanup;
      }
      offset = delta;
      status = mag_stream_rbytes_view(err, stream, &blob, nb);
      if (mag_unlikely(status != MAG_STATUS_OK)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to read tensor data in '%s'.", filename);
        goto cleanup;
      }
      if (mag_unlikely(al & (uintptr_t)blob)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: tensor %zu data in '%s' is not aligned to %u bytes (address %p).", i, filename, MAG_SNAP_TBUF_ALIGN, (void *)blob);
        goto cleanup;
      }
      tensor = NULL;
      status = mag_borrow_cpu_buffer(err, &tensor, ctx, (void *)blob, nb, desc->dtype, desc->rank, shape, false, &mag_snapshot_mmap_borrow_release, snap->mmap_owner);
      if (mag_unlikely(status != MAG_STATUS_OK)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to borrow CPU buffer for tensor %zu in '%s'.", i, filename);
        goto cleanup;
      }
      mag_rc_incref(snap->mmap_owner);
      if (mag_unlikely(!mag_snapshot_insert_tensor_by_id(snap, desc->key_id, tensor))) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to insert tensor with key id %u into '%s'.", desc->key_id, filename);
        mag_tensor_decref(tensor);
        tensor = NULL;
        goto cleanup;
      }
      mag_tensor_decref(tensor); /* Decref as the snapshot now holds a reference */
      tensor = NULL;
      offset += nb;
    }
  }
  snap->nb_storage = mag_stream_needle(stream) - snap->nb_meta;
  if (mag_unlikely(!(snap->nb_total == snap->nb_meta + snap->nb_storage))) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: '%s' size mismatch (file has %zu bytes, but metadata + storage sum to %zu).", filename, snap->nb_total, snap->nb_meta+snap->nb_storage);
    goto cleanup;
  }
  (*mag_alloc)(stable, 0, 0);
  *out_snap = snap;
  return MAG_STATUS_OK;
cleanup:
  if (stable) (*mag_alloc)(stable, 0, 0);
  mag_snapshot_free(snap);
  return status;
}

mag_status_t mag_snapshot_serialize(mag_error_t *err, mag_snapshot_t *snap, const char *filename) {
  if (mag_unlikely(!(filename && *filename))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: file path must not be empty.");
  }
  const char *ext = strrchr(filename, '.'); /* check that the file extension is .mag */
  if (mag_unlikely(!(ext != NULL && strcmp(ext, ".mag") == 0))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: file '%s' must have a '.mag' extension.", filename);
  }
  if (mag_unlikely(!(snap->tensor_map.nitems <= UINT32_MAX))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: contains %zu tensors (maximum is %u).", snap->tensor_map.nitems, UINT32_MAX);
  }
  mag_mem_stream_t stream = {0};
  mag_mapped_file_t map = {0};
  if (mag_unlikely(!mag_isok(mag_stream_mmap_file_w(err, &stream, &map, filename, mag_snap_compute_size(snap))))) {
    return mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to memory-map file '%s' for writing.", filename);
  }
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
  mag_status_t status = MAG_STATUS_OK;

  /* File header */
  marker = mag_stream_needle(&stream);
  uint8_t *u32_chk_patch_needle; /* Where to patch the checksum */
  status = mag_file_header_serialize(err, &header, &stream, &u32_chk_patch_needle);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write the file header to '%s'.", filename);
    goto cleanup;
  }
  const uint8_t *chk_start = u32_chk_patch_needle+sizeof(uint32_t); /* Checksum start region, excluding checksum field itself */
  mag_assert2(mag_stream_needle(&stream)-marker == MAG_FILE_HEADER_SIZE); /* Verify exact file header bytes written */

  /* String pool */
  marker = mag_stream_needle(&stream);
  status = mag_stream_wu32_le(err, &stream, MAG_SNAP_SECTION_STR_POOL);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write the string pool section marker to '%s'.", filename);
    goto cleanup;
  }
  status = mag_pool_serialize(err, &snap->str_pool, &stream);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write the string pool to '%s'.", filename);
    goto cleanup;
  }
  mag_assert2(mag_stream_needle(&stream)-marker == 4+mag_pool_compute_size(&snap->str_pool)); /* Verify exact section marker + pool bytes written */

  status = mag_stream_wu32_le(err, &stream, MAG_SNAP_SECTION_META_DATA);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write the metadata section marker to '%s'.", filename);
    goto cleanup;
  }

  stable = (*mag_alloc)(NULL, snap->tensor_map.nitems*sizeof(*stable), 0);
  uint64_t offs = 0;
  size_t iter = 0, klen = 0; /* Write tensor headers */
  void *key = NULL, *val = NULL;
  size_t k;
  status = mag_stream_wu32_le(err, &stream, MAG_SNAP_SECTION_TENSOR_DESC);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write the tensor descriptor section marker to '%s'.", filename);
    goto cleanup;
  }
  for (k=0; k < snap->tensor_map.nitems && (key = mag_map_next(&snap->tensor_map, &iter, &klen, &val)); ++k) {  /* Tensor descriptors */
    mag_assert2(klen == sizeof(uint32_t));
    uint32_t key_id = *(const uint32_t *)key;
    offs = mag_snap_alignup(offs, MAG_SNAP_TBUF_ALIGN);
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
    if (mag_unlikely(!(tensor->coords.rank >= 0 && tensor->coords.rank <= MAG_SNAP_MAX_RANK))) {
      status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: tensor with key id %u has unsupported rank %d (maximum is %d).", key_id, (int)tensor->coords.rank, MAG_SNAP_MAX_RANK);
      goto cleanup;
    }
    for (int64_t i=0; i < tensor->coords.rank; ++i) {
      mag_assert2(tensor->coords.shape[i] >= 0);
      desc.shape[i] = (uint64_t)tensor->coords.shape[i];
    }
    marker = mag_stream_needle(&stream);
    status = mag_tensor_desc_serialize(err, &desc, &stream);
    if (mag_unlikely(status != MAG_STATUS_OK)) {
      status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write tensor descriptor for key id %u to '%s'.", key_id, filename);
      goto cleanup;
    }
    mag_assert2(mag_stream_needle(&stream)-marker == MAG_TENSOR_DESC_SIZE(tensor->coords.rank));
    offs += mag_tensor_numbytes(tensor);
    stable[k] = tensor;
  }
  mag_assert2(k == snap->tensor_map.nitems);

  /* Compute checksum of metadata before data section starts */
  {
    const uint8_t *chk_end = stream.pos;
    mag_device_t *dvc_interface;
    mag_backend_registry_get_backend_and_device_by_id(snap->ctx->backend_registry, mag_device(CPU, 0), NULL, &dvc_interface);
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
  }

  /* Tensor data section */
  status = mag_stream_wu32_le(err, &stream, MAG_SNAP_SECTION_TENSOR_DATA);
  if (mag_unlikely(status != MAG_STATUS_OK)) {
    status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write the tensor data section marker to '%s'.", filename);
    goto cleanup;
  }
  {
    size_t db = mag_stream_needle(&stream);
    size_t align = MAG_SNAP_TBUF_ALIGN-1;
    size_t db_al = (db+align)&~align;
    status = mag_stream_wzeros(err, &stream, db_al-db);
    if (mag_unlikely(status != MAG_STATUS_OK)) {
      status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write tensor data section padding to '%s'.", filename);
      goto cleanup;
    }
    marker = db_al;
    size_t data_offs = 0;
    mag_tensor_t *tensor = NULL;
    for (size_t i=0; i < snap->tensor_map.nitems; ++i) { /* Tensor data */
      tensor = stable[i];
      size_t al = (data_offs+align)&~align;
      status = mag_stream_wzeros(err, &stream, al-data_offs);
      if (mag_unlikely(status != MAG_STATUS_OK)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write tensor padding to '%s'.", filename);
        mag_tensor_decref(tensor);
        goto cleanup;
      }
      data_offs = al;
      if (mag_unlikely(!(tensor->storage->device->id.type == MAG_BACKEND_TYPE_CPU))) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: only CPU tensors can be serialized, but tensor %zu in '%s' resides on a non-CPU device.", i, filename);
        mag_tensor_decref(tensor);
        goto cleanup;
      }
      status = mag_contiguous(err, &tensor, tensor);
      if (mag_unlikely(status != MAG_STATUS_OK)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to make tensor %zu contiguous for serialization to '%s'.", i, filename);
        mag_tensor_decref(tensor);
        goto cleanup;
      }
      size_t nb = mag_tensor_numbytes(tensor);
      data_offs += nb;
      status = mag_stream_wbytes(err, &stream, (const void *)mag_tensor_data_ptr(tensor), nb);
      if (mag_unlikely(status != MAG_STATUS_OK)) {
        status = mag_set_error(err, MAG_STATUS_ERR_SERIALIZATION_ERROR, "snapshot: failed to write tensor %zu data to '%s'.", i, filename);
        mag_tensor_decref(tensor);
        goto cleanup;
      }
      mag_tensor_decref(tensor);
    }
    mag_assert2(mag_stream_needle(&stream)-marker == data_offs); /* total bytes */
    mag_assert2(mag_stream_needle(&stream) == stream.end-stream.base); /* All pre-estimated bytes must be written, down to the last crumb of cookie */
  }
  mag_stream_close(&stream);
  mag_unmap_file(&map);
  (*mag_alloc)(stable, 0, 0);
  return MAG_STATUS_OK;
cleanup:
  mag_stream_close(&stream);
  mag_unmap_file(&map);
  if (stable) (*mag_alloc)(stable, 0, 0);
  return status;
}

mag_tensor_t *mag_snapshot_get_tensor(mag_snapshot_t *snap, const char *key) {
  if (mag_unlikely(!(snap && key && *key))) return NULL;
  uint32_t key_id = 0;
  if (mag_unlikely(!mag_pool_find_id(&snap->str_pool, (const uint8_t*)key, strlen(key), &key_id))) return NULL;
  mag_tensor_t *found = mag_map_lookup(&snap->tensor_map, &key_id, sizeof(key_id));
  if (found) mag_tensor_incref(found);
  return found;
}

bool mag_snapshot_put_tensor(mag_snapshot_t *snap, const char *key, mag_tensor_t *tensor) {
  if (mag_unlikely(!(key && *key && tensor))) return false;
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
