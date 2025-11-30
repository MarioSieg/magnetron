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

#include "mag_tensor.h"
#include "mag_alloc.h"
#include "mag_mmap.h"

typedef union mag_record_payload_t {
    mag_tensor_t *tensor;
    int64_t int64_t;
    double f64;
} mag_record_payload_t;
mag_static_assert(sizeof(mag_record_payload_t) == 8);

typedef struct mag_record_tagged_payload_t {
    mag_record_type_t type;
    mag_record_payload_t payload;
} mag_record_tagged_payload_t;
mag_static_assert(sizeof(mag_record_tagged_payload_t) == 16);

typedef struct mag_storage_record_t {
    const char *key;
    size_t key_len;
    mag_record_tagged_payload_t payload;
} mag_storage_record_t;

typedef struct mag_record_map_t {
    mag_storage_record_t *arr;
    size_t len;
    size_t cap;
} mag_record_map_t;

static void mag_record_map_rehash(mag_record_map_t *map, size_t nc) {
    mag_storage_record_t *pv = map->arr;
    size_t oc = map->cap;
    mag_storage_record_t *fresh = (*mag_alloc)(NULL, nc*sizeof(*fresh), 0);
    memset(fresh, 0, nc*sizeof(*fresh));
    size_t nl = 0;
    size_t mask = nc-1;
    for (size_t k=0; k < oc; ++k) {
        mag_storage_record_t rec = pv[k];
        if (!rec.key) continue;
        size_t h = mag_hash(rec.key, rec.key_len, 0)&mask;
        for (size_t j=h;; j=(j+1)&mask) {
            if (!fresh[j].key) {
                fresh[j] = rec;
                ++nl;
                break;
            }
        }
    }
    (*mag_alloc)(pv, 0, 0);
    map->arr = fresh;
    map->cap = nc;
    map->len = nl;
}

static void mag_record_map_init(mag_record_map_t *map) {
    size_t cap = 32;
    *map = (mag_record_map_t) {
        .arr = (*mag_alloc)(NULL, cap*sizeof(*map->arr), 0),
        .len = 0,
        .cap = cap
    };
    memset(map->arr, 0, cap*sizeof(*map->arr));
}

static size_t mag_record_required_size(const mag_storage_record_t *records, size_t num_records) {
    size_t size = 0;
    for (size_t i=0; i < num_records; ++i) {
        size += records[i].key_len;
    }
    return size;
}

static void mag_record_map_free(mag_record_map_t *map) {
    for (size_t i=0; i < map->cap; ++i) {
        if (!map->arr[i].key) continue;
        (*mag_alloc)((void *)map->arr[i].key, 0, 0);
        if (map->arr[i].payload.type == MAG_RECORD_TYPE_TENSOR && map->arr[i].payload.payload.tensor)
            mag_rc_decref(map->arr[i].payload.payload.tensor);
    }
    (*mag_alloc)(map->arr, 0, 0);
    memset(map, 0, sizeof(*map));
}

static bool mag_record_map_insert(mag_record_map_t *map, const char *key, mag_record_tagged_payload_t val) {
    if (mag_unlikely(!key)) return false;
    size_t kl = strlen(key);
    if (mag_unlikely(!kl)) return false;
    if (mag_unlikely(!mag_utf8_validate(key, kl))) return false;
    if ((map->len+1)*10 > map->cap*7)
        mag_record_map_rehash(map, map->cap<<1);
    size_t h = mag_hash(key, kl, 0);
    size_t mask = map->cap-1;
    size_t i;
    for (i=h&mask;; i=(i+1)&mask) {
        mag_storage_record_t *slot = &map->arr[i];
        if (!slot->key) break;
        if (mag_unlikely(slot->key_len == kl && !memcmp(slot->key, key, kl))) return false;
    }
    char *cloned_key = mag_strdup(key);
    if (val.type == MAG_RECORD_TYPE_TENSOR) {
        if (mag_unlikely(!val.payload.tensor)) {
            (*mag_alloc)(cloned_key, 0, 0);
            return false;
        }
        mag_rc_incref(val.payload.tensor);
    }
    map->arr[i] = (mag_storage_record_t) {
        .key = cloned_key,
        .key_len = kl,
        .payload = val
    };
    ++map->len;
    return true;
}

static mag_record_tagged_payload_t *mag_record_map_find(mag_record_map_t *map, const char *key) {
    if (mag_unlikely(!key)) return false;
    size_t kl = strlen(key);
    if (mag_unlikely(!kl || !mag_utf8_validate(key, kl))) return NULL;
    size_t h = mag_hash(key, kl, 0);
    size_t mask = map->cap-1;
    for (size_t i=h&mask;; i=(i+1)&mask) {
        mag_storage_record_t *slot = &map->arr[i];
        if (!slot->key) return NULL;
        if (slot->key_len == kl && !memcmp(slot->key, key, kl))
            return &slot->payload;
    }
}

#define mag_sto_san(expr) do { if (mag_unlikely(!(expr))) return false; } while (0)
#define mag_sto_san_do(expr, then) do { if (mag_unlikely(!(expr))) { then; } } while (0)
#define mag_sto_magic4(a,b,c,d) ((((d)&255)<<24) + (((c)&255)<<16) + (((b)&255)<<8) + ((a)&255))
#define MAG_STO_FILE_MAGIC mag_sto_magic4('M', 'A', 'G', '!')
#define MAG_STO_MAX_STR_LEN 65535
#define MAG_STO_FILE_HEADER_SIZE (4*1 + 4 + 4 + 4 + 4 + 4)
#define MAG_STO_META_HEADER_SIZE (4 + 8) /* aux + payload (key+key_len excluded) */
#define MAG_STO_TENSOR_HEADER_SIZE (4 + 8 + 8) /* aux + numel + abs_offset */
#define mag_sto_pack_aux(a,b,c,d) ((((uint8_t)(d)&255)<<24) + (((uint8_t)(c)&255)<<16) + (((uint8_t)(b)&255)<<8) + ((uint8_t)(a)&255))
#define mag_sto_unpack_aux(v, a, b, c, d) do { \
    *(a) = (v)&255; \
    *(b) = ((v)>>8)&255; \
    *(c) = ((v)>>16)&255; \
    *(d) = ((v)>>24)&255; \
} while (0)

static size_t mag_sto_aligned_buf_size(const mag_tensor_t *t) {
    mag_assert2(mag_device_is(t->storage->host, "cpu"));
    size_t sz = mag_tensor_get_data_size(t);
    mag_assert2(t->storage->alignment == MAG_CPU_BUF_ALIGN);
    mag_sto_san(sz <= SIZE_MAX-MAG_CPU_BUF_ALIGN-1);
    return (sz+(MAG_CPU_BUF_ALIGN-1))&~(MAG_CPU_BUF_ALIGN-1);
}

static inline bool mag_sto_wu32le(uint8_t **p, uint8_t *e, uint32_t v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(v));
#ifdef MAG_BE
    v = mag_bswap32(v);
#endif
    memcpy(*p, &v, sizeof(v));
    *p += sizeof(v);
    return true;
}
static inline bool mag_sto_wu64le(uint8_t **p, uint8_t *e, uint64_t v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(v));
#ifdef MAG_BE
    v = mag_bswap64(v);
#endif
    memcpy(*p, &v, sizeof(v));
    *p += sizeof(v);
    return true;
}
static inline bool mag_sto_wstr(uint8_t **p, uint8_t *e, const char *str) {
    size_t len = strlen(str);
    mag_sto_san(len <= MAG_STO_MAX_STR_LEN);
    mag_sto_san(mag_utf8_validate(str, len));
    mag_sto_san(mag_sto_wu32le(p, e, len));
    mag_sto_san((size_t)(e - *p) >= len);
    memcpy(*p, str, len);
    *p += len;
    return true;
}

static inline bool mag_sto_ru32le(const uint8_t **p, const uint8_t *e, uint32_t *v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(*v));
    memcpy(v, *p, sizeof(*v));
#ifdef MAG_BE
    *v = mag_bswap32(*v);
#endif
    *p += sizeof(*v);
    return true;
}
static inline bool mag_sto_ru64le(const uint8_t **p, const uint8_t *e, uint64_t *v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(*v));
    memcpy(v, *p, sizeof(*v));
#ifdef MAG_BE
    *v = mag_bswap64(*v);
#endif
    *p += sizeof(*v);
    return true;
}
static inline bool mag_sto_rstr(const uint8_t **p, const uint8_t *e, char **str) {
    uint32_t len;
    mag_sto_san(mag_sto_ru32le(p, e, &len));
    mag_sto_san(len <= MAG_STO_MAX_STR_LEN);
    mag_sto_san((size_t)(e - *p) >= len);
    *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(*str, *p, len);
    (*str)[len] = '\0';
    mag_sto_san_do(mag_utf8_validate(*str, len), (*mag_alloc)(*str, 0, 0); return false);
    *p += len;
    return true;
}

static bool mag_sto_patch_checksum_window_field(uint8_t *checksum_needle, uint32_t checksum) {
#ifdef MAG_BE
    checksum = mag_bswap32(checksum);
#endif
    memcpy(checksum_needle, &checksum, sizeof(checksum));
    return true;
}

static bool mag_sto_file_hdr_ser(uint8_t **p, uint8_t *e, uint32_t ver, uint32_t num_tensors, uint32_t num_meta_kv, uint8_t **checksum_needle) {
    uint8_t *b = *p;
    mag_sto_san(mag_sto_wu32le(p, e, MAG_STO_FILE_MAGIC));
    mag_sto_san(mag_sto_wu32le(p, e, ver));
    *checksum_needle = *p; /* Save the pointer to the checksum field. */
    mag_sto_san(mag_sto_wu32le(p, e, 0)); /* Checksum is written later. */
    mag_sto_san(mag_sto_wu32le(p, e, num_tensors));
    mag_sto_san(mag_sto_wu32le(p, e, num_meta_kv));
    mag_sto_san(mag_sto_wu32le(p, e, 0));
    mag_assert2(*p-b == MAG_STO_FILE_HEADER_SIZE);
    return true;
}

static bool mag_sto_file_hdr_deser(const uint8_t **p, const uint8_t *e, uint32_t *ver, uint32_t *checksum, uint32_t *num_tensors, uint32_t *num_meta_kv) {
    const uint8_t *b = *p;
    uint32_t magic;
    mag_sto_san(mag_sto_ru32le(p, e, &magic));
    mag_sto_san(magic == MAG_STO_FILE_MAGIC);
    mag_sto_san(mag_sto_ru32le(p, e, ver));
    mag_sto_san(*ver >= 1 && *ver <= MAG_STORAGE_VERSION);
    mag_sto_san(mag_sto_ru32le(p, e, checksum));
    mag_sto_san(mag_sto_ru32le(p, e, num_tensors));
    mag_sto_san(mag_sto_ru32le(p, e, num_meta_kv));
    uint32_t aux;
    mag_sto_san(mag_sto_ru32le(p, e, &aux)); /* Reserved field, must be zero. */
    mag_sto_san(aux == 0);
    mag_assert2(*p-b == MAG_STO_FILE_HEADER_SIZE);
    return true;
}

static bool mag_sto_meta_hdr_ser(uint8_t **p, uint8_t *e, const char *key, mag_record_tagged_payload_t val) {
    const uint8_t *b = *p;
    mag_sto_san(mag_sto_wu32le(p, e, mag_sto_pack_aux(val.type, 0, 0, 0)));
    switch (val.type) {
    case MAG_RECORD_TYPE_I64: {
        uint64_t u;
        memcpy(&u, &val.payload.int64_t, sizeof(u));
        mag_sto_san(mag_sto_wu64le(p, e, u));
    }
    break;
    case MAG_RECORD_TYPE_F64: {
        uint64_t u;
        memcpy(&u, &val.payload.f64, sizeof(u));
        mag_sto_san(mag_sto_wu64le(p, e, u));
    }
    break;
    case MAG_RECORD_TYPE_TENSOR:
    default:
        mag_sto_san(false);
    }
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE);
    mag_sto_san(mag_sto_wstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE+4+strlen(key));
    return true;
}

static bool mag_sto_meta_hdr_deser(const uint8_t **p, const uint8_t *e, char **key, mag_record_tagged_payload_t *val) {
    const uint8_t *b = *p;
    uint32_t aux;
    mag_sto_san(mag_sto_ru32le(p, e, &aux));
    uint8_t t, u1, u2, u3;
    mag_sto_unpack_aux(aux, &t, &u1, &u2, &u3);
    mag_sto_san(t < MAG_RECORD_TYPE__COUNT && (aux>>8) == 0);
    val->type = (mag_record_type_t)t;
    switch (val->type) {
    case MAG_RECORD_TYPE_I64: {
        uint64_t v;
        mag_sto_san(mag_sto_ru64le(p, e, &v));
        memcpy(&val->payload.int64_t, &v, sizeof(v));
    }
    break;
    case MAG_RECORD_TYPE_F64: {
        uint64_t v;
        mag_sto_san(mag_sto_ru64le(p, e, &v));
        memcpy(&val->payload.f64, &v, sizeof(v));
    }
    break;
    case MAG_RECORD_TYPE_TENSOR:
    default:
        mag_sto_san(false);
    }
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE);
    mag_sto_san(mag_sto_rstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE+4+strlen(*key));
    return true;
}

static bool mag_sto_tensor_hdr_ser(uint8_t **p, uint8_t *e, const char *key, const mag_tensor_t *t, size_t data_base, size_t data_offs) {
    const uint8_t *b = *p;
    mag_sto_san(mag_device_is(t->storage->host, "cpu"));
    mag_sto_san(mag_tensor_is_contiguous(t));
    mag_sto_san(t->coords.rank > 0 && t->coords.rank <= MAG_MAX_DIMS);
    mag_sto_san(t->dtype < MAG_DTYPE__NUM);
    mag_sto_san(mag_sto_wu32le(p, e, mag_sto_pack_aux(t->dtype, (uint8_t)t->coords.rank, 0, 0)));
    mag_sto_san(mag_sto_wu64le(p, e, t->numel));
    mag_sto_san(mag_sto_wu64le(p, e, data_base+data_offs));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE);
    for (int64_t i=0; i<t->coords.rank; ++i) {
        mag_sto_san(t->coords.shape[i] > 0);
        mag_sto_san(mag_sto_wu64le(p, e, (uint64_t)t->coords.shape[i]));
    }
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + t->coords.rank*8);
    mag_sto_san(mag_sto_wstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + t->coords.rank*8 + 4+strlen(key));
    return true;
}

static bool mag_sto_tensor_hdr_deser(const uint8_t **p, const uint8_t *e, char **key, mag_dtype_t *odt, int64_t *ork, int64_t(*oshape)[MAG_MAX_DIMS], uint64_t *sz, uint64_t *abs_offs) {
    const uint8_t *b = *p;
    uint32_t aux;
    mag_sto_san(mag_sto_ru32le(p, e, &aux));
    uint8_t dt, rank, u2, u3;
    mag_sto_unpack_aux(aux, &dt, &rank, &u2, &u3);
    mag_sto_san(dt < MAG_DTYPE__NUM && rank > 0 && rank <= MAG_MAX_DIMS && (aux>>16) == 0);
    mag_sto_san(mag_sto_ru64le(p, e, sz));
    mag_sto_san(mag_sto_ru64le(p, e, abs_offs));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE);
    for (int64_t i=0; i<rank; ++i) {
        uint64_t d;
        mag_sto_san(mag_sto_ru64le(p, e, &d));
        mag_sto_san(d > 0 && d <= INT64_MAX);
        (*oshape)[i] = (int64_t)d;
    }
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + rank*8);
    mag_sto_san(mag_sto_rstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + rank*8 + 4+strlen(*key));
    *odt = (mag_dtype_t)dt;
    *ork = (int64_t)rank;
    return true;
}

static bool mag_sto_tensor_buf_ser(uint8_t **p, uint8_t *e, const mag_tensor_t *t) {
    mag_sto_san(mag_device_is(t->storage->host, "cpu"));
    mag_sto_san(mag_tensor_is_contiguous(t));
    size_t al = MAG_CPU_BUF_ALIGN;
    size_t payload = mag_tensor_get_data_size(t);
    size_t mis = (uintptr_t)*p&(al-1);
    size_t lead = mis ? al-mis : 0;
    size_t rem = payload&(al-1);
    size_t tail = rem ? al-rem : 0;
    mag_sto_san((size_t)(e-*p) >= lead+payload+tail);
    if (lead) memset(*p, 0xff, lead), *p += lead;
    const void *src = mag_tensor_get_data_ptr(t);
    memcpy(*p, src, payload);
    if (tail) memset(*p+payload, 0xff, tail);
    *p += payload+tail;
    mag_assert2(mag_sto_aligned_buf_size(t) == payload+tail);
    return true;
}

static bool mag_sto_tensor_buf_deser(const uint8_t **p, const uint8_t *file_begin, const uint8_t *e, size_t abs_offs, size_t logical_nb, size_t aligned_nb, void *dst) {
    mag_sto_san(logical_nb <= aligned_nb);
    const uint8_t *src = file_begin + abs_offs;
    const uint8_t *region_end = src + aligned_nb;
    mag_sto_san(src >= file_begin && region_end <= e);
    if (*p != src) {
        mag_sto_san(*p <= src);
        for (const uint8_t *q=*p; q < src; ++q)
            mag_sto_san(*q == 0xff);
        *p = src;
    }
    mag_sto_san(dst != NULL);
    mag_sto_san((size_t)(e - *p) >= logical_nb);
    memcpy(dst, *p, logical_nb);
    const uint8_t *pad = *p + logical_nb;
    const uint8_t *pad_end = src + aligned_nb;
    for (const uint8_t *q=pad; q < pad_end; ++q)
        mag_sto_san(*q == 0xff);
    *p = region_end;
    return true;
}

typedef enum mag_storage_mode_t {
    MAG_STORAGE_MODE_READ = 1<<0,
    MAG_STORAGE_MODE_WRITE = 1<<1,
} mag_storage_mode_t;

struct mag_storage_archive_t {
    mag_context_t *ctx;
    char *path;
    mag_storage_mode_t mode;
    mag_record_map_t tensors;
    mag_record_map_t metadata;
};

static size_t mag_sto_estimate_file_size(const mag_record_map_t *tensors, const mag_record_map_t *metadata) {
    size_t size = MAG_STO_FILE_HEADER_SIZE;
    for (size_t i=0; i < metadata->cap; ++i) {
        if (!metadata->arr[i].key) continue;
        size += MAG_STO_META_HEADER_SIZE;
        size += 4+metadata->arr[i].key_len;
        if (metadata->arr[i].payload.type != MAG_RECORD_TYPE_TENSOR)
            size += 8;
    }
    for (size_t i=0; i < tensors->cap; ++i) {
        if (!tensors->arr[i].key) continue;
        size += MAG_STO_TENSOR_HEADER_SIZE;
        size += 8*tensors->arr[i].payload.payload.tensor->coords.rank; /* shape */
        size += 4+tensors->arr[i].key_len; /* key length + key */
        size = (size+(MAG_CPU_BUF_ALIGN-1))&~(MAG_CPU_BUF_ALIGN-1);
        size += mag_sto_aligned_buf_size(tensors->arr[i].payload.payload.tensor);
    }
    return size;
}

static bool mag_storage_read_from_buffer(mag_storage_archive_t *archive, const uint8_t *buf, size_t size) {
    const uint8_t *p = buf;
    const uint8_t *e = p+size;
    uint32_t ver, checksum, num_tensors, num_meta_kv;
    mag_sto_san(mag_sto_file_hdr_deser(&p, e, &ver, &checksum, &num_tensors, &num_meta_kv));
    for (uint32_t i=0; i < num_meta_kv; ++i) { /* Read the metadata key-value pairs */
        char *key = NULL;
        mag_record_tagged_payload_t val = {0};
        mag_sto_san(mag_sto_meta_hdr_deser(&p, e, &key, &val));
        mag_sto_san_do(mag_record_map_insert(&archive->metadata, key, val), (*mag_alloc)(key, 0, 0); return false);
        (*mag_alloc)(key, 0, 0);
    }
    return true;
}

static bool mag_storage_read_from_disk(mag_storage_archive_t *archive, const char *filename) {
    if (mag_unlikely(!archive || !filename || !*filename)) return false;
    mag_mapped_file_t map;
    if (mag_unlikely(!mag_map_file(&map, filename, 0, MAG_MAP_READ)))
        return false;
    if (mag_unlikely(map.fs < MAG_STO_FILE_HEADER_SIZE || !mag_storage_read_from_buffer(archive, map.map, map.fs))) {
        mag_unmap_file(&map);
        return false;
    }
    return mag_unmap_file(&map);
}

static bool mag_storage_write_to_disk(mag_storage_archive_t *archive, const char *filename) {
    if (mag_unlikely(!archive || !filename || !*filename)) return false;
    if (mag_unlikely(archive->tensors.len > UINT32_MAX || archive->metadata.len > UINT32_MAX)) return false;
    size_t fs = mag_sto_estimate_file_size(&archive->tensors, &archive->metadata);
    mag_mapped_file_t map;
    if (mag_unlikely(!mag_map_file(&map, filename, fs, MAG_MAP_WRITE)))
        return false;
    uint8_t *b = map.map;
    uint8_t *p = b;
    uint8_t *e = p+fs;
    uint8_t *checksum_needle = NULL;
    /* Write the file header */
    mag_sto_san_do(mag_sto_file_hdr_ser(&p, e, MAG_STORAGE_VERSION, (uint32_t)archive->tensors.len, (uint32_t)archive->metadata.len, &checksum_needle), goto error);
    for (size_t i=0; i < archive->metadata.cap; ++i) { /* Write the metadata key-value pairs */
        if (!archive->metadata.arr[i].key) continue;
        mag_sto_san_do(mag_sto_meta_hdr_ser(&p, e, archive->metadata.arr[i].key, archive->metadata.arr[i].payload), goto error);
    }
    uint8_t *dp = p;
    for (size_t i=0; i < archive->tensors.cap; ++i) { /* Compute the base offset for tensor data */
        if (!archive->tensors.arr[i].key) continue;
        mag_tensor_t *t = archive->tensors.arr[i].payload.payload.tensor;
        dp += MAG_STO_TENSOR_HEADER_SIZE;
        dp += 8*t->coords.rank;
        dp += 4+archive->tensors.arr[i].key_len;
    }
    size_t nbh = (size_t)(dp-b);
    size_t al = MAG_CPU_BUF_ALIGN;
    size_t data_base = (nbh+(al-1))&~(al-1);
    for (size_t i=0,data_offs=0; i < archive->tensors.cap; ++i) {     /* Write the tensor records */
        if (!archive->tensors.arr[i].key) continue;
        const char *key = archive->tensors.arr[i].key;
        mag_tensor_t *tensor = archive->tensors.arr[i].payload.payload.tensor;
        mag_sto_san_do(mag_sto_tensor_hdr_ser(&p, e, key, tensor, data_base, data_offs), goto error);
        data_offs += mag_sto_aligned_buf_size(tensor);
    }
    uint32_t crc = mag_crc32c(4+checksum_needle, (size_t)(p-(4+checksum_needle)));     /* Compute checksum and patch the file header */
    mag_sto_san_do(mag_sto_patch_checksum_window_field(checksum_needle, crc), goto error);
    for (size_t i=0; i < archive->tensors.cap; ++i) { /* Write the tensor data */
        if (!archive->tensors.arr[i].key) continue;
        const mag_tensor_t *t = archive->tensors.arr[i].payload.payload.tensor;
        mag_sto_san_do(mag_sto_tensor_buf_ser(&p, e, t), goto error);
    }
    return mag_unmap_file(&map);
error:
    mag_unmap_file(&map);
    return false;
}

mag_storage_archive_t *mag_storage_open(mag_context_t *ctx, const char *filename, char mode) {
    if (mag_unlikely(!ctx || !filename || !*filename)) return NULL;
    if (mag_unlikely(mode != 'r' && mode != 'w')) return NULL;
    mag_storage_archive_t *archive = (*mag_alloc)(NULL, sizeof(*archive), 0);
    archive->ctx = ctx;
    archive->path = mag_strdup(filename);
    archive->mode = mode == 'r' ? MAG_STORAGE_MODE_READ : MAG_STORAGE_MODE_WRITE;
    mag_record_map_init(&archive->tensors);
    mag_record_map_init(&archive->metadata);
    if (archive->mode & MAG_STORAGE_MODE_READ && mag_unlikely(!mag_storage_read_from_disk(archive, archive->path))) {
        (*mag_alloc)(archive, 0, 0);
        return NULL;
    }
    return archive;
}

bool mag_storage_close(mag_storage_archive_t *archive) {
    if (mag_unlikely(!archive)) return false;
    bool sync_ok = true;
    if (archive->mode & MAG_STORAGE_MODE_WRITE)
        sync_ok &= mag_storage_write_to_disk(archive, archive->path);
    (*mag_alloc)(archive->path, 0, 0);
    mag_record_map_free(&archive->tensors);
    mag_record_map_free(&archive->metadata);
    (*mag_alloc)(archive, 0, 0);
    return sync_ok;
}

const char **mag_storage_get_all_tensor_keys(mag_storage_archive_t *archive, size_t *out_len) {
    char **mem = (*mag_alloc)(NULL, archive->tensors.len*sizeof(char *), 0);
    for (size_t i=0, j=0; i < archive->tensors.cap; ++i) {
        if (!archive->tensors.arr[i].key) continue;
        mem[j++] = mag_strdup(archive->tensors.arr[i].key);
    }
    *out_len = archive->tensors.len;
    return (const char **)mem;
}

const char **mag_storage_get_all_metadata_keys(mag_storage_archive_t *archive, size_t *out_len) {
    char **mem = (*mag_alloc)(NULL, archive->metadata.len*sizeof(char *), 0);
    for (size_t i=0, j=0; i < archive->metadata.cap; ++i) {
        if (!archive->metadata.arr[i].key) continue;
        mem[j++] = mag_strdup(archive->metadata.arr[i].key);
    }
    *out_len = archive->metadata.len;
    return (const char **)mem;
}

void mag_storage_get_all_keys_free(const char **keys, size_t len) {
    for (size_t i=0; i < len; ++i)
        (*mag_alloc)((void *)keys[i], 0, 0);
    (*mag_alloc)((void *)keys, 0, 0);
}

bool mag_storage_set_tensor(mag_storage_archive_t *archive, const char *key, mag_tensor_t *tensor) {
    if (mag_unlikely(!archive || !key || !*key || !tensor)) return false;
    mag_record_tagged_payload_t val = {
        .type = MAG_RECORD_TYPE_TENSOR,
        .payload = { .tensor = tensor }
    };
    return mag_record_map_insert(&archive->tensors, key, val);
}

mag_tensor_t *mag_storage_get_tensor(mag_storage_archive_t *archive, const char *key) {
    if (mag_unlikely(!archive || !key || !*key)) return NULL;
    mag_record_tagged_payload_t *val = mag_record_map_find(&archive->tensors, key);
    if (mag_unlikely(!val || val->type != MAG_RECORD_TYPE_TENSOR || !val->payload.tensor)) return NULL;
    return val->payload.tensor;
}

bool mag_storage_set_metadata_i64(mag_storage_archive_t *archive, const char *key, int64_t value) {
    if (mag_unlikely(!archive || !key || !*key)) return false;
    mag_record_tagged_payload_t val = {
        .type = MAG_RECORD_TYPE_I64,
        .payload = {.int64_t = value}
    };
    return mag_record_map_insert(&archive->metadata, key, val);
}

bool mag_storage_get_metadata_i64(mag_storage_archive_t *archive, const char *key, int64_t *value) {
    if (mag_unlikely(!archive || !key || !*key || !value)) return false;
    mag_record_tagged_payload_t *val = mag_record_map_find(&archive->metadata, key);
    if (mag_unlikely(!val || val->type != MAG_RECORD_TYPE_I64)) {
        *value = 0;
        return false;
    }
    *value = val->payload.int64_t;
    return true;
}

bool mag_storage_set_metadata_f64(mag_storage_archive_t *archive, const char *key, double value) {
    if (mag_unlikely(!archive || !key || !*key)) return false;
    mag_record_tagged_payload_t val = {
        .type = MAG_RECORD_TYPE_F64,
        .payload = {.f64 = value}
    };
    return mag_record_map_insert(&archive->metadata, key, val);
}

bool mag_storage_get_metadata_f64(mag_storage_archive_t *archive, const char *key, double *value) {
    if (mag_unlikely(!archive || !key || !*key || !value)) return false;
    mag_record_tagged_payload_t *val = mag_record_map_find(&archive->metadata, key);
    if (mag_unlikely(!val || val->type != MAG_RECORD_TYPE_F64)) {
        *value = 0;
        return false;
    }
    *value = val->payload.f64;
    return true;
}

mag_record_type_t mag_storage_get_metadata_type(mag_storage_archive_t *archive, const char *key) {
    if (mag_unlikely(!archive || !key || !*key)) return MAG_RECORD_TYPE__COUNT;
    mag_record_tagged_payload_t *val = mag_record_map_find(&archive->metadata, key);
    return val ? val->type : MAG_RECORD_TYPE__COUNT;
}
