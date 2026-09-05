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

#ifndef MAG_IO_SNAPSHOT_H
#define MAG_IO_SNAPSHOT_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* On disk layout rules */

#define MAG_SNAP_FILE_BLOB_ALIGN 0x1000   /* Page + O_DIRECT + GDS */
#define MAG_SNAP_TENSOR_BLOB_ALIGN 0x1000 /* Page + O_DIRECT + GDS aswell */
#define mag_snap_quad_pack(a,b,c,d) ((((d)&255)<<24)+(((c)&255)<<16)+(((b)&255)<<8)+((a)&255))
#define MAG_SNAP_FILE_MAGIC mag_snap_quad_pack('M','A','G','!')
mag_static_assert(MAG_SNAP_TENSOR_BLOB_ALIGN >= MAG_CPU_BUF_ALIGN);
mag_static_assert(MAG_SNAP_FILE_BLOB_ALIGN % MAG_SNAP_TENSOR_BLOB_ALIGN == 0);

/*
** Bit 0 auf aux encodes the endianness (byte order) of the data section.
** Bits 1..31 are reserved for future use and must be written as 0 rn.
*/
#define MAG_SNAP_AUX_BIG_ENDIAN ((uint32_t)1<<0)
#define MAG_SNAP_AUX_KNOWN_BITS (MAG_SNAP_AUX_BIG_ENDIAN) /* UNion of all used bits */
#define MAG_SNAP_AUX_RESERVED_BITS (~(uint32_t)MAG_SNAP_AUX_KNOWN_BITS) /* Reserved for future must be 0 */
#ifdef MAG_BIG_ENDIAN
#define MAG_SNAP_AUX_HOST_ENDIAN MAG_SNAP_AUX_BIG_ENDIAN
#else
#define MAG_SNAP_AUX_HOST_ENDIAN ((uint32_t)0)
#endif
#define MAG_SNAP_AUX_INIT MAG_SNAP_AUX_HOST_ENDIAN
mag_static_assert((MAG_SNAP_AUX_INIT & MAG_SNAP_AUX_RESERVED_BITS) == 0);

/* In memory decoded headers / descriptors */

typedef struct mag_snap_decoded_hdr_t {
  uint32_t magic;
  uint32_t version;
  uint32_t aux;
  struct {
    uint64_t offset;
    uint64_t span;
  } meta_range;
  struct {
    uint64_t offset;
    uint64_t span;
  } blob_range;
} mag_snap_decoded_hdr_t;
#define MAG_SNAP_HDR_SIZE (4+4+4 + 8+8 + 8+8)
extern void mag_snap_hdr_encode(uint8_t *p, const mag_snap_decoded_hdr_t *hdr);
extern bool mag_snap_hdr_decode(mag_snap_decoded_hdr_t *hdr, const uint8_t *p);

static inline void mag_snap_wb(uint8_t **p, const void *src, size_t nb) {
  memcpy(*p, src, nb);
  *p += nb;
}

static inline void mag_snap_wu16(uint8_t **p, uint16_t v) {
#ifdef MAG_BIG_ENDIAN
  v = mag_bswap16(v);
#endif
  memcpy(*p, &v, sizeof(v));
  *p += sizeof(v);
}

static inline void mag_snap_wu32(uint8_t **p, uint32_t v) {
#ifdef MAG_BIG_ENDIAN
  v = mag_bswap32(v);
#endif
  memcpy(*p, &v, sizeof(v));
  *p += sizeof(v);
}

static inline void mag_snap_wu64(uint8_t **p, uint64_t v) {
#ifdef MAG_BIG_ENDIAN
  v = mag_bswap64(v);
#endif
  memcpy(*p, &v, sizeof(v));
  *p += sizeof(v);
}

static inline void mag_snap_rb(const uint8_t **p, void *dst, size_t nb) {
  memcpy(dst, *p, nb);
  *p += nb;
}

static inline uint16_t mag_snap_ru16(const uint8_t **p) {
  uint16_t v;
  memcpy(&v, *p, sizeof(v));
  *p += sizeof(v);
#ifdef MAG_BIG_ENDIAN
  v = mag_bswap16(v);
#endif
  return v;
}

static inline uint32_t mag_snap_ru32(const uint8_t **p) {
  uint32_t v;
  memcpy(&v, *p, sizeof(v));
  *p += sizeof(v);
#ifdef MAG_BIG_ENDIAN
  v = mag_bswap32(v);
#endif
  return v;
}

static inline uint64_t mag_snap_ru64(const uint8_t **p) {
  uint64_t v;
  memcpy(&v, *p, sizeof(v));
  *p += sizeof(v);
#ifdef MAG_BIG_ENDIAN
  v = mag_bswap64(v);
#endif
  return v;
}

extern bool mag_snap_has_ext(const char *filename);
extern bool mag_snap_aux_is_host_endian(uint32_t aux);
extern bool mag_snap_aux_reserved_is_clear(uint32_t aux);
extern bool mag_snap_aux_is_host_readable(uint32_t aux);
extern bool mag_snap_within_region(uint64_t off, uint64_t len, uint64_t fs);
extern bool mag_snap_verify_meta_document(const char *meta, uint64_t len);

#ifdef __cplusplus
}
#endif

#endif
