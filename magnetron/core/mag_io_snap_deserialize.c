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

#include "mag_io_snapshot_layout.h"
#include "mag_alloc.h"
#include "mag_mmap.h"
#include "mag_rc.h"

bool mag_snap_has_ext(const char *filename) {
  const char *dot = strrchr(filename, '.');
  return dot && strcmp(dot, ".mag") == 0;
}

bool mag_snap_aux_is_host_endian(uint32_t aux) {
  return (aux & MAG_SNAP_AUX_BIG_ENDIAN) == MAG_SNAP_AUX_HOST_ENDIAN;
}

bool mag_snap_aux_reserved_is_clear(uint32_t aux) {
  return !(aux & MAG_SNAP_AUX_RESERVED_BITS);
}

bool mag_snap_aux_is_host_readable(uint32_t aux) {
  return mag_snap_aux_reserved_is_clear(aux) && mag_snap_aux_is_host_endian(aux);
}

bool mag_snap_within_region(uint64_t off, uint64_t len, uint64_t fs) {
  return off <= fs && len <= fs-off;
}

bool mag_snap_verify_meta_document(const char *meta, uint64_t len) {
  return !memchr(meta, 0, len) && mag_utf8_validate((const uint8_t *)meta, len);
}

struct mag_snapshot_stream_reader_t {
  MAG_RC_INJECT_HEADER; /* RC control block must be first */
  mag_context_t *ctx;
  mag_mapped_file_t mf;
  mag_snap_decoded_hdr_t hdr;
  const uint8_t *meta;      /* !Points into mapping NOT NUL terminated!!! */
  const uint8_t *blob;    /* Page aligned and points into the mapping too */
};
MAG_RC_OBJECT_IS_VALID(mag_snapshot_stream_reader_t);

static mag_status_t mag_snap_reader_dtor(void *self) {
  mag_snapshot_stream_reader_t *r = self;
  mag_munmap_file(&r->mf);
  (*mag_alloc)(r, 0, 0);
  return MAG_OK;
}

static void mag_snap_reader_release(void *usr) {
  mag_rc_decref(usr);
}

bool mag_snap_hdr_decode(mag_snap_decoded_hdr_t *hdr, const uint8_t *p) {
  const uint8_t *delta = p;
  hdr->magic = mag_snap_ru32(&delta);
  hdr->version = mag_snap_ru32(&delta);
  hdr->aux = mag_snap_ru32(&delta);
  hdr->meta_range.offset = mag_snap_ru64(&delta);
  hdr->meta_range.span = mag_snap_ru64(&delta);
  hdr->blob_range.offset = mag_snap_ru64(&delta);
  hdr->blob_range.span = mag_snap_ru64(&delta);
  return delta-p == MAG_SNAP_HDR_SIZE;
}

mag_status_t mag_snapshot_stream_reader_open(
  mag_error_t *err,
  mag_snapshot_stream_reader_t **reader,
  mag_context_t *ctx,
  const char *filepath
) {
  *reader = NULL;
  if (mag_unlikely(!(filepath && *filepath)))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: file path must not be empty.");
  mag_mapped_file_t mf = {0};
  if (mag_unlikely(!mag_mmap_file(&mf, filepath, 0, MAG_MAP_READ)))
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to open '%s' for reading.", filepath);
  mag_status_t stat = MAG_OK;
  uint64_t fs = mf.fs;
  if (mag_unlikely(fs < MAG_SNAP_HDR_SIZE)) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' is %" PRIu64 " bytes, too short for the %d byte header.", filepath, fs, MAG_SNAP_HDR_SIZE);
    goto fail;
  }
  mag_snap_decoded_hdr_t hdr = {0};
  mag_snap_hdr_decode(&hdr, mf.map);
  if (mag_unlikely(hdr.magic != MAG_SNAP_FILE_MAGIC)) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' is not a magnetron snapshot, magic is 0x%08x and not 0x%08x.", filepath, hdr.magic, (uint32_t)MAG_SNAP_FILE_MAGIC);
    goto fail;
  }
  if (mag_unlikely(!mag_snap_aux_reserved_is_clear(hdr.aux))) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' sets reserved aux bits 0x%08x, it was written by a newer magnetron.", filepath, hdr.aux & MAG_SNAP_AUX_RESERVED_BITS);
    goto fail;
  }
  if (mag_unlikely(!mag_snap_aux_is_host_endian(hdr.aux))) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' holds %s endian data and this host is %s endian.", filepath, (hdr.aux & MAG_SNAP_AUX_BIG_ENDIAN) ? "big" : "little", (MAG_SNAP_AUX_HOST_ENDIAN & MAG_SNAP_AUX_BIG_ENDIAN) ? "big" : "little");
    goto fail;
  }
  if (mag_unlikely(mag_ver_major(hdr.version) != mag_ver_major(MAG_SNAPSHOT_VERSION) || hdr.version > MAG_SNAPSHOT_VERSION)) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' is format version %u.%u.%u, this build reads %u.%u.%u.", filepath, mag_ver_major(hdr.version), mag_ver_minor(hdr.version), mag_ver_patch(hdr.version), mag_ver_major(MAG_SNAPSHOT_VERSION), mag_ver_minor(MAG_SNAPSHOT_VERSION), mag_ver_patch(MAG_SNAPSHOT_VERSION));
    goto fail;
  }
  if (mag_unlikely(hdr.meta_range.offset < MAG_SNAP_HDR_SIZE || !mag_snap_within_region(hdr.meta_range.offset, hdr.meta_range.span, fs))) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' puts its metadata at [%" PRIu64 ", %" PRIu64 ") of a %" PRIu64 " byte file.", filepath, hdr.meta_range.offset, hdr.meta_range.offset+hdr.meta_range.span, fs);
    goto fail;
  }
  if (mag_unlikely(!mag_snap_within_region(hdr.blob_range.offset, hdr.blob_range.span, fs))) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' puts its data section at [%" PRIu64 ", %" PRIu64 ") of a %" PRIu64 " byte file.", filepath, hdr.blob_range.offset, hdr.blob_range.offset+hdr.blob_range.span, fs);
    goto fail;
  }
  if (mag_unlikely(hdr.blob_range.offset < hdr.meta_range.offset+hdr.meta_range.span)) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' has a data section starting at %" PRIu64 " that overlaps its metadata ending at %" PRIu64 ".", filepath, hdr.blob_range.offset, hdr.meta_range.offset+hdr.meta_range.span);
    goto fail;
  }
  if (mag_unlikely(hdr.blob_range.offset % MAG_SNAP_FILE_BLOB_ALIGN)) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' starts its data section at %" PRIu64 ", which is not a multiple of %d.", filepath, hdr.blob_range.offset, MAG_SNAP_FILE_BLOB_ALIGN);
    goto fail;
  }
  if (mag_unlikely(!mag_snap_verify_meta_document((const char *)mf.map+hdr.meta_range.offset, hdr.meta_range.span))) {
    stat = mag_set_error(err, MAG_ERR_PARAM, "snapshot: the metadata of '%s' is not NUL free UTF-8.", filepath);
    goto fail;
  }
  mag_snapshot_stream_reader_t *r = (*mag_try_alloc)(NULL, sizeof(*r), 0);
  if (mag_unlikely(!r)) {
    stat = mag_set_error(err, MAG_ERR_OOM, "snapshot: failed to allocate the reader.");
    goto fail;
  }
  *r = (mag_snapshot_stream_reader_t) {
    .ctx = ctx,
    .mf = mf,
    .hdr = hdr,
    .meta = mf.map+hdr.meta_range.offset,
    .blob = mf.map+hdr.blob_range.offset
  };
  mag_rc_init_object(r, &mag_snap_reader_dtor);
  *reader = r;
  return MAG_OK;
  fail:
    mag_munmap_file(&mf);
  return stat;
}

const char *mag_snapshot_stream_reader_meta(const mag_snapshot_stream_reader_t *reader, uint64_t *out_len) {
  *out_len = reader->hdr.meta_range.span;
  return (const char *)reader->meta;
}

uint64_t mag_snapshot_stream_reader_blob_len(const mag_snapshot_stream_reader_t *reader) { return reader->hdr.blob_range.span; }
uint32_t mag_snapshot_stream_reader_version(const mag_snapshot_stream_reader_t *reader) { return reader->hdr.version; }

mag_status_t mag_snapshot_stream_reader_borrow_tensor(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_snapshot_stream_reader_t *reader,
  uint64_t offset,
  uint64_t size,
  mag_dtype_t dtype,
  int64_t rank,
  const int64_t *shape
) {
  *out = NULL;
  if (mag_unlikely(!reader))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: null reader.");
  if (mag_unlikely((unsigned)dtype >= MAG_DTYPE__NUM))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: dtype ordinal %u is not a dtype this build knows.", (unsigned)dtype);
  if (mag_unlikely(!size))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: cannot borrow an empty tensor, it owns no bytes to borrow.");
  uint64_t blob_len = reader->hdr.blob_range.span;
  if (mag_unlikely(!mag_snap_within_region(offset, size, blob_len)))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: a tensor spanning [%" PRIu64 ", %" PRIu64 ") does not fit the %" PRIu64 " byte data section.", offset, offset+size, blob_len);
  if (mag_unlikely(offset % MAG_SNAP_TENSOR_BLOB_ALIGN))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: a tensor starts at %" PRIu64 ", which is not a multiple of the %d byte tensor alignment, so its bytes cannot be borrowed.", offset, MAG_SNAP_TENSOR_BLOB_ALIGN);
  mag_rc_incref(reader);
  mag_status_t stat = mag_borrow_cpu_buffer(
    err,
    out,
    reader->ctx,
    (void *)(reader->blob+offset),
    size,
    dtype,
    rank,
    shape,
    false,
    &mag_snap_reader_release,
    reader
  );
  if (mag_unlikely(stat != MAG_OK)) mag_rc_decref(reader);
  return stat;
}

void mag_snapshot_stream_reader_close(mag_snapshot_stream_reader_t *reader) {
  if (reader) mag_rc_decref(reader);
}
