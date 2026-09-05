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

void mag_snap_hdr_encode(uint8_t *p, const mag_snap_decoded_hdr_t *hdr) {
  uint8_t *delta = p;
  mag_snap_wu32(&delta, hdr->magic);
  mag_snap_wu32(&delta, hdr->version);
  mag_snap_wu32(&delta, hdr->aux);
  mag_snap_wu64(&delta, hdr->meta_range.offset);
  mag_snap_wu64(&delta, hdr->meta_range.span);
  mag_snap_wu64(&delta, hdr->blob_range.offset);
  mag_snap_wu64(&delta, hdr->blob_range.span);
  mag_assert2(delta-p == MAG_SNAP_HDR_SIZE);
}

struct mag_snapshot_stream_writer_t {
  mag_context_t *ctx;
  char *path;
  char *tmp_path;
  FILE *F;
  uint64_t blob_len;
  uint64_t nb_emitted;
  bool is_closed;
  bool preserve_tmp;
};

static bool mag_snap_write(mag_snapshot_stream_writer_t *w, const void *p, size_t nb) {
  return !nb || fwrite(p,1,nb, w->F) == nb;
}

static bool mag_snap_write_zero_pad(mag_snapshot_stream_writer_t *w, size_t nb) {
  static const uint8_t zero_chunk[512] = {0};
  while (nb) {
    size_t chunk = mag_vmin(nb, sizeof(zero_chunk));
    if (mag_unlikely(!mag_snap_write(w, zero_chunk, chunk))) return false;
    nb -= chunk;
  }
  return true;
}

mag_status_t mag_snapshot_stream_writer_open(
  mag_error_t *err,
  mag_snapshot_stream_writer_t **writer,
  mag_context_t *ctx,
  const char *filepath,
  const char *meta_document,
  uint64_t meta_len,
  uint64_t blob_len
) {
  *writer = NULL;
  if (mag_unlikely(!(filepath && *filepath)))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: file path must not be empty.");
  if (mag_unlikely(!mag_snap_has_ext(filepath)))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: file '%s' must have a '.mag' extension.", filepath);
  if (mag_unlikely(meta_len && !meta_document))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: metadata length is %" PRIu64 " but the pointer is NULL.", meta_len);
  if (mag_unlikely(meta_len && !mag_snap_verify_meta_document(meta_document, meta_len)))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: metadata must be NUL-free UTF-8.");
  if (mag_unlikely(!blob_len || blob_len > (uint64_t)INT64_MAX))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: data section length must be within [1, %" PRIi64 "], but got %" PRIu64 ".", (int64_t)INT64_MAX, blob_len);
  mag_snapshot_stream_writer_t *w = (*mag_try_alloc)(NULL, sizeof(*w), 0);
  if (mag_unlikely(!w)) return mag_set_error(err, MAG_ERR_OOM, "snapshot: failed to allocate the writer.");
  memset(w, 0, sizeof(*w));
  w->ctx = ctx;
  w->blob_len = blob_len;
  size_t plen = strlen(filepath);
  w->path = (*mag_try_alloc)(NULL, plen+1, 0);
  w->tmp_path = (*mag_try_alloc)(NULL, plen+sizeof(".tmp"), 0);
  if (mag_unlikely(!(w->path && w->tmp_path))) {
    mag_snapshot_stream_writer_abort(w);
    return mag_set_error(err, MAG_ERR_OOM, "snapshot: failed to allocate writer paths.");
  }
  memcpy(w->path, filepath, plen+1);
  memcpy(w->tmp_path, filepath, plen);
  memcpy(w->tmp_path+plen, ".tmp", sizeof(".tmp"));
  if (mag_unlikely(!((w->F = mag_fopen(w->tmp_path, "wb"))))) {
    mag_snapshot_stream_writer_abort(w);
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to open '%s' for writing.", w->tmp_path);
  }
  /* Ok base structures are online let's encode and serialize the header already */
  mag_snap_decoded_hdr_t hdr = {
    .magic = MAG_SNAP_FILE_MAGIC,
    .version = MAG_SNAPSHOT_VERSION,
    .aux = MAG_SNAP_AUX_INIT,
    .meta_range = {
      .offset = MAG_SNAP_HDR_SIZE,
      .span = meta_len
    },
    .blob_range = {
      .offset = mag_align_up(MAG_SNAP_HDR_SIZE+meta_len, MAG_SNAP_FILE_BLOB_ALIGN),
      .span = blob_len
    }
  };
  uint8_t raw[MAG_SNAP_HDR_SIZE];
  mag_snap_hdr_encode(raw, &hdr);
  if (mag_unlikely(!mag_snap_write(w, raw, sizeof(raw)))) { /* Write header */
    mag_snapshot_stream_writer_abort(w);
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to write header to '%s'.", w->tmp_path);
  }
  if (mag_unlikely(!mag_snap_write(w, meta_document, meta_len))) { /* Write metadata */
    mag_snapshot_stream_writer_abort(w);
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to write metadata to '%s'.", w->tmp_path);
  }
  if (mag_unlikely(!mag_snap_write_zero_pad(w, hdr.blob_range.offset-(hdr.meta_range.offset+hdr.meta_range.span)))) { /* Write padding */
    mag_snapshot_stream_writer_abort(w);
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to write padding to '%s'.", w->tmp_path);
  }
  *writer = w;
  return MAG_OK;
}

mag_status_t mag_snapshot_stream_writer_submit_blob(
  mag_error_t *err,
  mag_snapshot_stream_writer_t *writer,
  const void *blob,
  uint64_t size
) {
  if (mag_unlikely(!writer || writer->is_closed || !writer->F))
    return mag_set_error(err, MAG_ERR_STATE, "snapshot: the writer is closed.");
  if (!size) return MAG_OK;
  if (mag_unlikely(!blob))
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: chunk is NULL but %" PRIu64 " bytes were promised.", size);
  uint64_t pad = mag_align_up(writer->nb_emitted, MAG_SNAP_TENSOR_BLOB_ALIGN)-writer->nb_emitted;
  uint64_t remaining = writer->blob_len-writer->nb_emitted;
  if (mag_unlikely(pad > remaining || size > remaining-pad)) /* Split to stay clear of overflow */
    return mag_set_error(err, MAG_ERR_PARAM, "snapshot: '%s' received %" PRIu64 " bytes more than the %" PRIu64 " it declared (%" PRIu64 " of which is alignment padding).", writer->path, pad+size-remaining, writer->blob_len, pad);
  if (mag_unlikely(pad && !mag_snap_write_zero_pad(writer, pad)))
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to write %" PRIu64 " bytes of alignment padding to '%s'.", pad, writer->tmp_path);
  writer->nb_emitted += pad;
  if (mag_unlikely(!mag_snap_write(writer, blob, size)))
    return mag_set_error(err, MAG_ERR_IO, "snapshot: failed to write %" PRIu64 " bytes to '%s'.", size, writer->tmp_path);
  writer->nb_emitted += size;
  return MAG_OK;
}

mag_status_t mag_snapshot_stream_writer_close(mag_error_t *err, mag_snapshot_stream_writer_t *writer) {
  if (mag_unlikely(!writer)) return mag_set_error(err, MAG_ERR_PARAM, "snapshot: null writer.");
  if (writer->is_closed) return MAG_OK;
  mag_status_t st = MAG_OK;
  if (mag_unlikely(writer->nb_emitted != writer->blob_len)) { /* Data is missing, would leave invalid corrupted file */
    st = mag_set_error(err, MAG_ERR_STATE, "snapshot: the data section got %" PRIu64 " of the %" PRIu64 " bytes it declared.", writer->nb_emitted, writer->blob_len);
    goto fail;
  }
  /* fflush only reaches the page cache, so a crash here would leave a renamed but empty file */
  if (mag_unlikely(!mag_fsync_stream(writer->F))) {
    st = mag_set_error(err, MAG_ERR_IO, "snapshot: failed to flush '%s' to stable storage.", writer->tmp_path);
    goto fail;
  }
  if (mag_unlikely(fclose(writer->F) != 0)) {
    writer->F = NULL;
    st = mag_set_error(err, MAG_ERR_IO, "snapshot: failed to close '%s'.", writer->tmp_path);
    goto fail;
  }
  writer->F = NULL;
  #ifdef _WIN32
    remove(writer->path); /* Unlike POSIX, rename() here cannot replace an existing file */
  #endif
  if (mag_unlikely(rename(writer->tmp_path, writer->path) != 0)) {
    /* The temp file is the only complete copy now, so keep it rather than let abort() bin it */
    writer->preserve_tmp = true;
    st = mag_set_error(err, MAG_ERR_IO, "snapshot: failed to rename '%s' to '%s', the complete snapshot has been left at '%s'.", writer->tmp_path, writer->path, writer->tmp_path);
    goto fail;
  }
  mag_fsync_parent_dir(writer->path); /* Best effort, this is what makes the rename itself durable */
  writer->is_closed = true;
  mag_snapshot_stream_writer_abort(writer);
  return MAG_OK;
  fail:
    mag_snapshot_stream_writer_abort(writer);
  return st;
}

void mag_snapshot_stream_writer_abort(mag_snapshot_stream_writer_t *writer) {
  if (!writer) return;
  if (writer->F) fclose(writer->F);
  if (!writer->is_closed && !writer->preserve_tmp && writer->tmp_path)
    remove(writer->tmp_path);
  (*mag_alloc)(writer->path, 0, 0);
  (*mag_alloc)(writer->tmp_path, 0, 0);
  (*mag_alloc)(writer, 0, 0);
}
