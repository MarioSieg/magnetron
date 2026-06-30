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

#include "mag_def.h"
#include "mag_alloc.h"
#include "mag_tensor.h"

#define DR_FLAC_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION

#include <dr_flac.h>
#include <dr_wav.h>
#include <dr_mp3.h>

static void *dr_lib_malloc(size_t sz, void *usr) { (void)usr; return (*mag_try_alloc)(NULL, sz, 0); }
static void *dr_lib_realloc(void *p, size_t sz, void *usr) { (void)usr; return (*mag_try_alloc)(p, sz, 0); }
static void dr_lib_free(void *p, void *usr) { (void)usr; (*mag_alloc)(p, 0, 0); }
static const drwav_allocation_callbacks wav_alloc_hooks = {
  .onMalloc = &dr_lib_malloc,
  .onRealloc = &dr_lib_realloc,
  .onFree = &dr_lib_free,
};
static const drflac_allocation_callbacks flac_alloc_hooks = {
  .onMalloc = &dr_lib_malloc,
  .onRealloc = &dr_lib_realloc,
  .onFree = &dr_lib_free,
};
static const drmp3_allocation_callbacks mp3_alloc_hooks = {
  .onMalloc = &dr_lib_malloc,
  .onRealloc = &dr_lib_realloc,
  .onFree = &dr_lib_free,
};

mag_status_t mag_load_audio(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, const char *file, uint32_t *out_sample_rate, mag_device_id_t device) {
  if (mag_unlikely(out == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "load_audio: output tensor pointer must not be NULL.");
  if (mag_unlikely(ctx == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "load_audio: context must not be NULL.");
  if (mag_unlikely(file == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "load_audio: file path must not be NULL.");
  const char *ext = strrchr(file, '.');
  if (mag_unlikely(ext == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "load_audio: file '%s' has no extension.", file);

  mag_status_t status = MAG_OK;
  uint32_t c = 0;
  uint32_t sample_rate = 0;
  uint64_t frames = 0;
  float *restrict samples = NULL;
  mag_tensor_t *tensor = NULL;
  mag_tensor_t *transferred = NULL;
  enum {
    AUDIO_NONE,
    AUDIO_WAV,
    AUDIO_FLAC,
    AUDIO_MP3
  } fmt = AUDIO_NONE;

  if (!strcmp(ext, ".wav")) {
    drwav_uint64 n = 0;
    samples = drwav_open_file_and_read_pcm_frames_f32(file, &c, &sample_rate, &n, &wav_alloc_hooks);
    frames = (uint64_t)n;
    fmt = AUDIO_WAV;
  } else if (!strcmp(ext, ".flac")) {
    drflac_uint64 n = 0;
    samples = drflac_open_file_and_read_pcm_frames_f32(file, &c, &sample_rate, &n, &flac_alloc_hooks);
    frames = (uint64_t)n;
    fmt = AUDIO_FLAC;
  } else if (!strcmp(ext, ".mp3")) {
    drmp3_config cfg;
    drmp3_uint64 n = 0;
    samples = drmp3_open_file_and_read_pcm_frames_f32(file, &cfg, &n, &mp3_alloc_hooks);
    if (samples) {
      c = cfg.channels;
      sample_rate = cfg.sampleRate;
      frames = (uint64_t)n;
    }
    fmt = AUDIO_MP3;
  } else {
    return mag_set_error(err, MAG_ERR_PARAM, "load_audio: unsupported audio format '%s' (supported: .wav, .flac, .mp3).", ext);
  }

  if (mag_unlikely(samples == NULL)) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "load_audio: failed to decode '%s'.", file);
    goto cleanup;
  }
  if (mag_unlikely(c == 0)) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "load_audio: '%s' has no audio channels.", file);
    goto cleanup;
  }
  if (mag_unlikely(sample_rate == 0)) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "load_audio: '%s' has an invalid sample rate.", file);
    goto cleanup;
  }
  if (mag_unlikely(frames == 0)) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "load_audio: '%s' contains no audio frames.", file);
    goto cleanup;
  }

  status = mag_empty(err, &tensor, ctx, MAG_DTYPE_FLOAT32, 2, (int64_t[2]){(int64_t)c, (int64_t)frames}, mag_device(CPU, 0));
  if (mag_iserr(status))
    goto cleanup;

  float *restrict dst = (float *)mag_tensor_data_ptr_mut(tensor);

  /* (T,C) interleaved -> (C,T) planar */
  for (uint64_t k=0; k < c; ++k)
    for (uint64_t t=0; t < frames; ++t)
      dst[t + frames*k] = samples[k + c*t];

  switch (fmt) {
    case AUDIO_WAV: drwav_free(samples, &wav_alloc_hooks); break;
    case AUDIO_FLAC: drflac_free(samples, &flac_alloc_hooks); break;
    case AUDIO_MP3: drmp3_free(samples, &mp3_alloc_hooks); break;
    default: break;
  }
  samples = NULL;

  status = mag_transfer(err, &transferred, tensor, device);
  if (mag_iserr(status))
    goto cleanup;
  mag_tensor_decref(tensor);
  tensor = NULL;

  if (out_sample_rate) *out_sample_rate = sample_rate;
  *out = transferred;
  transferred = NULL;
  return MAG_OK;

cleanup:
  if (transferred) mag_tensor_decref(transferred);
  if (tensor) mag_tensor_decref(tensor);
  if (samples) {
    switch (fmt) {
      case AUDIO_WAV: drwav_free(samples, &wav_alloc_hooks); break;
      case AUDIO_FLAC: drflac_free(samples, &flac_alloc_hooks); break;
      case AUDIO_MP3: drmp3_free(samples, &mp3_alloc_hooks); break;
      default: break;
    }
  }
  return status;
}

mag_status_t mag_save_audio(mag_error_t *err, mag_tensor_t *tensor, const char *file, uint32_t sample_rate) {
  if (mag_unlikely(tensor == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: tensor must not be NULL.");
  if (mag_unlikely(file == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: file path must not be NULL.");
  if (mag_unlikely(!(sample_rate > 0)))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: sample_rate must be > 0.");
  if (mag_unlikely(tensor->dtype != MAG_DTYPE_FLOAT32))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: requires a float32 tensor, but got %s.", mag_type_trait(tensor->dtype)->name);
  if (mag_unlikely(tensor->coords.rank != 2))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: requires a 2D tensor of shape (channels, frames), but got rank %" PRIi64 ".", tensor->coords.rank);
  if (mag_unlikely(!(tensor->coords.shape[0] > 0)))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: tensor must have at least one channel.");
  if (mag_unlikely(!(tensor->coords.shape[1] > 0)))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: tensor must have at least one frame.");

  const char *ext = strrchr(file, '.');
  if (mag_unlikely(ext == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: file '%s' has no extension.", file);
  if (mag_unlikely(strcmp(ext, ".wav") != 0))
    return mag_set_error(err, MAG_ERR_PARAM, "save_audio: only the '.wav' format is supported.");

  mag_status_t status = MAG_OK;
  mag_tensor_t *host = NULL;
  mag_tensor_t *contig = NULL;
  float *samples = NULL;
  bool wav_open = false;
  drwav wav;
  memset(&wav, 0, sizeof(wav));
  wav.allocationCallbacks = wav_alloc_hooks;

  status = mag_transfer(err, &host, tensor, mag_device(CPU, 0));
  if (mag_iserr(status))
    goto cleanup;
  status = mag_contiguous(err, &contig, host);
  if (mag_iserr(status))
    goto cleanup;
  mag_tensor_decref(host);
  host = NULL;

  int64_t c = contig->coords.shape[0];
  int64_t frames = contig->coords.shape[1];

  const float *restrict src = (const float *)mag_tensor_data_ptr(contig);

  size_t n = (size_t)c * (size_t)frames;
  samples = (*mag_try_alloc)(NULL, n*sizeof(*samples), 0);
  if (mag_unlikely(!samples)) {
    status = mag_set_error(err, MAG_ERR_OOM, "save_audio: failed to allocate %zu bytes for interleaved samples.", n*sizeof(*samples));
    goto cleanup;
  }

  /* (C,T) planar -> (T,C) interleaved */
  for (int64_t k=0; k < c; ++k)
    for (int64_t t=0; t < frames; ++t)
      samples[k + c*t] = fminf(fmaxf(src[t + frames*k], -1.0f), 1.0f);

  drwav_data_format fmt = {0};
  fmt.container = drwav_container_riff;
  fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  fmt.channels = (drwav_uint32)c;
  fmt.sampleRate = sample_rate;
  fmt.bitsPerSample = 32;

  if (mag_unlikely(!drwav_init_file_write(&wav, file, &fmt, NULL))) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "save_audio: failed to open WAV file '%s' for writing.", file);
    goto cleanup;
  }
  wav_open = true;

  drwav_uint64 written = drwav_write_pcm_frames(&wav, (drwav_uint64)frames, samples);
  drwav_uninit(&wav);
  wav_open = false;

  if (mag_unlikely(written != (drwav_uint64)frames)) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "save_audio: wrote %" PRIu64 " of %" PRIu64 " WAV frames.", (uint64_t)written, (uint64_t)frames);
    goto cleanup;
  }

cleanup:
  if (wav_open) drwav_uninit(&wav);
  if (samples) (*mag_alloc)(samples, 0, 0);
  if (contig) mag_tensor_decref(contig);
  if (host) mag_tensor_decref(host);
  return status;
}
