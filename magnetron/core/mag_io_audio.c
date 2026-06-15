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

static void *dr_lib_malloc(size_t sz, void *usr) { (void)usr; return (*mag_alloc)(NULL, sz, 0); }
static void *dr_lib_realloc(void *p, size_t sz, void *usr) { (void)usr; return (*mag_alloc)(p, sz, 0); }
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
  mag_contract(err, ERR_INVALID_PARAM, {}, out != NULL, "load_audio: output tensor pointer must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, ctx != NULL, "load_audio: context must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, file != NULL, "load_audio: file path must not be NULL.");
  uint32_t c = 0;
  uint32_t sample_rate = 0;
  uint64_t frames = 0;
  float *restrict samples = NULL;
  const char *ext = strrchr(file, '.');
  mag_contract(err, ERR_INVALID_PARAM, {}, ext != NULL, "load_audio: file '%s' has no extension.", file);

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
    return MAG_STATUS_ERR_INVALID_PARAM;
  }

  mag_contract(err, ERR_IMAGE_ERROR, {}, samples != NULL, "load_audio: failed to decode '%s'.", file);
  mag_contract(err, ERR_IMAGE_ERROR, {
    switch (fmt) {
      case AUDIO_WAV: drwav_free(samples, &wav_alloc_hooks); break;
      case AUDIO_FLAC: drflac_free(samples, &flac_alloc_hooks); break;
      case AUDIO_MP3: drmp3_free(samples, &mp3_alloc_hooks); break;
      default: break;
    }
  }, c > 0, "load_audio: '%s' has no audio channels.", file);
  mag_contract(err, ERR_IMAGE_ERROR, {
    switch (fmt) {
      case AUDIO_WAV: drwav_free(samples, &wav_alloc_hooks); break;
      case AUDIO_FLAC: drflac_free(samples, &flac_alloc_hooks); break;
      case AUDIO_MP3: drmp3_free(samples, &mp3_alloc_hooks); break;
      default: break;
    }
  }, sample_rate > 0, "load_audio: '%s' has an invalid sample rate.", file);
  mag_contract(err, ERR_IMAGE_ERROR, {
    switch (fmt) {
      case AUDIO_WAV: drwav_free(samples, &wav_alloc_hooks); break;
      case AUDIO_FLAC: drflac_free(samples, &flac_alloc_hooks); break;
      case AUDIO_MP3: drmp3_free(samples, &mp3_alloc_hooks); break;
      default: break;
    }
  }, frames > 0, "load_audio: '%s' contains no audio frames.", file);

  mag_tensor_t *tensor;
  mag_try_or(mag_empty(err, &tensor, ctx, MAG_DTYPE_FLOAT32, 2, (int64_t[2]){(int64_t)c, (int64_t)frames}, mag_device(CPU, 0)), {
    switch (fmt) {
      case AUDIO_WAV: drwav_free(samples, &wav_alloc_hooks); break;
      case AUDIO_FLAC: drflac_free(samples, &flac_alloc_hooks); break;
      case AUDIO_MP3: drmp3_free(samples, &mp3_alloc_hooks); break;
      default: break;
    }
  });

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

  mag_tensor_t *transferred = NULL;
  mag_try_or(mag_transfer(err, &transferred, tensor, device), {
    mag_tensor_decref(tensor);
  });
  mag_tensor_decref(tensor);

  if (out_sample_rate) *out_sample_rate = sample_rate;
  *out = transferred;
  return MAG_STATUS_OK;
}

mag_status_t mag_save_audio(mag_error_t *err, mag_tensor_t *tensor, const char *file, uint32_t sample_rate) {
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor != NULL, "save_audio: tensor must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, file != NULL, "save_audio: file path must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, sample_rate > 0, "save_audio: sample_rate must be > 0.");
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->dtype == MAG_DTYPE_FLOAT32, "save_audio: requires a float32 tensor, but got %s.", mag_type_trait(tensor->dtype)->name);
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->coords.rank == 2, "save_audio: requires a 2D tensor of shape (channels, frames), but got rank %" PRIi64 ".", tensor->coords.rank);
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->coords.shape[0] > 0, "save_audio: tensor must have at least one channel.");
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->coords.shape[1] > 0, "save_audio: tensor must have at least one frame.");

  const char *ext = strrchr(file, '.');
  mag_contract(err, ERR_INVALID_PARAM, {}, ext != NULL, "save_audio: file '%s' has no extension.", file);
  mag_contract(err, ERR_INVALID_PARAM, {}, !strcmp(ext, ".wav"), "save_audio: only the '.wav' format is supported.");

  mag_tensor_t *host = NULL;
  mag_try(mag_transfer(err, &host, tensor, mag_device(CPU, 0)));
  mag_tensor_t *contig = NULL;
  mag_try_or(mag_contiguous(err, &contig, host), {mag_tensor_decref(host);});
  mag_tensor_decref(host);

  int64_t c = contig->coords.shape[0];
  int64_t frames = contig->coords.shape[1];

  const float *restrict src = (const float *)mag_tensor_data_ptr(contig);

  size_t n = (size_t)c * (size_t)frames;
  float *samples = (*mag_alloc)(NULL, n*sizeof(*samples), 0);

  /* (C,T) planar -> (T,C) interleaved */
  for (int64_t k=0; k < c; ++k)
    for (int64_t t=0; t < frames; ++t)
      samples[k + c*t] = fminf(fmaxf( src[t + frames*k], -1.0f), 1.0f);

  drwav_data_format fmt = {0};
  fmt.container = drwav_container_riff;
  fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
  fmt.channels = (drwav_uint32)c;
  fmt.sampleRate = sample_rate;
  fmt.bitsPerSample = 32;

  drwav wav;
  memset(&wav, 0, sizeof(wav));
  wav.allocationCallbacks = wav_alloc_hooks;
  mag_contract(err, ERR_IMAGE_ERROR, {
    (*mag_alloc)(samples, 0, 0);
    mag_tensor_decref(contig);
  }, drwav_init_file_write(&wav, file, &fmt, NULL), "save_audio: failed to open WAV file '%s' for writing.", file);

  drwav_uint64 written = drwav_write_pcm_frames(&wav, (drwav_uint64)frames, samples);
  drwav_uninit(&wav);

  (*mag_alloc)(samples, 0, 0);
  mag_tensor_decref(contig);

  mag_contract(err, ERR_IMAGE_ERROR, {}, written == (drwav_uint64)frames, "save_audio: wrote %" PRIu64 " of %" PRIu64 " WAV frames.", (uint64_t)written, (uint64_t)frames);
  return MAG_STATUS_OK;
}

