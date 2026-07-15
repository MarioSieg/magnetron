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

#define STBI_MALLOC(sz) ((*mag_try_alloc)(NULL, (sz), 0))
#define STBI_FREE(ptr) ((*mag_try_alloc)((ptr), 0, 0))
#define STBI_REALLOC(ptr, sz) ((*mag_try_alloc)((ptr), (sz), 0))
#define STBIW_MALLOC(sz) ((*mag_try_alloc)(NULL, (sz), 0))
#define STBIW_FREE(ptr) ((*mag_try_alloc)((ptr), 0, 0))
#define STBIW_REALLOC(ptr, sz) ((*mag_try_alloc)((ptr), (sz), 0))
#define STBIR_MALLOC(sz, usr) ((*mag_try_alloc)(NULL, (sz), 0))
#define STBIR_FREE(ptr, usr) ((*mag_try_alloc)((ptr), 0, 0))
#define STBIR_REALLOC(ptr, sz, usr) ((*mag_try_alloc)((ptr), (sz), 0))
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>
#include <stb_image_write.h>

mag_status_t mag_load_image(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, const char *file, const char *channels, uint32_t resize_width, uint32_t resize_height, mag_device_id_t device) {
  int c = !strcmp(channels, "GRAY") ? 1 : !strcmp(channels, "GRAY_ALPHA") ? 2 : !strcmp(channels, "RGB") ? 3 : !strcmp(channels, "RGBA") ? 4 : -1;
  if (mag_unlikely((unsigned)c-1 >= 4u))
    return mag_set_error(err, MAG_ERR_PARAM, "load_image: channels must be in [1, 4], but got %d.", c);

  mag_status_t status = MAG_OK;
  stbi_uc *restrict pixels = NULL;
  mag_tensor_t *tensor = NULL;
  mag_tensor_t *transferred = NULL;

  int w, h, cf;
  pixels = stbi_load(file, &w, &h, &cf, c);
  if (mag_unlikely(!pixels || w <= 0 || h <= 0 || c <= 0)) {
    if (pixels) stbi_image_free(pixels);
    return mag_set_error(err, MAG_ERR_IMAGE, "load_image: failed to decode image '%s': %s.", file, stbi_failure_reason() ? stbi_failure_reason() : "unsupported or corrupt image");
  }

  uint32_t target_w = resize_width > 0 ? resize_width : (uint32_t)w;
  uint32_t target_h = resize_height > 0 ? resize_height : (uint32_t)h;
  if ((uint32_t)w != target_w || (uint32_t)h != target_h) {
    stbir_pixel_layout layout = c == 1 ? STBIR_1CHANNEL : c == 2 ? STBIR_RA : c == 3 ? STBIR_RGB : STBIR_RGBA;
    stbi_uc *resized = stbir_resize_uint8_srgb(pixels, w, h, 0, NULL, (int)target_w, (int)target_h, 0, layout);
    if (mag_unlikely(!resized)) {
      stbi_image_free(pixels);
      return mag_set_error(err, MAG_ERR_IMAGE, "load_image: failed to resize image '%s' to %ux%u.", file, target_w, target_h);
    }
    stbi_image_free(pixels);
    pixels = resized;
    w = (int)target_w;
    h = (int)target_h;
  }

  status = mag_empty(err, &tensor, ctx, MAG_DTYPE_UINT8, 3, (int64_t[3]){c, h, w}, mag_device(CPU, 0));
  if (mag_iserr(status))
    goto cleanup;

  uint8_t *restrict dst = (uint8_t *)mag_tensor_data_ptr_mut(tensor);
  for (int64_t k=0; k < c; ++k) /* (W,H,C) -> (C,H,W) interleaved to planar */
    for (int64_t j=0; j < h; ++j)
      for (int64_t i=0; i < w; ++i)
        dst[i + w*j + w*h*k] = pixels[k + c*i + c*w*j];

  if (mag_unlikely(w*h*c != mag_tensor_numel(tensor))) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "load_image: decoded pixel count (%d) does not match the tensor element count (%zu).", w*h*c, (size_t)mag_tensor_numel(tensor));
    goto cleanup;
  }
  stbi_image_free(pixels);
  pixels = NULL;

  status = mag_transfer(err, &transferred, tensor, device);
  if (mag_iserr(status))
    goto cleanup;
  mag_tensor_decref(tensor);
  tensor = NULL;
  *out = transferred;
  transferred = NULL;
  return MAG_OK;

cleanup:
  if (transferred) mag_tensor_decref(transferred);
  if (tensor) mag_tensor_decref(tensor);
  if (pixels) stbi_image_free(pixels);
  return status;
}

mag_status_t mag_save_image(mag_error_t *err, mag_tensor_t *tensor, const char *file) {
  if (mag_unlikely(tensor == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "save_image: tensor must not be NULL.");
  if (mag_unlikely(file == NULL))
    return mag_set_error(err, MAG_ERR_PARAM, "save_image: file path must not be NULL.");
  if (mag_unlikely(tensor->meta.dtype != MAG_DTYPE_UINT8))
    return mag_set_error(err, MAG_ERR_PARAM, "save_image: requires a uint8 tensor, but got %s.", mag_type_trait(tensor->meta.dtype)->name);
  if (mag_unlikely(tensor->meta.coords.rank != 3))
    return mag_set_error(err, MAG_ERR_PARAM, "save_image: requires a 3D tensor of shape (channels, height, width), but got rank %" PRIi64 ".", tensor->meta.coords.rank);
  if (mag_unlikely(!(tensor->meta.coords.shape[0] >= 1 && tensor->meta.coords.shape[0] <= 4)))
    return mag_set_error(err, MAG_ERR_PARAM, "save_image: channels must be in [1, 4], but got %" PRIi64 ".", tensor->meta.coords.shape[0]);

  const char *ext = strrchr(file, '.');
  if (mag_unlikely(!(ext && *ext)))
    return mag_set_error(err, MAG_ERR_PARAM, "save_image: file '%s' has no extension.", file);

  mag_status_t status = MAG_OK;
  mag_tensor_t *host = NULL;
  mag_tensor_t *contig = NULL;
  uint8_t *pixels = NULL;

  status = mag_transfer(err, &host, tensor, mag_device(CPU, 0));
  if (mag_iserr(status))
    goto cleanup;
  status = mag_contiguous(err, &contig, host);
  if (mag_iserr(status))
    goto cleanup;
  mag_tensor_decref(host);
  host = NULL;

  int64_t c = contig->meta.coords.shape[0];
  int64_t h = contig->meta.coords.shape[1];
  int64_t w = contig->meta.coords.shape[2];

  const uint8_t *src = (const uint8_t *)mag_tensor_data_ptr(contig);
  pixels = (*mag_try_alloc)(NULL, w*h*c, 0);
  if (mag_unlikely(!pixels)) {
    status = mag_set_error(err, MAG_ERR_OOM, "save_image: failed to allocate %zu bytes for pixel buffer.", (size_t)(w*h*c));
    goto cleanup;
  }

  for (int64_t j=0; j < h; ++j)
    for (int64_t i=0; i < w; ++i)
      for (int64_t k=0; k < c; ++k)
        pixels[j*w*c + i*c + k] = src[k*w*h + j*w + i];

  int ok = 0;
  if (!strcmp(ext, ".png")) {
    ok = stbi_write_png(file, (int)w, (int)h, (int)c, pixels, (int)(w*c));
  } else if (!strcmp(ext, ".jpg") || !strcmp(ext, ".jpeg")) {
    if (mag_unlikely(!(c == 1 || c == 3))) {
      status = mag_set_error(err, MAG_ERR_PARAM, "save_image: JPEG only supports 1 or 3 channels, but got %" PRIi64 ".", c);
      goto cleanup;
    }
    ok = stbi_write_jpg(file, (int)w, (int)h, (int)c, pixels, 100);
  } else if (!strcmp(ext, ".bmp")) {
    ok = stbi_write_bmp(file, (int)w, (int)h, (int)c, pixels);
  } else if (!strcmp(ext, ".tga")) {
    ok = stbi_write_tga(file, (int)w, (int)h, (int)c, pixels);
  } else {
    status = MAG_ERR_PARAM;
    goto cleanup;
  }

  if (mag_unlikely(ok == 0)) {
    status = mag_set_error(err, MAG_ERR_IMAGE, "save_image: failed to write '%s'.", file);
    goto cleanup;
  }

cleanup:
  if (pixels) (*mag_alloc)(pixels, 0, 0);
  if (contig) mag_tensor_decref(contig);
  if (host) mag_tensor_decref(host);
  return status;
}
