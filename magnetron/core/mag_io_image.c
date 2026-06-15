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

/* Include STB libraries and override their allocator with ours. */
#define STBI_MALLOC(sz) ((*mag_alloc)(NULL, (sz), 0))
#define STBI_FREE(ptr) ((*mag_alloc)((ptr), 0, 0))
#define STBI_REALLOC(ptr, sz) ((*mag_alloc)((ptr), (sz), 0))
#define STBIW_MALLOC(sz) ((*mag_alloc)(NULL, (sz), 0))
#define STBIW_FREE(ptr) ((*mag_alloc)((ptr), 0, 0))
#define STBIW_REALLOC(ptr, sz) ((*mag_alloc)((ptr), (sz), 0))
#define STBIR_MALLOC(sz, usr) ((*mag_alloc)(NULL, (sz), 0))
#define STBIR_FREE(ptr, usr) ((*mag_alloc)((ptr), 0, 0))
#define STBIR_REALLOC(ptr, sz, usr) ((*mag_alloc)((ptr), (sz), 0))
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
  mag_contract(err, ERR_INVALID_PARAM, {}, (unsigned)c-1 < 4u, "load_image: channels must be in [1, 4], but got %d.", c);
  int w, h, cf;
  stbi_uc *restrict pixels = stbi_load(file, &w, &h, &cf, c);
  if (mag_unlikely(!pixels || w <= 0 || h <= 0 || c <= 0)) {
    if (pixels) stbi_image_free(pixels);
    return MAG_STATUS_ERR_IMAGE_ERROR;
  }

  uint32_t target_w = resize_width > 0 ? resize_width : (uint32_t)w;
  uint32_t target_h = resize_height > 0 ? resize_height : (uint32_t)h;
  if ((uint32_t)w != target_w || (uint32_t)h != target_h) {
    stbir_pixel_layout layout = c == 1 ? STBIR_1CHANNEL : c == 2 ? STBIR_RA : c == 3 ? STBIR_RGB : STBIR_RGBA;
    stbi_uc *resized = stbir_resize_uint8_srgb(pixels, w, h, 0, NULL, (int)target_w, (int)target_h, 0, layout);
    if (mag_unlikely(!resized)) {
      stbi_image_free(pixels);
      return MAG_STATUS_ERR_IMAGE_ERROR;
    }
    stbi_image_free(pixels);
    pixels = resized;
    w = (int)target_w;
    h = (int)target_h;
  }

  mag_tensor_t *tensor;
  mag_try_or(mag_empty(err, &tensor, ctx, MAG_DTYPE_UINT8, 3, (int64_t[3]){c, h, w}, mag_device(CPU, 0)), {
    stbi_image_free(pixels);
  });

  uint8_t *restrict dst = (uint8_t *)mag_tensor_data_ptr_mut(tensor);
  for (int64_t k=0; k < c; ++k) /* (W,H,C) -> (C,H,W) interleaved to planar */
    for (int64_t j=0; j < h; ++j)
      for (int64_t i=0; i < w; ++i)
        dst[i + w*j + w*h*k] = pixels[k + c*i + c*w*j];

  mag_contract(err, ERR_IMAGE_ERROR, { stbi_image_free(pixels); }, w*h*c == mag_tensor_numel(tensor), "load_image: decoded pixel count (%d) does not match the tensor element count (%zu).", w*h*c, (size_t)mag_tensor_numel(tensor));
  stbi_image_free(pixels);

  mag_tensor_t *transferred = NULL;
  mag_try_or(mag_transfer(err, &transferred, tensor, device), {
    mag_tensor_decref(tensor);
  });
  mag_tensor_decref(tensor);
  *out = transferred;
  return MAG_STATUS_OK;
}

mag_status_t mag_save_image(mag_error_t *err, mag_tensor_t *tensor, const char *file) {
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor != NULL, "save_image: tensor must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, file != NULL, "save_image: file path must not be NULL.");
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->dtype == MAG_DTYPE_UINT8, "save_image: requires a uint8 tensor, but got %s.", mag_type_trait(tensor->dtype)->name);
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->coords.rank == 3, "save_image: requires a 3D tensor of shape (channels, height, width), but got rank %" PRIi64 ".", tensor->coords.rank);
  mag_contract(err, ERR_INVALID_PARAM, {}, tensor->coords.shape[0] >= 1 && tensor->coords.shape[0] <= 4, "save_image: channels must be in [1, 4], but got %" PRIi64 ".", tensor->coords.shape[0]);

  const char *ext = strrchr(file, '.');
  mag_contract(err, ERR_INVALID_PARAM, {}, ext && *ext, "save_image: file '%s' has no extension.", file);

  mag_tensor_t *host = NULL;
  mag_try(mag_transfer(err, &host, tensor, mag_device(CPU, 0)));
  mag_tensor_t *contig = NULL;
  mag_try_or(mag_contiguous(err, &contig, host), {mag_tensor_decref(host);});
  mag_tensor_decref(host);

  int64_t c = contig->coords.shape[0];
  int64_t h = contig->coords.shape[1];
  int64_t w = contig->coords.shape[2];

  const uint8_t *src = (const uint8_t *)mag_tensor_data_ptr(contig);
  uint8_t *pixels = (*mag_alloc)(NULL, w*h*c, 0);

  for (int64_t j=0; j < h; ++j)
    for (int64_t i=0; i < w; ++i)
      for (int64_t k=0; k < c; ++k)
        pixels[j*w*c + i*c + k] = src[k*w*h + j*w + i];

  int ok = 0;
  if (!strcmp(ext, ".png")) {
    ok = stbi_write_png(file, (int)w, (int)h, (int)c, pixels, (int)(w*c));
  } else if (!strcmp(ext, ".jpg") || !strcmp(ext, ".jpeg")) {
    mag_contract(err, ERR_INVALID_PARAM, {
      (*mag_alloc)(pixels, 0, 0);
      mag_tensor_decref(contig);
    }, c == 1 || c == 3, "save_image: JPEG only supports 1 or 3 channels, but got %" PRIi64 ".", c);
    ok = stbi_write_jpg(file, (int)w, (int)h, (int)c, pixels, 100);
  } else if (!strcmp(ext, ".bmp")) {
    ok = stbi_write_bmp(file, (int)w, (int)h, (int)c, pixels);
  } else if (!strcmp(ext, ".tga")) {
    ok = stbi_write_tga(file, (int)w, (int)h, (int)c, pixels);
  } else {
    (*mag_alloc)(pixels, 0, 0);
    mag_tensor_decref(contig);
    return MAG_STATUS_ERR_INVALID_PARAM;
  }

  (*mag_alloc)(pixels, 0, 0);
  mag_tensor_decref(contig);
  mag_contract(err, ERR_IMAGE_ERROR, {}, ok != 0, "save_image: failed to write '%s'.", file);
  return MAG_STATUS_OK;
}
