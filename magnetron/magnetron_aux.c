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

#include "magnetron_internal.h"

#define MAG_USE_STB_IMAGE_RESIZE /* Use stb_image_resize for image resizing. */

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
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize2.h>

mag_tensor_t* mag_tensor_load_image(mag_context_t* ctx, const char* file, mag_color_channels_t channels, uint32_t rw, uint32_t rh) {
   mag_assert2(file && *file);
   int w, h, c, dc;
   switch (channels) {
      default: dc = 0; break;
      case MAG_COLOR_CHANNELS_GRAY: dc = 1; break;
      case MAG_COLOR_CHANNELS_GRAY_A: dc = 2; break;
      case MAG_COLOR_CHANNELS_RGB: dc = 3; break;
      case MAG_COLOR_CHANNELS_RGBA: dc = 4; break;
   }
   uint8_t* buf = stbi_load(file, &w, &h, &c, dc);
   mag_assert(buf && w && h && c && c <= 4, "Failed to load image: %s", file);
   uint32_t whc[3] = {w, h, c};
   if (rw && rh) { /* Resize to requested dims */
    #ifdef MAG_USE_STB_IMAGE_RESIZE
        uint8_t* res = stbir_resize_uint8_srgb(
            buf,
            whc[0],
            whc[1],
            0,
            NULL,
            rw,
            rh,
            0,
            dc
       );
       if (res) { /* Replace original image data with resized data */
           stbi_image_free(buf);
           buf = res;
           whc[0] = rw;
           whc[1] = rh;
       }
    #else
        mag_e8m23_t* ori = (*mag_alloc)(NULL, whc[2]*whc[1]*whc[0]*sizeof(*ori));
        for (int64_t k=0; k < whc[2]; ++k)
         for (int64_t j=0; j < whc[1]; ++j)
             for (int64_t i=0; i < whc[0]; ++i)
                 ori[i + whc[0]*j + whc[0]*whc[1]*k] = (mag_e8m23_t)buf[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;
        mag_tensor_t* t = mag_tensor_create_3d(ctx, MAG_DTYPE_E8M23, whc[2], rh, rw);
        mag_e8m23_t* dst = mag_tensor_get_data_ptr(t);
        mag_e8m23_t* part = (*mag_alloc)(NULL, whc[2] * whc[1] * rw * sizeof(*part));
        mag_e8m23_t ws = (mag_e8m23_t)(whc[0] - 1)/(mag_e8m23_t)(rw - 1);
        mag_e8m23_t hs = (mag_e8m23_t)(whc[1] - 1)/(mag_e8m23_t)(rh - 1);
        for (uint32_t k = 0; k < whc[2]; ++k)
         for (uint32_t r = 0; r < whc[1]; ++r)
            for (uint32_t c = 0; c < rw; ++c) {
              mag_e8m23_t val = 0;
              if (c == rw - 1 || whc[0] == 1)
                  val = ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (whc[0] - 1)];
              else {
                  mag_e8m23_t sx = (mag_e8m23_t)c*ws;
                  uint32_t ix = (uint32_t)sx;
                  mag_e8m23_t dx = sx - (mag_e8m23_t)ix;
                  val = (1-dx) * (ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + ix]) + dx*(ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (ix + 1)]);
              }
              part[k * rw * (whc[1]) + r * rw + c] = val;
            }
        for (uint32_t k = 0; k < whc[2]; ++k)
         for (uint32_t r = 0; r < rh; ++r) {
             mag_e8m23_t sy = (mag_e8m23_t)r*hs;
             uint32_t iy = (uint32_t)sy;
             mag_e8m23_t dy = sy - (mag_e8m23_t)iy;
             for (uint32_t c = 0; c < rw; ++c) {
                 mag_e8m23_t val = (1-dy)*(part[k * rw * whc[1] + iy * rw + c]);
                 dst[k * rw * rh + r * rw + c] = val;
             }
             if (r == rh - 1 || whc[1] == 1) continue;
             for (uint32_t c = 0; c < rw; ++c) {
                 mag_e8m23_t val = dy*(part[k * rw * (whc[1]) + (iy + 1) * rw + c]);
                 dst[k * rw * rh + r * rw + c] += val;
             }
         }
        (*mag_alloc)(ori, 0);
        (*mag_alloc)(part, 0);
        mag_assert(rw*rh*whc[2] == mag_tensor_get_numel(t), "Buffer size mismatch: %zu != %zu", rw*rh*whc[2], (size_t)mag_tensor_get_numel(t));
        stbi_image_free(buf);
       mag_log_info("Loaded and resized tensor from image: %s, %u x %u x %u", file, rw, rh, whc[2]);
       return t;
    #endif
   }
   mag_tensor_t* t = mag_tensor_empty(ctx, MAG_DTYPE_E8M23, 3, (int64_t[3]){whc[2], whc[1], whc[0]});
   mag_e8m23_t* dst = mag_tensor_get_data_ptr(t);
   for (int64_t k = 0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
     for (int64_t j = 0; j < whc[1]; ++j) {
         for (int64_t i = 0; i < whc[0]; ++i) {
             dst[i + whc[0]*j + whc[0]*whc[1]*k] = (mag_e8m23_t)buf[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
         }
     }
   }
   mag_assert(whc[0]*whc[1]*whc[2] == mag_tensor_get_numel(t), "Buffer size mismatch: %zu != %zu", whc[0]*whc[1]*whc[2], (size_t)mag_tensor_get_numel(t));
   stbi_image_free(buf);
   mag_log_info("Loaded tensor from image: %s, %u x %u x %u", file, whc[0], whc[1], whc[2]);
   return t;
}

void mag_tensor_save_image(const mag_tensor_t* t, const char* file) {
   int64_t rank = mag_tensor_get_rank(t);
   mag_assert(rank == 3, "Tensor rank must be 3, but is: %" PRIi64, (size_t)rank);
   int64_t w = mag_tensor_get_width(t);
   int64_t h = mag_tensor_get_height(t);
   int64_t c = mag_tensor_get_channels(t);
   mag_assert(c == 1 || c == 3 || c == 4, "Invalid number of channels: %zu", (size_t)c);
   mag_assert(w*h*c == mag_tensor_get_numel(t), "Buffer size mismatch: %zu != %zu", w*h*c, (size_t)mag_tensor_get_numel(t));
   uint8_t* dst = (*mag_alloc)(NULL, w*h*c, 0); /* Allocate memory for image data */
   const mag_e8m23_t* src = mag_tensor_get_data_ptr(t);
   for (int64_t k = 0; k < c; ++k) /* Convert from planar to interleaved format. */
      for (int64_t i = 0; i < w*h; ++i)
         dst[i*c + k] = (uint8_t)(src[i + k*w*h]*255.0f);
   char ext[4+1] = {0};
   const char* dot = strrchr(file, '.');
   mag_assert(dot && *(dot+1) != '\0', "Invalid image file extension: %s", file);
   strncpy(ext, dot+1, 4);
   ext[4] = '\0';
   int stat = 0;
   if (strncmp(ext, "png", 3) == 0)
      stat = stbi_write_png(file, w, h, c, dst, 1);
   else if (strncmp(ext, "bmp", 3) == 0)
      stat = stbi_write_bmp(file, w, h, c, dst);
   else if (strncmp(ext, "tga", 3) == 0)
      stat = stbi_write_tga(file, w, h, c, dst);
   else if (strncmp(ext, "jpg", 3) == 0 || strncmp(ext, "jpeg", 4) == 0)
      stat = stbi_write_jpg(file, w, h, c, dst, 100);
   else
      mag_panic("Invalid image file extension: %s", file);
   mag_assert(stat, "Failed to save tensor to image file %s", file);
   (*mag_alloc)(dst, 0, 0); /* Free image data */
   mag_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file, (int)w, (int)h, (int)c);
}
