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

#include "mag_def.h"
#include "mag_alloc.h"

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
#include <stb/stb_image_resize2.h>

mag_status_t mag_load_image(mag_tensor_t **out, mag_context_t *ctx, const char *file, const char *channels, uint32_t resize_width, uint32_t resize_height) {
    int c = strcmp(channels, "GRAY")==0 ? 1 :
           strcmp(channels, "GRAY_ALPHA")==0 ? 2 :
           strcmp(channels, "RGB")==0 ? 3 :
           strcmp(channels, "RGBA")==0 ? 4 : -1;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, (unsigned)c-1 < 4u, "c must be in {1,2,3,4}, got %d", c);
    int w, h, cf;
    stbi_uc *pixels = stbi_load(file, &w, &h, &cf, c);
    if (mag_unlikely(!pixels || w <= 0 || h <= 0 || c <= 0)) {
        if (pixels) stbi_image_free(pixels);
        return MAG_STATUS_ERR_IMAGE_ERROR;
    }
    uint32_t target_w = resize_width  > 0 ? resize_width  : (uint32_t)w;
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
    mag_status_t stat = mag_empty(&tensor, ctx, MAG_DTYPE_UINT8, 3, (int64_t[3]){c, h, w});
    if (mag_unlikely(stat != MAG_STATUS_OK)) {
        stbi_image_free(pixels);
        return stat;
    }
    uint8_t *restrict dst = (uint8_t *)mag_tensor_data_ptr_mut(tensor);
    for (int64_t k=0; k < c; ++k) /* (W,H,C) -> (C,H,W) interleaved to planar */
    for (int64_t j=0; j < h; ++j)
    for (int64_t i=0; i < w; ++i)
        dst[i + w*j + w*h*k] = pixels[k + c*i + c*w*j];
    mag_contract(ctx, ERR_IMAGE_ERROR, { stbi_image_free(pixels); }, w*h*c == mag_tensor_numel(tensor), "Buffer size mismatch: %d != %zu", w*h*c, (size_t)mag_tensor_numel(tensor));
    stbi_image_free(pixels);
    *out = tensor;
    return MAG_STATUS_OK;
}
