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

bool mag_snap_has_ext(const char *filename) {
  const char *dot = strrchr(filename, '.');
  return dot && strcmp(dot, ".mag") == 0;
}

bool mag_snap_within_region(uint64_t off, uint64_t len, uint64_t fs) {
  return off <= fs && len <= fs-off;
}

bool mag_snap_verify_meta_document(const char *meta, uint64_t len) {
  return len <= MAG_SNAP_META_LIM && !memchr(meta, 0, len) && mag_utf8_validate((const uint8_t *)meta, len);
}

