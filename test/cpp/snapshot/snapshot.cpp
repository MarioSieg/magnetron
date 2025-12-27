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

#include <prelude.hpp>

#include <core/mag_snapshot.h>

using namespace magnetron;

TEST(snapshot, metadata) {
    context ctx {};
    mag_snapshot_t *snap = mag_snapshot_new(&*ctx);

    mag_snapshot_metadata_insert(snap, "vocab_size", mag_scalar_from_u64(152064));
    mag_snapshot_metadata_insert(snap, "rms_norm_eps", mag_scalar_from_f64(1e-6));
    mag_snapshot_metadata_insert(snap, "rope_theta", mag_scalar_from_f64( 1000000.0));
    mag_snapshot_metadata_insert(snap, "max_position_embeddings", mag_scalar_from_i64(-32768));

    ASSERT_EQ(mag_scalar_as_u64(*mag_snapshot_metadata_lookup(snap, "vocab_size")), 152064u);
    ASSERT_DOUBLE_EQ(mag_scalar_as_f64(*mag_snapshot_metadata_lookup(snap, "rms_norm_eps")), 1e-6);
    ASSERT_DOUBLE_EQ(mag_scalar_as_f64(*mag_snapshot_metadata_lookup(snap, "rope_theta")), 1000000.0);
    ASSERT_EQ(mag_scalar_as_i64(*mag_snapshot_metadata_lookup(snap, "max_position_embeddings")), -32768);

    mag_snapshot_free(snap);
}
