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

#include "mag_tensor.h"

bool mag_coords_broadcast_shape(const mag_coords_t *x, const mag_coords_t *y, int64_t *dims, int64_t *rank) {
    int64_t ar = x->rank, br = y->rank;
    int64_t r = *rank = ar > br ? ar : br;
    for (int64_t i=0; i < r; ++i) {
        int64_t ra = ar-1-i >= 0 ? x->shape[ar-1-i] : 1;
        int64_t rb = br-1-i >= 0 ? y->shape[br-1-i] : 1;
        if (mag_unlikely(!(ra == rb || ra == 1 || rb == 1))) /* Incompatible shapes */
            return false;
        dims[r-1-i] = ra == 1 ? rb : ra;
    }
    return true;
}

bool mag_coords_shape_cmp(const mag_coords_t *x, const mag_coords_t *y) {
    if (x->rank != y->rank) return false;
    for (int64_t i=0; i < x->rank; ++i)
        if (x->shape[i] != y->shape[i])
            return false;
    return true;
}

bool mag_coords_strides_cmp(const mag_coords_t *x, const mag_coords_t *y) {
    if (x->rank != y->rank) return false;
    for (int64_t i=0; i < x->rank; ++i)
        if (x->strides[i] != y->strides[i])
            return false;
    return true;
}

bool mag_coords_can_broadcast(const mag_coords_t *x, const mag_coords_t *y) {
    int64_t mr = mag_xmax(x->rank, y->rank);
    for (int64_t i=0; i < mr; ++i) {
        int64_t asz = i < x->rank ? x->shape[x->rank-1-i] : 1;
        int64_t bsz = i < y->rank ? y->shape[y->rank-1-i] : 1;
        if (asz != bsz && asz != 1 && bsz != 1)
            return false;
    }
    return true;
}

bool mag_coords_transposed(const mag_coords_t *x) {
    if (x->rank < 2) return false;
    for (int64_t i=0; i < x->rank-1; ++i) {
        int64_t s0 = x->strides[i];
        int64_t s1 = x->strides[i+1];
        if (s0 == 0 || s1 == 0) continue;
        if (s0 < s1) return true;
    }
    return false;
}

bool mag_coords_permuted(const mag_coords_t *x) {
    if (x->rank < 2) return false;
    for (int64_t i=0; i < x->rank-1; ++i) {
        int64_t s0 = x->strides[i];
        int64_t s1 = x->strides[i+1];
        if (s0 == 0 || s1 == 0) continue;
        if (s0 < s1) return true;
    }
    return false;
}

bool mag_coords_contiguous(const mag_coords_t *x) {
    int64_t y=1;
    int64_t i=x->rank-1, j;
    for (; i >= 0; --i) {
        j = x->shape[i];
        if (j == 1) continue;
        if (x->strides[i] != y) return false;
        y *= j;
    }
    return true;
}

void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank) {
    memset(*buf, 0, sizeof(*buf));
    char *p = *buf;
    *p++ = '(';
    for (int64_t i=0; i < rank; ++i) {
        p += snprintf(p, 21, "%" PRIi64, (*dims)[i]);
        if (i < rank-1) {
            *p++ = ',';
            *p++ = ' ';
        }
    }
    *p++ = ')';
    *p = '\0';
}

bool mag_solve_view_strides(int64_t (*out)[MAG_MAX_DIMS], const int64_t *osz, const int64_t *ost, int64_t ork, const int64_t *nsz, int64_t nrk) {
    int64_t numel = 1;
    for (int64_t i=0; i < ork; ++i)
        mag_assert2(!mag_mulov64(numel, osz[i], &numel));
    if (!numel) {
        if (!nrk) return false;
        (*out)[nrk-1] = 1;
        for (int64_t d=nrk-2; d >= 0; --d)
            mag_assert2(!mag_mulov64((*out)[d+1], nsz[d+1], &(*out)[d]));
        return true;
    }
    int64_t oi = ork-1;
    int64_t ni = nrk-1;
    while (oi >= 0 && ni >= 0) {
        if (nsz[ni] == 1) {
            (*out)[ni] = 0;
            --ni;
            continue;
        }
        for (; oi >= 0 && osz[oi] == 1; --oi);
        if (oi < 0) return false;
        if (nsz[ni] == osz[oi]) {
            (*out)[ni] = ost[oi];
            --ni;
            --oi;
            continue;
        }
        int64_t nc = nsz[ni];
        int64_t oc = osz[oi];
        int64_t cs = ost[oi];
        int64_t nkf = ni;
        while (nc != oc) {
            if (nc < oc) {
                --ni;
                if (ni < 0) return false;
                nc *= nsz[ni];
            } else {
                --oi;
                for (; oi >= 0 && osz[oi] == 1; --oi);
                if (oi < 0) return false;
                if (ost[oi] != osz[oi+1]*ost[oi+1])
                    return false;
                oc *= osz[oi];
            }
        }
        int64_t stride = cs;
        for (int64_t k=ni; k <= nkf; ++k) {
            (*out)[k] = stride;
            mag_assert2(!mag_mulov64(stride, nsz[k], &stride));
        }
        --ni;
        --oi;
    }
    while (ni >= 0) {
        (*out)[ni] = 0;
        --ni;
    }
    for (; oi >= 0 && osz[oi] == 1; --oi);
    return oi < 0;
}

bool mag_infer_missing_dim(int64_t(*out)[MAG_MAX_DIMS], const int64_t *dims, int64_t rank, int64_t numel) {
    int64_t prod = 1, infer = -1;
    for (int64_t i=0; i < rank; ++i) {
        int64_t ax = dims[i];
        if (ax == -1) {
            if (mag_unlikely(infer != -1)) /* Only one dimension can be inferred */
                return false;
            infer = i;
            (*out)[i] = 1;
        } else {
            if (mag_unlikely(ax <= 0)) /* Dim must be positive or -1 */
                return false;
            (*out)[i] = ax;
            mag_assert2(!mag_mulov64(prod, ax, &prod));
        }
    }
    if (infer >= 0) {
        if (mag_unlikely(numel % prod != 0)) /* Inferred dimension does not divide numel */
            return false;
        (*out)[infer] = numel / prod;
    } else if (mag_unlikely(prod != numel)) return false; /* Product does not match numel */
    return true;
}
