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

#include "mag_shape.h"
#include "mag_tensor.h"

void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    memset(*buf, 0, sizeof(*buf));
    char *p = *buf;
    mag_assert2(p+rank*21+3 < *buf+MAG_FMT_DIM_BUF_SIZE);
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

bool mag_compute_broadcast_shape(const mag_tensor_t *a, const mag_tensor_t *b, int64_t *dims, int64_t *rank) {
    int64_t ar = a->rank, br = b->rank;
    int64_t r = *rank = ar > br ? ar : br;
    for (int64_t i=0; i < r; ++i) {
        int64_t ra = ar-1-i >= 0 ? a->shape[ar-1-i] : 1;
        int64_t rb = br-1-i >= 0 ? b->shape[br-1-i] : 1;
        if (mag_unlikely(!(ra == rb || ra == 1 || rb == 1))) /* Incompatible shapes */
            return false;
        dims[r-1-i] = ra == 1 ? rb : ra;
    }
    return true;
}

bool mag_tensor_is_shape_eq(const mag_tensor_t *x, const mag_tensor_t *y) {
    return memcmp(x->shape, y->shape, sizeof(x->shape)) == 0;
}

bool mag_tensor_are_strides_eq(const mag_tensor_t *x, const mag_tensor_t *y) {
    return memcmp(x->strides, y->strides, sizeof(x->strides)) == 0;
}

bool mag_tensor_can_broadcast(const mag_tensor_t *small, const mag_tensor_t *big) {
    int64_t mr = mag_xmax(small->rank, big->rank);
    for (int64_t d=0; d < mr; ++d) {
        int64_t asz = d < small->rank ? small->shape[small->rank-1-d] : 1;
        int64_t bsz = d < big->rank ? big->shape[big->rank-1-d] : 1;
        if (asz != bsz && asz != 1 && bsz != 1)
            return false;
    }
    return true;
}

bool mag_tensor_is_transposed(const mag_tensor_t *t) {
    return t->strides[0] > t->strides[1];
}

bool mag_tensor_is_permuted(const mag_tensor_t *t) {
    for (int i=0; i < MAG_MAX_DIMS-1; ++i)
        if (t->strides[i] > t->strides[i+1])
            return true;
    return false;
}

bool mag_tensor_is_contiguous(const mag_tensor_t *t) {
    int64_t str = 1;
    for (int64_t d=t->rank-1; d >= 0; --d) {
        int64_t size_d = t->shape[d];
        if (size_d == 1) continue;
        if (t->strides[d] != str) return false;
        str *= size_d;
    }
    return true;
}

bool mag_tensor_can_view(const mag_tensor_t *t, const int64_t *dims, int64_t rank) {
    int64_t tmp[MAG_MAX_DIMS];
    return mag_solve_view_strides(&tmp, t->shape, t->strides, t->rank, dims, rank);
}
