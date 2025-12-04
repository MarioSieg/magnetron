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

typedef struct mag_discrete_sample_pair_t {
    float score;
    int64_t idx;
} mag_discrete_sample_pair_t;

static int mag_discrete_sample_pair_cmp(const void *a, const void *b) {
    const mag_discrete_sample_pair_t *A = a;
    const mag_discrete_sample_pair_t *B = b;
    return A->score < B->score ? 1 : A->score > B->score ? -1 : 0;
}

#define mag_gen_stub_multinomial(T, TF, CVT) \
    static void MAG_HOTPROC mag_multinomial_##TF(const mag_kernel_payload_t *payload) { \
        mag_tensor_t *r = mag_cmd_out(0); \
        const mag_tensor_t *x = mag_cmd_in(0); \
        mag_assert2(r->dtype == MAG_DTYPE_INT64); \
        int64_t *br = (int64_t *)mag_tensor_data_ptr_mut(r); \
        const T *bx = (const T *)mag_tensor_data_ptr(x); \
        int64_t num_samples = mag_op_attr_unwrap_int64(mag_cmd_attr(0)); \
        mag_philox4x32_stream_t *rng = payload->prng; \
        int64_t K = x->coords.shape[x->coords.rank-1]; \
        if (mag_unlikely(K <= 0)) return; \
        int64_t B = x->numel / K; \
        int64_t tc = payload->thread_num; \
        int64_t ti = payload->thread_idx; \
        int64_t chunk = (B + tc - 1)/tc; \
        int64_t ra = ti*chunk; \
        int64_t rb = mag_xmin(ra + chunk, B); \
        for (int64_t b=ra; b < rb; ++b) { \
            const T *w = bx + b*K; \
            int64_t *o = br + b*num_samples; \
            float sumw = .0f; \
            int64_t nnz = 0; \
            for (int64_t i=0; i < K; ++i) { \
                float wi = CVT(w[i]); \
                if (!isfinite(wi) || wi <= .0f) wi = .0f; \
                sumw += wi; \
                if (wi > .0f) ++nnz; \
            } \
            if (!(sumw > .0f) || nnz == 0) { \
                for (int64_t s=0; s < num_samples; ++s) o[s] = -1; \
                continue; \
            } \
            int64_t k = num_samples; \
            if (k > nnz) k = nnz; \
            if (mag_unlikely(k <= 0)) { \
                for (int64_t s=0; s < num_samples; ++s) o[s] = -1; \
                continue; \
            } \
            mag_discrete_sample_pair_t *arr = (mag_discrete_sample_pair_t*)alloca(nnz*sizeof(*arr)); \
            int64_t m=0; \
            for (int64_t i=0; i < K; ++i) { \
                float wi = CVT(w[i]); \
                if (mag_unlikely(!isfinite(wi) || wi <= .0f)) continue; \
                float u = mag_philox4x32_next_float32(rng); \
                float g = -logf(-logf(u)); \
                arr[m].score = logf(wi) + g; \
                arr[m].idx = i; \
                ++m; \
            } \
            qsort(arr, (size_t)m, sizeof(*arr), mag_discrete_sample_pair_cmp); \
            for (int64_t s=0; s < k; ++s) o[s] = arr[s].idx; \
            for (int64_t s=k; s < num_samples; ++s) o[s] = -1; \
        } \
    }

mag_gen_stub_multinomial(float, float32, mag_cvt_nop)
mag_gen_stub_multinomial(mag_float16_t, float16, mag_float16_to_float32)
