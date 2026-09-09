/*
 * ggml-hnx-cuda.h — the CUDA entry points, for the ggml backend to call.
 *
 * Declared in plain C so ggml-cuda.cu can call them without seeing the
 * templates, and so a build with GGML_HNX_CUDA off can leave the whole
 * file out with nothing but a link-time absence.
 *
 * `row_stride_bytes` is passed rather than computed because ggml pads
 * rows: a tensor's nb[1] is not always nblocks * block_bytes, and
 * assuming it is reads the second row from the wrong offset -- which
 * produces a model that runs and is subtly wrong, the failure this
 * directory is arranged around.
 */
#ifndef GGML_HNX_CUDA_H
#define GGML_HNX_CUDA_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dst[row] = dot(row of x, y), for nrows rows of nblocks blocks each. */
#define HNX_CUDA_DECLARE(suffix)                                              \
    void hnx_cuda_mul_mat_vec_##suffix(                                       \
        const void* x, const float* y, float* dst, int nblocks, int nrows,    \
        int row_stride_bytes, cudaStream_t stream);                           \
    void hnx_cuda_dequantize_##suffix(                                        \
        const void* x, float* dst, int nblocks, cudaStream_t stream);

HNX_CUDA_DECLARE(iq0_9)
HNX_CUDA_DECLARE(iq0_75)
HNX_CUDA_DECLARE(iq0_5)
HNX_CUDA_DECLARE(iq0_25)
HNX_CUDA_DECLARE(int1)

#undef HNX_CUDA_DECLARE

#ifdef __cplusplus
}
#endif

#endif /* GGML_HNX_CUDA_H */
