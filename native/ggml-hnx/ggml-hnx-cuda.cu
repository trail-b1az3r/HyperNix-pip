/*
 * ggml-hnx-cuda.cu — the sub-bit types on a GPU.
 *
 * Beta 1 shipped these types CPU-only, and said so: a sub-bit tensor was
 * dequantised on the host and the rest of the graph ran wherever it was
 * sent. That works and it is slow in the one way that matters — the
 * whole point of a 0.5-bit model is that it fits on a small card, and
 * paying a host round trip per matmul gives back the speed you bought
 * the memory with.
 *
 * These are the kernels. Two per type, mirroring the CPU side:
 *
 *   dequantize_block  writes HNX_BLOCK_SIZE floats per block, for the
 *                     paths that need a materialised tensor (a copy, a
 *                     conversion, an op with no fused kernel)
 *   mul_mat_vec       the one that actually matters: dot a quantised row
 *                     with an activation vector without materialising
 *                     the row at all
 *
 * Why the dot product is nearly free here
 * ---------------------------------------
 * Every weight is ±scale. So a block reduces to `scale * Σ(±y)`: one add
 * or subtract per weight, one multiply per block, and *no* weight ever
 * loaded as a float. A 30-byte block covers 256 weights, which means a
 * row that would be 1 KB in F32 is 30 bytes of L2 — the arithmetic is
 * trivial and the whole kernel is bound by reading the activations
 * rather than the weights. That is the compensation for throwing the
 * magnitudes away, and it is the reason these types are worth having at
 * all beyond the file size.
 *
 * One warp per row
 * ----------------
 * 32 lanes each take a stride of blocks, accumulate in a register, and
 * reduce with __shfl_down_sync. No shared memory and no __syncthreads:
 * a warp-level reduction needs neither, and the block count per row
 * (thousands, for a 7B model) is large enough that the stride keeps
 * every lane busy.
 *
 * Correctness
 * -----------
 * The bit order, group stride and repeat rule are the CPU
 * implementation's, and `hnx_selftest --cuda` checks the two agree on
 * the same vectors the CPU side is checked against. They have to: a
 * kernel that decodes one bit differently produces a model that runs at
 * full speed and talks nonsense, which is the failure mode this whole
 * directory is arranged around catching.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "ggml-hnx.h"

// One warp per row. 32 rather than a larger block because the reduction
// is warp-level: a wider block would need shared memory and a barrier to
// combine partial sums, for no gain on a kernel this memory-bound.
#define HNX_CUDA_WARP 32

namespace {

/* The decode step, shared by both kernels.
 *
 * `group` and `kept` are template parameters and not arguments so the
 * inner loops unroll and the modulo arithmetic folds away at compile
 * time. That matters: with runtime values this is a loop with a
 * data-dependent trip count around two bit extractions, and it measured
 * roughly 3x slower.
 */
template <int group, int kept>
__device__ __forceinline__ float hnx_signed_sum(
    const uint8_t* __restrict__ payload, const float* __restrict__ y) {
    float total = 0.0f;
    int bit = 0;
#pragma unroll
    for (int out = 0; out < HNX_BLOCK_SIZE; out += group) {
        float sign = 1.0f;
#pragma unroll
        for (int i = 0; i < kept; i++) {
            // LSB-first within the byte, continuous across the payload —
            // the CPU side's order, and the thing that must not drift.
            sign = ((payload[bit >> 3] >> (bit & 7)) & 1) ? 1.0f : -1.0f;
            total += sign * y[out + i];
            bit++;
        }
        // The group's remaining positions repeat the last stored sign.
        // Summed and multiplied once rather than added in a loop: the
        // sign is constant across the tail by construction.
        if (group > kept) {
            float tail = 0.0f;
#pragma unroll
            for (int i = kept; i < group; i++) tail += y[out + i];
            total += sign * tail;
        }
    }
    return total;
}

template <int group, int kept>
__device__ __forceinline__ void hnx_expand(
    const uint8_t* __restrict__ payload, float scale, float* __restrict__ dst) {
    int bit = 0;
#pragma unroll
    for (int out = 0; out < HNX_BLOCK_SIZE; out += group) {
        float last = 0.0f;
#pragma unroll
        for (int i = 0; i < kept; i++) {
            last = ((payload[bit >> 3] >> (bit & 7)) & 1) ? scale : -scale;
            dst[out + i] = last;
            bit++;
        }
#pragma unroll
        for (int i = kept; i < group; i++) dst[out + i] = last;
    }
}

/* dot(row, y) for one row of `nblocks` blocks.
 *
 * `block_bytes` is a template parameter too, so the per-block pointer
 * advance is a compile-time constant multiply rather than a runtime one.
 */
template <int group, int kept, int block_bytes>
__global__ void hnx_mul_mat_vec(
    const uint8_t* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ dst, int nblocks, int nrows, int row_stride_bytes) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int lane = threadIdx.x;

    const uint8_t* row_base = x + (size_t)row * (size_t)row_stride_bytes;

    // Accumulated in float per lane and reduced in float. The terms
    // within a lane are all the same magnitude, which is the case where
    // float32 loses the most — but a double accumulator here costs about
    // 8x on consumer cards, where FP64 is 1/32 rate, and the reduction
    // across 32 lanes recovers most of the precision a serial float sum
    // would lose. The CPU side accumulates in double and the selftest
    // compares them with a relative tolerance for exactly this reason.
    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += HNX_CUDA_WARP) {
        const uint8_t* block = row_base + (size_t)b * block_bytes;
        // __half rather than a memcpy: the scale is 2 bytes at a 2-byte
        // aligned offset in every one of these layouts, and __half2float
        // is a single instruction.
        const __half d = *reinterpret_cast<const __half*>(block);
        acc += __half2float(d) *
               hnx_signed_sum<group, kept>(block + 2, y + (size_t)b * HNX_BLOCK_SIZE);
    }

#pragma unroll
    for (int offset = HNX_CUDA_WARP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0) dst[row] = acc;
}

/* Materialise `nblocks` blocks as floats. One thread per block: the
 * expansion is 256 sequential stores and there is nothing to reduce.
 */
template <int group, int kept, int block_bytes>
__global__ void hnx_dequantize(
    const uint8_t* __restrict__ x, float* __restrict__ dst, int nblocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t* block = x + (size_t)b * block_bytes;
    const __half d = *reinterpret_cast<const __half*>(block);
    hnx_expand<group, kept>(block + 2, __half2float(d),
                            dst + (size_t)b * HNX_BLOCK_SIZE);
}

}  // namespace

// The five types, as (group, kept, block_bytes). Kept beside the launchers
// rather than derived from hnx_type_lookup at runtime, because these have
// to be compile-time constants for the templates above — and a mismatch
// with the CPU table is what hnx_selftest --cuda exists to catch.
#define HNX_CUDA_LAUNCHERS(suffix, group, kept, bytes)                        \
    extern "C" void hnx_cuda_mul_mat_vec_##suffix(                            \
        const void* x, const float* y, float* dst, int nblocks, int nrows,    \
        int row_stride_bytes, cudaStream_t stream) {                          \
        if (nrows <= 0 || nblocks <= 0) return;                               \
        hnx_mul_mat_vec<group, kept, bytes>                                   \
            <<<nrows, HNX_CUDA_WARP, 0, stream>>>(                            \
                (const uint8_t*) x, y, dst, nblocks, nrows, row_stride_bytes);\
    }                                                                         \
                                                                              \
    extern "C" void hnx_cuda_dequantize_##suffix(                             \
        const void* x, float* dst, int nblocks, cudaStream_t stream) {        \
        if (nblocks <= 0) return;                                             \
        const int threads = 128;                                              \
        const int blocks = (nblocks + threads - 1) / threads;                 \
        hnx_dequantize<group, kept, bytes>                                    \
            <<<blocks, threads, 0, stream>>>((const uint8_t*) x, dst,         \
                                             nblocks);                        \
    }

HNX_CUDA_LAUNCHERS(iq0_9,  8,  7, 30)
HNX_CUDA_LAUNCHERS(iq0_75, 4,  3, 26)
HNX_CUDA_LAUNCHERS(iq0_5,  4,  2, 18)
HNX_CUDA_LAUNCHERS(iq0_25, 16, 3, 8)
HNX_CUDA_LAUNCHERS(int1,   1,  1, 34)

#undef HNX_CUDA_LAUNCHERS
