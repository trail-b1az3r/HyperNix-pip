/*
 * ggml-hnx.h — HyperNix sub-1-bit tensor types for ggml / llama.cpp.
 *
 * Stock llama.cpp cannot load a HyperNix sub-bit model, and no amount of
 * header rewriting changes that: IQ0.9_L, IQ0.75_M, IQ0.5_XXXL and
 * IQ0.25_UXL are not llama.cpp quantisations with a different name, they
 * are a different arithmetic. `hyprslug-headers` can make the file
 * *identify* as something llama.cpp knows; the tensors inside it are
 * still packed the HyperNix way, and a loader that believes the header
 * reads 30 bytes as though they were a 210-byte Q3_K block.
 *
 * So this is the decoder, in C, to be compiled into llama.cpp.
 *
 * How the format works
 * --------------------
 * Below one bit per weight you cannot store a fraction of a bit. What
 * you can do is store *fewer signs than weights* and reconstruct the
 * rest. Every type here keeps one FP16 scale per 256-weight block and
 * throws the magnitudes away entirely — the scale stands in for all of
 * them. What separates the types is how many signs survive:
 *
 *   type            group  kept  block bytes  bits/weight
 *   HNX_IQ0_9           8     7           30        0.938
 *   HNX_IQ0_75          4     3           26        0.812
 *   HNX_IQ0_5           4     2           18        0.562
 *   HNX_IQ0_25         16     3            8        0.250
 *   HNX_INT1            1     1           34        1.062
 *
 * Within each group the *first* `kept` signs are the stored ones, and
 * the positions after them repeat the last stored sign. That mapping is
 * fixed and has to be: there are no bits left to describe a cleverer
 * choice, so encoder and decoder can only agree on "the first k,
 * always". Repeating rather than alternating is a deliberate choice too
 * — adjacent weights in a row correlate, so a repeat is right more
 * often than a coin.
 *
 * Bit order is LSB-first within each byte and continuous across the
 * whole block payload, matching `hypernix.quant.subbit`. Getting that
 * wrong produces a model that loads, runs, and emits fluent nonsense,
 * which is the worst possible failure mode — hence hnx_selftest, which
 * checks these decoders against vectors the Python implementation
 * generated.
 *
 * What this does not do
 * ---------------------
 * It does not make a 0.5-bit model good. Below roughly 1.5 bits per
 * weight a model stops being a slightly worse version of itself and
 * becomes a different, much worse model. This makes such a file
 * *loadable and correct*; it cannot make it accurate.
 */
#ifndef GGML_HNX_H
#define GGML_HNX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Weights per block. 256, matching the K-quant family, so a tensor that
 * divides evenly for Q4_K divides evenly here too. */
#define HNX_BLOCK_SIZE 256

/* Type ids. These match hypernix.quant.gguf.GGMLType and start at 200 to
 * stay clear of upstream: llama.cpp's own enum is well under 100 and has
 * room to grow before it reaches here. A collision would mean a stock
 * build silently reinterpreting HyperNix tensors as whatever upstream
 * added, so the gap is the safety margin. */
#define HNX_TYPE_IQ0_9   200
#define HNX_TYPE_IQ0_75  201
#define HNX_TYPE_IQ0_5   202
#define HNX_TYPE_IQ0_25  203
#define HNX_TYPE_INT1    204
/* 1.375 bits/weight: every sign, plus a 5-bit magnitude index per
 * 16-weight sub-block. The first type here whose rate buys
 * structure rather than more signs -- once all 256 signs are
 * stored there are no more to buy. */
#define HNX_TYPE_1375    207

/* The fixed-codebook pair. Every weight carries its own code into a
 * table of levels, so there is nothing reconstructed and nothing folded.
 *
 * They are here because leaving them out was worse than including them:
 * GGML_TYPE_COUNT sizes ggml's trait tables, so raising it past 205 to
 * reach 207 would leave 205 and 206 as in-range entries that are all
 * zeroes -- and `ne[0] % ggml_blck_size(type)` on a zero block size is a
 * division by zero, not a refusal. Before, such a file was cleanly
 * rejected for being out of range. */
#define HNX_TYPE_INT4    205
#define HNX_TYPE_FP2     206

/* One block of each type. Laid out to match the file exactly: an FP16
 * scale then the packed sign bits, with no padding. The static asserts
 * below are load-bearing — a compiler that padded these would read every
 * block from the wrong offset. */
typedef struct { uint16_t d; uint8_t qs[28]; } hnx_block_iq0_9;   /* 30 */
typedef struct { uint16_t d; uint8_t qs[24]; } hnx_block_iq0_75;  /* 26 */
typedef struct { uint16_t d; uint8_t qs[16]; } hnx_block_iq0_5;   /* 18 */
typedef struct { uint16_t d; uint8_t qs[6];  } hnx_block_iq0_25;  /*  8 */
typedef struct { uint16_t d; uint8_t qs[32]; } hnx_block_int1;    /* 34 */
/* 32 bytes of signs then 10 bytes holding 16 five-bit indices. */
typedef struct { uint16_t d; uint8_t qs[42]; } hnx_block_1375;    /* 44 */
typedef struct { uint16_t d; uint8_t qs[128]; } hnx_block_int4;   /* 130 */
typedef struct { uint16_t d; uint8_t qs[64]; } hnx_block_fp2;     /*  66 */

/* Everything a caller needs to handle one of these types without a
 * switch over the ids. */
typedef struct {
    int         type;
    const char *name;        /* "IQ0.9_L" etc, as hyprslug spells it */
    int         group;       /* weights covered by one code */
    int         kept;        /* signs stored per group */
    size_t      block_bytes;
    float       bits_per_weight;
    /* Sub-blocks carrying their own magnitude index, and the width of
     * that index. Zero for the sign-only types, where the block scale is
     * the whole of the magnitude -- which is why they can be appended
     * here without touching the existing table rows. */
    int         sub_blocks;
    int         sub_bits;
    /* Fixed-codebook types: every weight is levels[code] * scale.
     * NULL for the sign-and-scale family. */
    const float *levels;
    int          level_bits;
} hnx_type_info;

/* NULL for an id this build does not implement. */
const hnx_type_info *hnx_type_lookup(int type);
/* NULL for a name it does not know. Accepts the hyprslug spelling. */
const hnx_type_info *hnx_type_by_name(const char *name);
/* 1 when `type` is one of ours, so a loader can branch once. */
int hnx_is_hnx_type(int type);

/* Bytes one block of `type` occupies, or 0 for an unknown type. */
size_t hnx_block_bytes(int type);

/* Decode `nblocks` blocks of `type` from `src` into `dst`.
 *
 * `dst` must have room for nblocks * HNX_BLOCK_SIZE floats. Returns the
 * number of floats written, or 0 if the type is unknown — never a
 * partial write, so a caller that ignores the return value gets zeros
 * rather than uninitialised memory. */
size_t hnx_dequantize_rows(int type, const void *src, float *dst, size_t nblocks);

/* Decode exactly one block. `src` must hold hnx_block_bytes(type) bytes
 * and `dst` HNX_BLOCK_SIZE floats. Returns 0 on success. */
int hnx_dequantize_block(int type, const void *src, float *dst);

/* dot(x, y) where x is `nblocks` blocks of `type` and y is plain floats.
 *
 * Kept separate from dequantise-then-dot because it never materialises
 * the weights: the whole value of a sub-bit model is that its weights
 * stay small, and expanding a row to float32 to multiply it gives that
 * back at exactly the moment it matters. */
float hnx_vec_dot(int type, const void *x, const float *y, size_t nblocks);

/* Encode `nblocks * HNX_BLOCK_SIZE` floats into `dst`.
 *
 * Present so the round trip can be tested in one language, and so a C
 * caller can quantise without going through Python. `importance` may be
 * NULL, in which case the scale is the plain mean absolute value.
 * Returns bytes written, or 0 for an unknown type. */
size_t hnx_quantize_rows(int type, const float *src, void *dst, size_t nblocks,
                         const float *importance);

/* FP16 <-> float32. Exposed because the selftest needs to build blocks
 * byte-identical to Python's, and because a scale is the one value in
 * this format whose exact bits matter. */
float    hnx_fp16_to_fp32(uint16_t h);
uint16_t hnx_fp32_to_fp16(float f);

#ifdef __cplusplus
}
#endif

#endif /* GGML_HNX_H */
