/*
 * ggml-hnx.c — see ggml-hnx.h for the format and why this exists.
 *
 * Deliberately plain C99 with no ggml headers included. Two reasons:
 * it can be compiled and tested on its own (which is how the bit order
 * was verified against Python rather than assumed), and it does not
 * have to be rewritten each time upstream moves a header. The glue that
 * registers these with ggml's type table lives in patches/, where it is
 * small enough to re-apply by hand when a rebase needs it.
 */
#include "ggml-hnx.h"

#include <math.h>
#include <string.h>

/* Order matters only for readability; lookup is a linear scan over six
 * entries, which is faster than anything cleverer at this size.
 *
 * The trailing fields are sub_blocks/sub_bits then the codebook
 * pointer and its width. Written out rather than left to C's
 * zero-fill so -Werror builds stay quiet and the three families --
 * signs only, signs plus sub-magnitude, fixed codebook -- are
 * visible as three shapes in one table.
 * Written out rather than left to C's zero-fill so -Werror builds
 * stay quiet and the distinction is visible in the table. */
/* The codebooks, matching hypernix.quant.lowbit.CODECS exactly. INT4 is
 * asymmetric because two's complement is: -8 exists and +8 does not. */
static const float HNX_LEVELS_INT4[16] = {
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
     0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
};
static const float HNX_LEVELS_FP2[4] = { -2.0f, -1.0f, 1.0f, 2.0f };
/* INT2 is the two's complement 2-bit range, so it has the zero FP2
 * deliberately does not -- and the same asymmetry INT4 has, for the same
 * reason: -2 exists and +2 does not. */
static const float HNX_LEVELS_INT2[4] = { -2.0f, -1.0f, 0.0f, 1.0f };
/* -128..127, built by macro rather than transcribed. 256 hand-written
 * rows is a table nobody can proofread, and a lazily-filled one would
 * need a guard on first use from whichever thread got there first. This
 * is a compile-time constant with neither problem. */
#define HNX_I8_8(b)  (float)(b),        (float)((b) +  1), (float)((b) +  2), \
                     (float)((b) +  3), (float)((b) +  4), (float)((b) +  5), \
                     (float)((b) +  6), (float)((b) +  7)
#define HNX_I8_64(b) HNX_I8_8(b),        HNX_I8_8((b) +  8), \
                     HNX_I8_8((b) + 16), HNX_I8_8((b) + 24), \
                     HNX_I8_8((b) + 32), HNX_I8_8((b) + 40), \
                     HNX_I8_8((b) + 48), HNX_I8_8((b) + 56)
static const float HNX_LEVELS_INT8[256] = {
    HNX_I8_64(-128), HNX_I8_64(-64), HNX_I8_64(0), HNX_I8_64(64),
};

static const float HNX_LEVELS_INT3[8] = {
    -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f,
};
/* E4M3 by byte, generated from hypernix.quant.lowbit.e4m3_value and
 * checked against it by the selftest vectors. 0x7f and 0xff are the NaN
 * codes and read as zero. */
static const float HNX_LEVELS_FP8[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,  /* 0x00 */
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,  /* 0x08 */
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,  /* 0x10 */
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,  /* 0x18 */
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,  /* 0x20 */
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,  /* 0x28 */
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,  /* 0x30 */
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,  /* 0x38 */
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,  /* 0x40 */
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,  /* 0x48 */
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,  /* 0x50 */
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,  /* 0x58 */
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,  /* 0x60 */
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,  /* 0x68 */
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,  /* 0x70 */
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,  /* 0x78 */
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,  /* 0x80 */
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,  /* 0x88 */
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,  /* 0x90 */
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,  /* 0x98 */
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,  /* 0xa0 */
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,  /* 0xa8 */
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,  /* 0xb0 */
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,  /* 0xb8 */
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,  /* 0xc0 */
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,  /* 0xc8 */
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,  /* 0xd0 */
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,  /* 0xd8 */
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,  /* 0xe0 */
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,  /* 0xe8 */
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,  /* 0xf0 */
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f,  /* 0xf8 */
};

static const hnx_type_info HNX_TYPES[] = {
    { HNX_TYPE_IQ0_9,  "IQ0.9_L",     8, 7, sizeof(hnx_block_iq0_9),  0.9375f,  0, 0 , NULL, 0 },
    { HNX_TYPE_IQ0_75, "IQ0.75_M",    4, 3, sizeof(hnx_block_iq0_75), 0.8125f,  0, 0 , NULL, 0 },
    { HNX_TYPE_IQ0_5,  "IQ0.5_XXXL",  4, 2, sizeof(hnx_block_iq0_5),  0.5625f,  0, 0 , NULL, 0 },
    { HNX_TYPE_IQ0_25, "IQ0.25_UXL", 16, 3, sizeof(hnx_block_iq0_25), 0.2500f,  0, 0 , NULL, 0 },
    { HNX_TYPE_INT1,   "INT1",        1, 1, sizeof(hnx_block_int1),   1.0625f,  0, 0 , NULL, 0 },
    { HNX_TYPE_1375,   "HNX_1375BIT", 1, 1, sizeof(hnx_block_1375),   1.3750f, 16, 5, NULL, 0 },
    { HNX_TYPE_INT4,   "INT4",        1, 1, sizeof(hnx_block_int4),   4.0625f,  0, 0, HNX_LEVELS_INT4, 4 },
    { HNX_TYPE_FP2,    "FP2",         1, 1, sizeof(hnx_block_fp2),    2.0625f,  0, 0, HNX_LEVELS_FP2,  2 },
    { HNX_TYPE_INT8,   "INT8",        1, 1, sizeof(hnx_block_int8),   8.0625f,  0, 0, HNX_LEVELS_INT8, 8 },
    { HNX_TYPE_INT2,   "INT2",        1, 1, sizeof(hnx_block_int2),   2.0625f,  0, 0, HNX_LEVELS_INT2, 2 },
    { HNX_TYPE_INT3,   "INT3",        1, 1, sizeof(hnx_block_int3),   3.0625f,  0, 0, HNX_LEVELS_INT3, 3 },
    { HNX_TYPE_FP8,    "FP8",         1, 1, sizeof(hnx_block_fp8),    8.0625f,  0, 0, HNX_LEVELS_FP8,  8 },
};
static const size_t HNX_TYPE_COUNT = sizeof(HNX_TYPES) / sizeof(HNX_TYPES[0]);

/* The structs must be exactly as wide as the file's blocks. A compiler
 * that inserted padding would make every block after the first read
 * from the wrong offset -- and the model would still load, and still
 * generate text, just wrong text. Caught here instead. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(hnx_block_iq0_9)  == 30, "IQ0.9 block must be 30 bytes");
_Static_assert(sizeof(hnx_block_iq0_75) == 26, "IQ0.75 block must be 26 bytes");
_Static_assert(sizeof(hnx_block_iq0_5)  == 18, "IQ0.5 block must be 18 bytes");
_Static_assert(sizeof(hnx_block_iq0_25) ==  8, "IQ0.25 block must be 8 bytes");
_Static_assert(sizeof(hnx_block_int1)   == 34, "INT1 block must be 34 bytes");
_Static_assert(sizeof(hnx_block_1375)   == 44, "HNX_1375BIT block must be 44 bytes");
_Static_assert(sizeof(hnx_block_int4)   == 130, "INT4 block must be 130 bytes");
_Static_assert(sizeof(hnx_block_fp2)    == 66, "FP2 block must be 66 bytes");
_Static_assert(sizeof(hnx_block_int8)   == 258, "INT8 block must be 258 bytes");
_Static_assert(sizeof(hnx_block_int2)   == 66, "INT2 block must be 66 bytes");
_Static_assert(sizeof(hnx_block_int3)   == 98, "INT3 block must be 98 bytes");
_Static_assert(sizeof(hnx_block_fp8)    == 258, "FP8 block must be 258 bytes");
_Static_assert(sizeof(hnx_block_q8_k)   == HNX_Q8K_BYTES, "Q8_K block must be 292 bytes");
#endif

const hnx_type_info *hnx_type_lookup(int type) {
    for (size_t i = 0; i < HNX_TYPE_COUNT; i++) {
        if (HNX_TYPES[i].type == type) return &HNX_TYPES[i];
    }
    return NULL;
}

const hnx_type_info *hnx_type_by_name(const char *name) {
    if (name == NULL) return NULL;
    for (size_t i = 0; i < HNX_TYPE_COUNT; i++) {
        if (strcmp(HNX_TYPES[i].name, name) == 0) return &HNX_TYPES[i];
    }
    return NULL;
}

int hnx_is_hnx_type(int type) { return hnx_type_lookup(type) != NULL; }

size_t hnx_block_bytes(int type) {
    const hnx_type_info *info = hnx_type_lookup(type);
    return info ? info->block_bytes : 0;
}

/* --- FP16 ---------------------------------------------------------------
 *
 * Written out rather than using _Float16 or a compiler builtin: this has
 * to give bit-identical results to Python's struct.pack('<e', ...) on
 * every target, including ones without hardware half support, and a
 * scale that differs in the last bit changes every weight in its block.
 */
float hnx_fp16_to_fp32(uint16_t h) {
    const uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1Fu;
    const uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;                       /* +/- zero */
        } else {
            /* Subnormal: normalise it by hand. Left-shift the mantissa
             * until its leading 1 appears, decrementing the exponent to
             * match, then drop that leading 1. */
            uint32_t m = mantissa;
            int shift = 0;
            while ((m & 0x400u) == 0) { m <<= 1; shift++; }
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - shift + 1) << 23) | (m << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);   /* inf / NaN */
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t hnx_fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        /* inf or NaN. A NaN must stay a NaN: zeroing the mantissa here
         * would turn it into an infinity, and an infinite scale
         * dequantises a whole block to +/-inf instead of NaN. */
        return (uint16_t)(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    }
    if (exponent >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);              /* overflow to inf */
    }
    if (exponent <= 0) {
        /* Subnormal or zero. Round to nearest even, as Python does. */
        if (exponent < -10) return sign;
        mantissa |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t value = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (value & 1u))) value++;
        return (uint16_t)(sign | value);
    }
    /* Normal, round to nearest even. */
    uint16_t out = (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
    const uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (out & 1u))) out++;
    return out;
}

/* --- Decode -------------------------------------------------------------
 *
 * One implementation over the whole family, driven by group/kept from
 * the type table. Four near-identical unrolled copies were written first
 * and the IQ0.25 one had the bit cursor advancing by `group` instead of
 * `kept` -- a bug that produces a model which loads and talks nonsense.
 * One loop that cannot disagree with itself is worth more here than the
 * unrolling.
 */
/* Every sign stored, and one magnitude per sub-block.
 *
 * The signs occupy one bit each, so they end on a byte boundary and the
 * indices start at payload[HNX_BLOCK_SIZE/8] -- no bit cursor has to
 * cross between the two halves. Kept deliberately separate from
 * hnx_decode: folding a second layout into that loop is how the IQ0.25
 * cursor bug happened.
 */
static void hnx_decode_sub(const uint8_t *payload, float scale,
                           int sub_blocks, int sub_bits, float *dst) {
    const int sub_size = HNX_BLOCK_SIZE / sub_blocks;
    const int levels   = (1 << sub_bits) - 1;
    const uint8_t *indices = payload + (HNX_BLOCK_SIZE / 8);
    size_t bit = 0;
    for (int s = 0; s < sub_blocks; s++) {
        int index = 0;
        for (int b = 0; b < sub_bits; b++) {
            index |= ((indices[bit >> 3] >> (bit & 7)) & 1) << b;
            bit++;
        }
        const float magnitude = scale * (float)index / (float)levels;
        const size_t base = (size_t)s * (size_t)sub_size;
        for (int i = 0; i < sub_size; i++) {
            const size_t at = base + (size_t)i;
            const int set = (payload[at >> 3] >> (at & 7)) & 1;
            dst[at] = set ? magnitude : -magnitude;
        }
    }
}

/* Fixed codebook: one code per weight, LSB-first, into `levels`. */
static void hnx_decode_codebook(const uint8_t *payload, float scale,
                                const float *levels, int bits, float *dst) {
    const int mask = (1 << bits) - 1;
    size_t bit = 0;
    for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) {
        int code = 0;
        for (int b = 0; b < bits; b++) {
            code |= ((payload[bit >> 3] >> (bit & 7)) & 1) << b;
            bit++;
        }
        dst[i] = levels[code & mask] * scale;
    }
}

static void hnx_decode(const uint8_t *payload, float scale, int group, int kept,
                       float *dst) {
    size_t bit = 0;
    size_t out = 0;
    while (out < HNX_BLOCK_SIZE) {
        float last = 0.0f;
        for (int i = 0; i < kept; i++) {
            /* LSB-first within the byte, continuous across the payload. */
            const int set = (payload[bit >> 3] >> (bit & 7)) & 1;
            last = set ? scale : -scale;
            dst[out + (size_t)i] = last;
            bit++;
        }
        /* The rest of the group repeats the last stored sign. */
        for (int i = kept; i < group; i++) {
            dst[out + (size_t)i] = last;
        }
        out += (size_t)group;
    }
}

int hnx_dequantize_block(int type, const void *src, float *dst) {
    const hnx_type_info *info = hnx_type_lookup(type);
    if (info == NULL || src == NULL || dst == NULL) return -1;
    const uint8_t *bytes = (const uint8_t *)src;
    uint16_t raw;
    memcpy(&raw, bytes, sizeof(raw));    /* memcpy, not a cast: the file
                                          * gives no alignment guarantee */
    if (info->levels) {
        hnx_decode_codebook(bytes + 2, hnx_fp16_to_fp32(raw), info->levels,
                            info->level_bits, dst);
    } else if (info->sub_blocks) {
        hnx_decode_sub(bytes + 2, hnx_fp16_to_fp32(raw),
                       info->sub_blocks, info->sub_bits, dst);
    } else {
        hnx_decode(bytes + 2, hnx_fp16_to_fp32(raw), info->group, info->kept, dst);
    }
    return 0;
}

size_t hnx_dequantize_rows(int type, const void *src, float *dst, size_t nblocks) {
    const hnx_type_info *info = hnx_type_lookup(type);
    if (info == NULL || src == NULL || dst == NULL) return 0;
    const uint8_t *in = (const uint8_t *)src;
    for (size_t b = 0; b < nblocks; b++) {
        uint16_t raw;
        memcpy(&raw, in, sizeof(raw));
        if (info->levels) {
            hnx_decode_codebook(in + 2, hnx_fp16_to_fp32(raw), info->levels,
                                info->level_bits, dst + b * HNX_BLOCK_SIZE);
        } else if (info->sub_blocks) {
            hnx_decode_sub(in + 2, hnx_fp16_to_fp32(raw), info->sub_blocks,
                           info->sub_bits, dst + b * HNX_BLOCK_SIZE);
        } else {
            hnx_decode(in + 2, hnx_fp16_to_fp32(raw), info->group, info->kept,
                       dst + b * HNX_BLOCK_SIZE);
        }
        in += info->block_bytes;
    }
    return nblocks * HNX_BLOCK_SIZE;
}

/* --- Dot product --------------------------------------------------------
 *
 * Never materialises the weights. Since every weight is +/-scale, the
 * whole block reduces to scale * sum(+/-y_i): one add or subtract per
 * weight and one multiply per block. That is the compensation for
 * throwing the magnitudes away -- these types are cheap to multiply
 * precisely because there is so little left of them.
 */
float hnx_vec_dot(int type, const void *x, const float *y, size_t nblocks) {
    const hnx_type_info *info = hnx_type_lookup(type);
    if (info == NULL || x == NULL || y == NULL) return 0.0f;
    const uint8_t *in = (const uint8_t *)x;
    const int group = info->group;
    const int kept  = info->kept;

    if (info->levels) {
        /* Nothing to fold: every weight has its own code, so this is the
         * ordinary dot product with a table lookup in it. */
        const int bits = info->level_bits;
        double total = 0.0;
        for (size_t b = 0; b < nblocks; b++) {
            uint16_t raw;
            memcpy(&raw, in, sizeof(raw));
            const float scale = hnx_fp16_to_fp32(raw);
            const uint8_t *payload = in + 2;
            const float *row = y + b * HNX_BLOCK_SIZE;
            double acc = 0.0;
            size_t bit = 0;
            for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) {
                int code = 0;
                for (int k = 0; k < bits; k++) {
                    code |= ((payload[bit >> 3] >> (bit & 7)) & 1) << k;
                    bit++;
                }
                acc += (double)info->levels[code] * (double)row[i];
            }
            total += (double)scale * acc;
            in += info->block_bytes;
        }
        return (float)total;
    }

    if (info->sub_blocks) {
        /* Every weight is +/- its sub-block's magnitude, so the block
         * factors the same way the sign-only types do -- just sixteen
         * times instead of once:
         *
         *   sum_i w_i y_i  ==  sum_s m_s * sum_{i in s} (+/-) y_i
         *
         * One multiply per sub-block, an add or subtract per weight, and
         * the weights are still never materialised. */
        const int sub_size = HNX_BLOCK_SIZE / info->sub_blocks;
        const int levels   = (1 << info->sub_bits) - 1;
        double total = 0.0;
        for (size_t b = 0; b < nblocks; b++) {
            uint16_t raw;
            memcpy(&raw, in, sizeof(raw));
            const float scale = hnx_fp16_to_fp32(raw);
            const uint8_t *payload = in + 2;
            const uint8_t *indices = payload + (HNX_BLOCK_SIZE / 8);
            const float *row = y + b * HNX_BLOCK_SIZE;
            size_t bit = 0;
            for (int s = 0; s < info->sub_blocks; s++) {
                int index = 0;
                for (int k = 0; k < info->sub_bits; k++) {
                    index |= ((indices[bit >> 3] >> (bit & 7)) & 1) << k;
                    bit++;
                }
                const size_t base = (size_t)s * (size_t)sub_size;
                double signed_sum = 0.0;
                for (int i = 0; i < sub_size; i++) {
                    const size_t at = base + (size_t)i;
                    const int set = (payload[at >> 3] >> (at & 7)) & 1;
                    signed_sum += set ? (double)row[at] : -(double)row[at];
                }
                total += (double)scale * (double)index / (double)levels * signed_sum;
            }
            in += info->block_bytes;
        }
        return (float)total;
    }

    /* Accumulated in double. A 7B row is thousands of blocks and the
     * terms are all the same magnitude, which is the case where float32
     * accumulation loses the most. */
    double total = 0.0;
    for (size_t b = 0; b < nblocks; b++) {
        uint16_t raw;
        memcpy(&raw, in, sizeof(raw));
        const float scale = hnx_fp16_to_fp32(raw);
        const uint8_t *payload = in + 2;
        const float *row = y + b * HNX_BLOCK_SIZE;

        double signed_sum = 0.0;
        size_t bit = 0;
        size_t out = 0;
        while (out < HNX_BLOCK_SIZE) {
            double tail = 0.0;
            int sign = 1;
            for (int i = 0; i < kept; i++) {
                sign = ((payload[bit >> 3] >> (bit & 7)) & 1) ? 1 : -1;
                signed_sum += sign * (double)row[out + (size_t)i];
                bit++;
            }
            for (int i = kept; i < group; i++) {
                tail += (double)row[out + (size_t)i];
            }
            signed_sum += sign * tail;
            out += (size_t)group;
        }
        total += (double)scale * signed_sum;
        in += info->block_bytes;
    }
    return (float)total;
}

/* --- Encode -------------------------------------------------------------
 *
 * The scale is the importance-weighted *mean* absolute value, which
 * minimises weighted squared error for a sign-and-scale code. Not the
 * maximum: that minimises a different thing and makes every
 * reconstruction systematically too large.
 */
/* The codebook types: scale from the block's peak, then each weight to
 * the nearest level. Python searches seventeen shrink factors for the
 * scale and so lands slightly closer; this is the plain version, for a C
 * caller and for the selftest's round trip, and every file it writes
 * decodes the same way Python's do. Before, the codebook types fell
 * through to the sign-writing loop below and came out as noise. */
static void hnx_quantize_codebook(const hnx_type_info *info, const float *src,
                                  uint8_t *out, size_t nblocks) {
    const int ncodes = 1 << info->level_bits;
    float peak_level = 0.0f;
    for (int c = 0; c < ncodes; c++) {
        if (fabsf(info->levels[c]) > peak_level) peak_level = fabsf(info->levels[c]);
    }
    for (size_t b = 0; b < nblocks; b++) {
        const float *w = src + b * HNX_BLOCK_SIZE;
        float amax = 0.0f;
        for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) {
            if (isfinite(w[i]) && fabsf(w[i]) > amax) amax = fabsf(w[i]);
        }
        const uint16_t d = hnx_fp32_to_fp16(peak_level > 0.0f ? amax / peak_level : 0.0f);
        const float scale = hnx_fp16_to_fp32(d);
        memcpy(out, &d, sizeof(d));
        uint8_t *payload = out + 2;
        memset(payload, 0, info->block_bytes - 2);
        size_t bit = 0;
        for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) {
            const float target = (scale > 0.0f && isfinite(w[i])) ? w[i] / scale : 0.0f;
            int best = 0;
            float best_error = INFINITY;
            for (int c = 0; c < ncodes; c++) {
                /* Strict < keeps the first of equal levels, so a zero
                 * is always 0x00: FP8 never writes -0 (0x80) or a NaN
                 * code (0x7f, 0xff), whose table entries are zero too. */
                const float error = fabsf(info->levels[c] - target);
                if (error < best_error) { best_error = error; best = c; }
            }
            for (int k = 0; k < info->level_bits; k++) {
                if ((best >> k) & 1) payload[bit >> 3] |= (uint8_t)(1u << (bit & 7));
                bit++;
            }
        }
        out += info->block_bytes;
    }
}

size_t hnx_quantize_rows(int type, const float *src, void *dst, size_t nblocks,
                         const float *importance) {
    const hnx_type_info *info = hnx_type_lookup(type);
    if (info == NULL || src == NULL || dst == NULL) return 0;
    uint8_t *out = (uint8_t *)dst;

    if (info->levels) {
        hnx_quantize_codebook(info, src, out, nblocks);
        return nblocks * info->block_bytes;
    }

    for (size_t b = 0; b < nblocks; b++) {
        const float *w = src + b * HNX_BLOCK_SIZE;
        const float *imp = importance ? importance + b * HNX_BLOCK_SIZE : NULL;

        double magnitude = 0.0;
        double divisor = 0.0;
        for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) {
            const double weight = imp ? (imp[i] > 0.0f ? (double)imp[i] : 0.0) : 1.0;
            magnitude += fabs((double)w[i]) * weight;
            divisor += weight;
        }
        double scale = divisor > 0.0 ? magnitude / divisor : 0.0;
        /* A block containing an inf gives an infinite scale, which
         * serialises and then dequantises the whole block to NaN. A
         * poisoned tensor should degrade to zeros, not spread. */
        if (!isfinite(scale)) scale = 0.0;

        const uint16_t d = hnx_fp32_to_fp16((float)scale);
        memcpy(out, &d, sizeof(d));
        uint8_t *payload = out + 2;
        memset(payload, 0, info->block_bytes - 2);

        size_t bit = 0;
        for (size_t start = 0; start < HNX_BLOCK_SIZE; start += (size_t)info->group) {
            for (int i = 0; i < info->kept; i++) {
                /* >= 0, so a zero weight stores a positive sign -- the
                 * same tie-break Python makes. */
                if (w[start + (size_t)i] >= 0.0f) {
                    payload[bit >> 3] |= (uint8_t)(1u << (bit & 7));
                }
                bit++;
            }
        }
        out += info->block_bytes;
    }
    return nblocks * info->block_bytes;
}

/* --- Q8_K as a weight -----------------------------------------------------
 *
 * Read field by field with memcpy rather than through the struct, so an
 * unaligned tensor in a mmapped file is read correctly on every target.
 */
float hnx_vec_dot_q8_k(const void *x, const void *y, size_t nblocks) {
    if (x == NULL || y == NULL) return 0.0f;
    const uint8_t *a = (const uint8_t *)x;
    const uint8_t *b = (const uint8_t *)y;
    double total = 0.0;
    for (size_t k = 0; k < nblocks; k++) {
        float da, db;
        memcpy(&da, a, sizeof(da));
        memcpy(&db, b, sizeof(db));
        const int8_t *qa = (const int8_t *)(a + 4);
        const int8_t *qb = (const int8_t *)(b + 4);
        int32_t acc = 0;
        for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) acc += (int32_t)qa[i] * (int32_t)qb[i];
        total += (double)da * (double)db * (double)acc;
        a += HNX_Q8K_BYTES;
        b += HNX_Q8K_BYTES;
    }
    return (float)total;
}

void hnx_dequantize_q8_k(const void *src, float *dst, size_t nblocks) {
    if (src == NULL || dst == NULL) return;
    const uint8_t *in = (const uint8_t *)src;
    for (size_t k = 0; k < nblocks; k++) {
        float d;
        memcpy(&d, in, sizeof(d));
        const int8_t *q = (const int8_t *)(in + 4);
        for (size_t i = 0; i < HNX_BLOCK_SIZE; i++) dst[k * HNX_BLOCK_SIZE + i] = d * (float)q[i];
        in += HNX_Q8K_BYTES;
    }
}
