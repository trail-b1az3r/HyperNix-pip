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

/* Order matters only for readability; lookup is a linear scan over five
 * entries, which is faster than anything cleverer at this size. */
static const hnx_type_info HNX_TYPES[] = {
    { HNX_TYPE_IQ0_9,  "IQ0.9_L",     8, 7, sizeof(hnx_block_iq0_9),  0.9375f },
    { HNX_TYPE_IQ0_75, "IQ0.75_M",    4, 3, sizeof(hnx_block_iq0_75), 0.8125f },
    { HNX_TYPE_IQ0_5,  "IQ0.5_XXXL",  4, 2, sizeof(hnx_block_iq0_5),  0.5625f },
    { HNX_TYPE_IQ0_25, "IQ0.25_UXL", 16, 3, sizeof(hnx_block_iq0_25), 0.2500f },
    { HNX_TYPE_INT1,   "INT1",        1, 1, sizeof(hnx_block_int1),   1.0625f },
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
    hnx_decode(bytes + 2, hnx_fp16_to_fp32(raw), info->group, info->kept, dst);
    return 0;
}

size_t hnx_dequantize_rows(int type, const void *src, float *dst, size_t nblocks) {
    const hnx_type_info *info = hnx_type_lookup(type);
    if (info == NULL || src == NULL || dst == NULL) return 0;
    const uint8_t *in = (const uint8_t *)src;
    for (size_t b = 0; b < nblocks; b++) {
        uint16_t raw;
        memcpy(&raw, in, sizeof(raw));
        hnx_decode(in + 2, hnx_fp16_to_fp32(raw), info->group, info->kept,
                   dst + b * HNX_BLOCK_SIZE);
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
size_t hnx_quantize_rows(int type, const float *src, void *dst, size_t nblocks,
                         const float *importance) {
    const hnx_type_info *info = hnx_type_lookup(type);
    if (info == NULL || src == NULL || dst == NULL) return 0;
    uint8_t *out = (uint8_t *)dst;

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
