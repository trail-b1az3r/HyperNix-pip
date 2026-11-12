/*
 * strife_kernels.c -- Strife's native kernels for quantised GGUF weights.
 *
 * One file, no dependencies beyond libc (and OpenMP when the compiler has
 * it). `hnx strife kernels build` compiles it for the machine it runs on and
 * checks every kernel against the Python decoders before Strife will use it.
 *
 * Two things are done here, for each tensor type listed in block_info():
 *
 *   strife_decode   blocks -> float32, the same values the Python decoders
 *                   in hypernix.quant.llamaquants produce (computed in
 *                   float32, as llama.cpp does);
 *   strife_matmul   out = x @ W.T with W still quantised: each row is decoded
 *                   a block at a time into a buffer that stays in the L1
 *                   cache and is multiplied straight away, so a model never
 *                   has to exist in RAM as float32.
 *
 * The layouts are llama.cpp's (ggml-quants.c). Little-endian hosts only.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "strife_kernels: little-endian hosts only"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define STRIFE_API __declspec(dllexport)
#else
#define STRIFE_API __attribute__((visibility("default")))
#endif

#define STRIFE_ABI 1

/* GGML type ids. */
enum {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30
};

/* ---- float16 ----------------------------------------------------------- */

static float f16_table[65536];
static int f16_ready = 0;

static float f16_slow(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                         /* subnormal: normalise */
            int shift = 0;
            while (!(man & 0x400u)) { man <<= 1; shift++; }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - shift + 1) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static void f16_init(void) {
    if (f16_ready) return;
    for (uint32_t i = 0; i < 65536; i++) f16_table[i] = f16_slow((uint16_t)i);
    f16_ready = 1;                       /* idempotent: a race writes the same values */
}

static inline float rd_f16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return f16_table[h];
}

/* ---- one block of each type -> float32 --------------------------------- */

/*
 * Sixteen quantised values to float32. Written out with intrinsics where
 * the CPU has them: left to itself the compiler keeps these loops scalar.
 *   scale16:  dst = s * src            (signed bytes)
 *   affine16: dst = dl * src - ml      (unsigned bytes)
 * The products are exact in float32 (an 11-bit float16 mantissa times at
 * most 11 more bits), so every form gives the same bits.
 */
static inline void scale16(const int8_t *src, float s, float *dst) {
#if defined(__AVX512F__)
    const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)src)));
    _mm512_storeu_ps(dst, _mm512_mul_ps(v, _mm512_set1_ps(s)));
#elif defined(__AVX2__)
    const __m256 sv = _mm256_set1_ps(s);
    for (int h = 0; h < 16; h += 8) {
        const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(src + h))));
        _mm256_storeu_ps(dst + h, _mm256_mul_ps(v, sv));
    }
#else
    for (int l = 0; l < 16; l++) dst[l] = s * (float)src[l];
#endif
}

static inline void affine16(const uint8_t *src, float dl, float ml, float *dst) {
#if defined(__AVX512F__)
    const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)src)));
    _mm512_storeu_ps(dst, _mm512_sub_ps(_mm512_mul_ps(v, _mm512_set1_ps(dl)), _mm512_set1_ps(ml)));
#elif defined(__AVX2__)
    const __m256 dv = _mm256_set1_ps(dl), mv = _mm256_set1_ps(ml);
    for (int h = 0; h < 16; h += 8) {
        const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(src + h))));
        _mm256_storeu_ps(dst + h, _mm256_sub_ps(_mm256_mul_ps(v, dv), mv));
    }
#else
    for (int l = 0; l < 16; l++) dst[l] = dl * (float)src[l] - ml;
#endif
}

static void dec_f32(const uint8_t *restrict b, float *restrict y) { memcpy(y, b, 32 * 4); }

static void dec_f16(const uint8_t *restrict b, float *restrict y) {
    for (int i = 0; i < 32; i++) y[i] = rd_f16(b + 2 * i);
}

static void dec_bf16(const uint8_t *restrict b, float *restrict y) {
    for (int i = 0; i < 32; i++) {
        uint16_t h;
        memcpy(&h, b + 2 * i, 2);
        uint32_t bits = (uint32_t)h << 16;
        memcpy(&y[i], &bits, 4);
    }
}

/* Sixteen bytes -> their sixteen low nibbles, then their sixteen high nibbles. */
static inline void nibbles32(const uint8_t *qs, uint8_t *out) {
#if defined(__AVX2__)
    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)qs);
    const __m128i m4 = _mm_set1_epi8(0xF);
    _mm_storeu_si128((__m128i *)(void *)out, _mm_and_si128(v, m4));
    _mm_storeu_si128((__m128i *)(void *)(out + 16), _mm_and_si128(_mm_srli_epi16(v, 4), m4));
#else
    for (int j = 0; j < 16; j++) {
        out[j] = qs[j] & 0x0F;
        out[j + 16] = qs[j] >> 4;
    }
#endif
}

static void dec_q4_0(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b);
    uint8_t q[32];
    nibbles32(b + 2, q);
    /* (q - 8) * d, as d * q - 8 * d: both products are exact. */
    affine16(q, d, 8.0f * d, y);
    affine16(q + 16, d, 8.0f * d, y + 16);
}

static void dec_q4_1(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b), m = rd_f16(b + 2);
    uint8_t q[32];
    nibbles32(b + 4, q);
    affine16(q, d, -m, y);
    affine16(q + 16, d, -m, y + 16);
}

static void dec_q5_0(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b);
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    const uint8_t *qs = b + 6;
    for (int j = 0; j < 16; j++) {
        const int h0 = (int)((qh >> j) & 1u) << 4;
        const int h1 = (int)((qh >> (j + 16)) & 1u) << 4;
        y[j] = (float)(((int)(qs[j] & 0x0F) | h0) - 16) * d;
        y[j + 16] = (float)(((int)(qs[j] >> 4) | h1) - 16) * d;
    }
}

static void dec_q5_1(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b), m = rd_f16(b + 2);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    const uint8_t *qs = b + 8;
    for (int j = 0; j < 16; j++) {
        const int h0 = (int)((qh >> j) & 1u) << 4;
        const int h1 = (int)((qh >> (j + 16)) & 1u) << 4;
        y[j] = (float)((int)(qs[j] & 0x0F) | h0) * d + m;
        y[j + 16] = (float)((int)(qs[j] >> 4) | h1) * d + m;
    }
}

static void dec_q8_0(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b);
    const int8_t *qs = (const int8_t *)(b + 2);
    scale16(qs, d, y);
    scale16(qs + 16, d, y + 16);
}

/*
 * The K-quants unpack in two steps: every quantised value into a small
 * byte array (32 bytes at a time with AVX2), then sixteen at a time to
 * float with their scale.
 */
static void dec_q2_k(const uint8_t *restrict b, float *restrict y) {
    const uint8_t *scales = b, *qs = b + 16;
    const float d = rd_f16(b + 80), dmin = rd_f16(b + 82);
    uint8_t q[256];
    for (int n = 0; n < 2; n++) {
        const uint8_t *src = qs + 32 * n;
        uint8_t *o = q + 128 * n;
#if defined(__AVX2__)
        const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)src);
        const __m256i m3 = _mm256_set1_epi8(3);
        _mm256_storeu_si256((__m256i *)(void *)(o), _mm256_and_si256(v, m3));
        _mm256_storeu_si256((__m256i *)(void *)(o + 32), _mm256_and_si256(_mm256_srli_epi16(v, 2), m3));
        _mm256_storeu_si256((__m256i *)(void *)(o + 64), _mm256_and_si256(_mm256_srli_epi16(v, 4), m3));
        _mm256_storeu_si256((__m256i *)(void *)(o + 96), _mm256_and_si256(_mm256_srli_epi16(v, 6), m3));
#else
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 32; l++) o[32 * j + l] = (uint8_t)((src[l] >> (2 * j)) & 3);
#endif
    }
    for (int g = 0; g < 16; g++)
        affine16(q + 16 * g, d * (float)(scales[g] & 0xF), dmin * (float)(scales[g] >> 4), y + 16 * g);
}

static void dec_q3_k(const uint8_t *restrict b, float *restrict y) {
    const uint8_t *hm = b, *qs = b + 32, *sp = b + 96;
    const float d = rd_f16(b + 108);
    int8_t scales[16];
    for (int j = 0; j < 16; j++) {
        const int low = j < 8 ? (sp[j] & 0xF) : (sp[j - 8] >> 4);
        const int high = (sp[8 + j % 4] >> (2 * (j / 4))) & 3;
        scales[j] = (int8_t)((low | (high << 4)) - 32);
    }
    int8_t q[256];
    for (int n = 0; n < 2; n++) {
        const uint8_t *src = qs + 32 * n;
        int8_t *o = q + 128 * n;
#if defined(__AVX2__)
        const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)src);
        const __m256i h = _mm256_loadu_si256((const __m256i *)(const void *)hm);
        const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4), zero = _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {
            const __m256i low = _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(2 * j)), m3);
            const __m256i bit = _mm256_and_si256(h, _mm256_set1_epi8((char)(1u << (4 * n + j))));
            /* The high bit clear means "subtract 4". */
            const __m256i sub = _mm256_and_si256(_mm256_cmpeq_epi8(bit, zero), four);
            _mm256_storeu_si256((__m256i *)(void *)(o + 32 * j), _mm256_sub_epi8(low, sub));
        }
#else
        for (int j = 0; j < 4; j++) {
            const uint8_t m = (uint8_t)(1u << (4 * n + j));
            for (int l = 0; l < 32; l++)
                o[32 * j + l] = (int8_t)((int)((src[l] >> (2 * j)) & 3) - ((hm[l] & m) ? 0 : 4));
        }
#endif
    }
    for (int g = 0; g < 16; g++) scale16(q + 16 * g, d * (float)scales[g], y + 16 * g);
}

static inline void scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *mn = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void dec_q4_k(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b), dmin = rd_f16(b + 2);
    const uint8_t *scales = b + 4, *qs = b + 16;
    uint8_t q[256];
    for (int j = 0; j < 4; j++) {
        const uint8_t *src = qs + 32 * j;
        uint8_t *o = q + 64 * j;
#if defined(__AVX2__)
        const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)src);
        const __m256i m4 = _mm256_set1_epi8(0xF);
        _mm256_storeu_si256((__m256i *)(void *)o, _mm256_and_si256(v, m4));
        _mm256_storeu_si256((__m256i *)(void *)(o + 32), _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
#else
        for (int l = 0; l < 32; l++) {
            o[l] = src[l] & 0xF;
            o[l + 32] = src[l] >> 4;
        }
#endif
    }
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mn;
        scale_min_k4(g, scales, &sc, &mn);
        const float dl = d * (float)sc, ml = dmin * (float)mn;
        affine16(q + 32 * g, dl, ml, y + 32 * g);
        affine16(q + 32 * g + 16, dl, ml, y + 32 * g + 16);
    }
}

static void dec_q5_k(const uint8_t *restrict b, float *restrict y) {
    const float d = rd_f16(b), dmin = rd_f16(b + 2);
    const uint8_t *scales = b + 4, *qh = b + 16, *ql = b + 48;
    uint8_t q[256];
    for (int j = 0; j < 4; j++) {
        const uint8_t *src = ql + 32 * j;
        uint8_t *o = q + 64 * j;
#if defined(__AVX2__)
        const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)src);
        const __m256i h = _mm256_loadu_si256((const __m256i *)(const void *)qh);
        const __m256i m4 = _mm256_set1_epi8(0xF), sixteen = _mm256_set1_epi8(16);
        const __m256i u1 = _mm256_set1_epi8((char)(1u << (2 * j))), u2 = _mm256_set1_epi8((char)(2u << (2 * j)));
        /* A set high bit adds 16. */
        const __m256i h1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(h, u1), u1), sixteen);
        const __m256i h2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(h, u2), u2), sixteen);
        _mm256_storeu_si256((__m256i *)(void *)o, _mm256_or_si256(_mm256_and_si256(v, m4), h1));
        _mm256_storeu_si256((__m256i *)(void *)(o + 32),
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), h2));
#else
        const uint8_t u1 = (uint8_t)(1u << (2 * j)), u2 = (uint8_t)(2u << (2 * j));
        for (int l = 0; l < 32; l++) {
            o[l] = (uint8_t)((src[l] & 0xF) + ((qh[l] & u1) ? 16 : 0));
            o[l + 32] = (uint8_t)((src[l] >> 4) + ((qh[l] & u2) ? 16 : 0));
        }
#endif
    }
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mn;
        scale_min_k4(g, scales, &sc, &mn);
        const float dl = d * (float)sc, ml = dmin * (float)mn;
        affine16(q + 32 * g, dl, ml, y + 32 * g);
        affine16(q + 32 * g + 16, dl, ml, y + 32 * g + 16);
    }
}

static void dec_q6_k(const uint8_t *restrict b, float *restrict y) {
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    const float d = rd_f16(b + 208);
    int8_t q[256];
    for (int n = 0; n < 2; n++) {
        const uint8_t *lo = ql + 64 * n, *hi = qh + 32 * n;
        int8_t *o = q + 128 * n;
#if defined(__AVX2__)
        const __m256i l0 = _mm256_loadu_si256((const __m256i *)(const void *)lo);
        const __m256i l1 = _mm256_loadu_si256((const __m256i *)(const void *)(lo + 32));
        const __m256i h = _mm256_loadu_si256((const __m256i *)(const void *)hi);
        const __m256i m4 = _mm256_set1_epi8(0xF), m30 = _mm256_set1_epi8(0x30), off = _mm256_set1_epi8(32);
        /* Two bits of `hi` go above each nibble: bits 0-1, 2-3, 4-5, 6-7 in turn. */
        const __m256i q1 = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_and_si256(_mm256_slli_epi16(h, 4), m30));
        const __m256i q2 = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_and_si256(_mm256_slli_epi16(h, 2), m30));
        const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_and_si256(h, m30));
        const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                           _mm256_and_si256(_mm256_srli_epi16(h, 2), m30));
        _mm256_storeu_si256((__m256i *)(void *)o, _mm256_sub_epi8(q1, off));
        _mm256_storeu_si256((__m256i *)(void *)(o + 32), _mm256_sub_epi8(q2, off));
        _mm256_storeu_si256((__m256i *)(void *)(o + 64), _mm256_sub_epi8(q3, off));
        _mm256_storeu_si256((__m256i *)(void *)(o + 96), _mm256_sub_epi8(q4, off));
#else
        for (int l = 0; l < 32; l++) {
            o[l] = (int8_t)((int)((lo[l] & 0xF) | ((hi[l] & 3) << 4)) - 32);
            o[l + 32] = (int8_t)((int)((lo[l + 32] & 0xF) | (((hi[l] >> 2) & 3) << 4)) - 32);
            o[l + 64] = (int8_t)((int)((lo[l] >> 4) | (((hi[l] >> 4) & 3) << 4)) - 32);
            o[l + 96] = (int8_t)((int)((lo[l + 32] >> 4) | (((hi[l] >> 6) & 3) << 4)) - 32);
        }
#endif
    }
    for (int g = 0; g < 16; g++) scale16(q + 16 * g, d * (float)sc[g], y + 16 * g);
}

typedef void (*block_fn)(const uint8_t *, float *);

/* Elements and bytes per block, and the decoder. Returns 0 for an unknown type. */
static int block_info(int type, int *elems, int *bytes, block_fn *fn) {
    switch (type) {
    case T_F32:  *elems = 32;  *bytes = 128; *fn = dec_f32;  return 1;
    case T_F16:  *elems = 32;  *bytes = 64;  *fn = dec_f16;  return 1;
    case T_BF16: *elems = 32;  *bytes = 64;  *fn = dec_bf16; return 1;
    case T_Q4_0: *elems = 32;  *bytes = 18;  *fn = dec_q4_0; return 1;
    case T_Q4_1: *elems = 32;  *bytes = 20;  *fn = dec_q4_1; return 1;
    case T_Q5_0: *elems = 32;  *bytes = 22;  *fn = dec_q5_0; return 1;
    case T_Q5_1: *elems = 32;  *bytes = 24;  *fn = dec_q5_1; return 1;
    case T_Q8_0: *elems = 32;  *bytes = 34;  *fn = dec_q8_0; return 1;
    case T_Q2_K: *elems = 256; *bytes = 84;  *fn = dec_q2_k; return 1;
    case T_Q3_K: *elems = 256; *bytes = 110; *fn = dec_q3_k; return 1;
    case T_Q4_K: *elems = 256; *bytes = 144; *fn = dec_q4_k; return 1;
    case T_Q5_K: *elems = 256; *bytes = 176; *fn = dec_q5_k; return 1;
    case T_Q6_K: *elems = 256; *bytes = 210; *fn = dec_q6_k; return 1;
    default: return 0;
    }
}

/* F32, F16 and BF16 are not block types: a row may be any length. */
static int is_plain(int type) { return type == T_F32 || type == T_F16 || type == T_BF16; }

static void dec_plain(int type, const uint8_t *src, int64_t n, float *dst) {
    if (type == T_F32) {
        memcpy(dst, src, (size_t)n * 4);
    } else if (type == T_F16) {
        int64_t i = 0;
#if defined(__F16C__) && defined(__AVX__)
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(src + 2 * i))));
#endif
        for (; i < n; i++) dst[i] = rd_f16(src + 2 * i);
    } else {
        for (int64_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, src + 2 * i, 2);
            uint32_t bits = (uint32_t)h << 16;
            memcpy(&dst[i], &bits, 4);
        }
    }
}

/* ---- dot products -------------------------------------------------------- */

/* A tile of decoded weights: 16 KiB of float32, inside any L1 data cache.
   TILE_ROWS is a multiple of 8, TILE_COLS a multiple of 256. */
#ifndef TILE_ROWS
#define TILE_ROWS 16
#endif
#ifndef TILE_COLS
#define TILE_COLS 256
#endif

/*
 * acc[r] += dot(w + r * stride, x, n) for r < 8.
 *
 * Eight rows against one input: x is loaded once per step and the eight
 * accumulators are independent, so the multiply-adds never wait on each
 * other.
 */
static inline void dot8(const float *w, int64_t stride, const float *x, int64_t n, float *acc) {
    int64_t i = 0;
#if defined(__AVX512F__)
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (; i + 16 <= n; i += 16) {
        const __m512 xv = _mm512_loadu_ps(x + i);
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 0 * stride + i), xv, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 1 * stride + i), xv, a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 2 * stride + i), xv, a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 3 * stride + i), xv, a3);
        a4 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 4 * stride + i), xv, a4);
        a5 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 5 * stride + i), xv, a5);
        a6 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 6 * stride + i), xv, a6);
        a7 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 7 * stride + i), xv, a7);
    }
    float s[8] = {_mm512_reduce_add_ps(a0), _mm512_reduce_add_ps(a1), _mm512_reduce_add_ps(a2),
                  _mm512_reduce_add_ps(a3), _mm512_reduce_add_ps(a4), _mm512_reduce_add_ps(a5),
                  _mm512_reduce_add_ps(a6), _mm512_reduce_add_ps(a7)};
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 a[8];
    for (int r = 0; r < 8; r++) a[r] = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        a[0] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0 * stride + i), xv, a[0]);
        a[1] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 1 * stride + i), xv, a[1]);
        a[2] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 2 * stride + i), xv, a[2]);
        a[3] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 3 * stride + i), xv, a[3]);
        a[4] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 4 * stride + i), xv, a[4]);
        a[5] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 5 * stride + i), xv, a[5]);
        a[6] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 6 * stride + i), xv, a[6]);
        a[7] = _mm256_fmadd_ps(_mm256_loadu_ps(w + 7 * stride + i), xv, a[7]);
    }
    float s[8];
    for (int r = 0; r < 8; r++) {
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a[r]), _mm256_extractf128_ps(a[r], 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        s[r] = _mm_cvtss_f32(lo);
    }
#else
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float lanes[8][8];
    memset(lanes, 0, sizeof lanes);
    for (; i + 8 <= n; i += 8)
        for (int r = 0; r < 8; r++)
            for (int k = 0; k < 8; k++) lanes[r][k] += w[r * stride + i + k] * x[i + k];
    for (int r = 0; r < 8; r++)
        s[r] = ((lanes[r][0] + lanes[r][1]) + (lanes[r][2] + lanes[r][3]))
             + ((lanes[r][4] + lanes[r][5]) + (lanes[r][6] + lanes[r][7]));
#endif
    for (; i < n; i++) {
        const float xv = x[i];
        for (int r = 0; r < 8; r++) s[r] += w[r * stride + i] * xv;
    }
    for (int r = 0; r < 8; r++) acc[r] += s[r];
}

/* Read `len` elements of `type` starting at element `first` of a row. */
static inline void decode_run(int type, block_fn fn, int elems, int bytes, const uint8_t *row,
                              int64_t first, int len, float *dst) {
    if (type == T_F32 || type == T_F16 || type == T_BF16) {
        dec_plain(type, row + first * (type == T_F32 ? 4 : 2), len, dst);
        return;
    }
    const uint8_t *src = row + first / elems * bytes;
    for (int done = 0; done < len; done += elems) fn(src + (done / elems) * bytes, dst + done);
}

/* ---- the exported functions -------------------------------------------- */

STRIFE_API int strife_abi(void) { return STRIFE_ABI; }

STRIFE_API int strife_openmp(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

/* A short name of the widest instruction set the dot product was built with. */
STRIFE_API const char *strife_simd(void) {
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__) && defined(__FMA__)
    return "avx2";
#elif defined(__ARM_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

STRIFE_API int strife_supported(int type) {
    int e, b;
    block_fn fn;
    return block_info(type, &e, &b, &fn);
}

/*
 * Decode n_elements of `type` from src into dst (float32).
 * Returns 0, or -1 for an unknown type, -2 when n_elements is not a whole
 * number of blocks.
 */
STRIFE_API int strife_decode(const uint8_t *src, int type, int64_t n_elements, float *dst,
                             int threads) {
    int elems, bytes;
    block_fn fn;
    if (!block_info(type, &elems, &bytes, &fn)) return -1;
    if (n_elements < 0) return -2;
    f16_init();
    if (is_plain(type)) {
        const int64_t chunk = 1 << 16;
        const int64_t chunks = (n_elements + chunk - 1) / chunk;
        const int width = type == T_F32 ? 4 : 2;
        (void)threads;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : omp_get_max_threads()) if (chunks > 4)
#endif
        for (int64_t c = 0; c < chunks; c++) {
            const int64_t start = c * chunk;
            const int64_t count = n_elements - start < chunk ? n_elements - start : chunk;
            dec_plain(type, src + start * width, count, dst + start);
        }
        return 0;
    }
    if (n_elements % elems) return -2;
    const int64_t blocks = n_elements / elems;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : omp_get_max_threads()) if (blocks > 256)
#endif
    for (int64_t i = 0; i < blocks; i++) fn(src + i * bytes, dst + i * elems);
    return 0;
}

/*
 * out[i * rows + r] = dot(W[r], x[i]) for i < n, r < rows.
 *
 * W is `rows` rows of `cols` elements of `type`, packed row after row.
 * x is n rows of cols float32, contiguous. out is n rows of `rows` float32.
 * Returns 0, -1 unknown type, -2 a shape the type cannot have, -3 out of
 * memory.
 *
 * The weight is walked in tiles of TILE_ROWS rows by TILE_COLS columns: a
 * tile is decoded into a buffer that stays in the L1 cache, then every
 * input's matching slice is multiplied against all of its rows. The
 * weight is read once, and nothing the size of the tensor is ever
 * allocated. (Decoding eight whole rows and sweeping the inputs across
 * them was tried and measured slower: the rows fall out of L1.)
 *
 * With several inputs, x is first repacked column-tile by column-tile, so
 * the slices a tile needs sit next to each other in memory and are read
 * in one forward sweep instead of one cache miss per input.
 */
#define MANY_INPUTS 8

STRIFE_API int strife_matmul(const uint8_t *w, int type, int64_t rows, int64_t cols,
                             const float *x, int64_t n, float *out, int threads) {
    int elems, bytes;
    block_fn fn;
    if (!block_info(type, &elems, &bytes, &fn)) return -1;
    if (rows < 0 || cols <= 0 || n < 0) return -2;
    f16_init();
    const int plain = is_plain(type);
    if (!plain && cols % elems) return -2;
    const int64_t row_bytes = plain ? cols * (type == T_F32 ? 4 : 2) : cols / elems * bytes;
    const int64_t tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;
#ifdef _OPENMP
    const int nthreads = threads > 0 ? threads : omp_get_max_threads();
    const int big = rows * cols * (n > 0 ? n : 1) > 200000;
#else
    (void)threads;
#endif
    float *packed = NULL;
    if (n >= MANY_INPUTS) {
        packed = (float *)malloc((size_t)n * (size_t)cols * sizeof(float));
        if (!packed) return -3;
        const int64_t col_tiles = (cols + TILE_COLS - 1) / TILE_COLS;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) if (big)
#endif
        for (int64_t c = 0; c < col_tiles; c++) {
            const int64_t c0 = c * TILE_COLS;
            const int len = (int)(cols - c0 < TILE_COLS ? cols - c0 : TILE_COLS);
            float *dst = packed + c0 * n;
            for (int64_t i = 0; i < n; i++)
                memcpy(dst + i * len, x + i * cols + c0, (size_t)len * sizeof(float));
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) if (big)
#endif
    for (int64_t t = 0; t < tiles; t++) {
#if defined(__GNUC__) || defined(__clang__)
        float buf[TILE_ROWS * TILE_COLS] __attribute__((aligned(64)));
#else
        float buf[TILE_ROWS * TILE_COLS];
#endif
        const int64_t r0 = t * TILE_ROWS;
        const int tr = (int)(rows - r0 < TILE_ROWS ? rows - r0 : TILE_ROWS);
        for (int64_t i = 0; i < n; i++) memset(out + i * rows + r0, 0, (size_t)tr * sizeof(float));
        /* Rows past the end of the weight read as zero and are never stored. */
        if (tr < TILE_ROWS) memset(buf + tr * TILE_COLS, 0, (size_t)(TILE_ROWS - tr) * TILE_COLS * sizeof(float));
        for (int64_t c0 = 0; c0 < cols; c0 += TILE_COLS) {
            const int len = (int)(cols - c0 < TILE_COLS ? cols - c0 : TILE_COLS);
            for (int r = 0; r < tr; r++)
                decode_run(type, fn, elems, bytes, w + (r0 + r) * row_bytes, c0, len,
                           buf + r * TILE_COLS);
            for (int64_t i = 0; i < n; i++) {
                const float *xs = packed ? packed + c0 * n + i * len : x + i * cols + c0;
                float acc[TILE_ROWS] = {0};
                for (int r = 0; r < TILE_ROWS; r += 8)
                    if (r < tr) dot8(buf + r * TILE_COLS, TILE_COLS, xs, len, acc + r);
                float *o = out + i * rows + r0;
                for (int r = 0; r < tr; r++) o[r] += acc[r];
            }
        }
    }
    free(packed);
    return 0;
}
