/*
 * ggml-hnx-shim.h — ggml's calling convention, over the plain-C decoder.
 *
 * Separate from ggml-hnx.c so the arithmetic can be built and tested
 * with no ggml headers anywhere in sight. That separation is what let
 * the bit order be verified against the Python encoder before any of
 * this touched upstream, and it is what keeps a llama.cpp rebase from
 * ever needing the decoder rewritten: only this file speaks ggml, and
 * all it does is adapt signatures.
 *
 * ggml wants one function per type with a fixed shape:
 *
 *     void to_float(const void *x, float *y, int64_t k)
 *     void vec_dot (int n, float *s, size_t bs, const void *x,
 *                   size_t bx, const void *y, size_t by, int nrc)
 *
 * where k and n are counts of *weights*, not blocks. Converting those to
 * block counts is the only real work here.
 */
#ifndef GGML_HNX_SHIM_H
#define GGML_HNX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#include "ggml-hnx.h"

#ifdef __cplusplus
extern "C" {
#endif

void hnx_ggml_to_float_iq0_9 (const void *x, float *y, int64_t k);
void hnx_ggml_to_float_iq0_75(const void *x, float *y, int64_t k);
void hnx_ggml_to_float_iq0_5 (const void *x, float *y, int64_t k);
void hnx_ggml_to_float_iq0_25(const void *x, float *y, int64_t k);
void hnx_ggml_to_float_int1  (const void *x, float *y, int64_t k);

void hnx_ggml_vec_dot_iq0_9 (int n, float *s, size_t bs, const void *x,
                             size_t bx, const void *y, size_t by, int nrc);
void hnx_ggml_vec_dot_iq0_75(int n, float *s, size_t bs, const void *x,
                             size_t bx, const void *y, size_t by, int nrc);
void hnx_ggml_vec_dot_iq0_5 (int n, float *s, size_t bs, const void *x,
                             size_t bx, const void *y, size_t by, int nrc);
void hnx_ggml_vec_dot_iq0_25(int n, float *s, size_t bs, const void *x,
                             size_t bx, const void *y, size_t by, int nrc);
void hnx_ggml_vec_dot_int1  (int n, float *s, size_t bs, const void *x,
                             size_t bx, const void *y, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* GGML_HNX_SHIM_H */
