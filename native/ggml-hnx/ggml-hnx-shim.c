/* ggml-hnx-shim.c — see ggml-hnx-shim.h. */
#include "ggml-hnx-shim.h"

/* One pair of wrappers per type, from one macro. Writing ten functions
 * out by hand is ten chances to paste the wrong type id into the wrong
 * body -- and a to_float that decoded IQ0.5 blocks as IQ0.75 would
 * still run, still fill the buffer, and produce a model that talks
 * nonsense. The macro cannot make that mistake.
 *
 * k and n are weight counts. A tensor's element count always divides
 * into HNX_BLOCK_SIZE because hyprslug refuses a tensor that does not
 * (padding one would change the model's shape), so the division is
 * exact; the +BLOCK-1 guards against a caller that asks for a partial
 * block anyway, which would otherwise silently decode nothing.
 */
#define HNX_SHIM(suffix, type_id)                                             \
    void hnx_ggml_to_float_##suffix(const void *x, float *y, int64_t k) {     \
        if (k <= 0) return;                                                   \
        const size_t nblocks =                                                \
            ((size_t)k + HNX_BLOCK_SIZE - 1) / HNX_BLOCK_SIZE;                \
        hnx_dequantize_rows((type_id), x, y, nblocks);                        \
    }                                                                         \
                                                                              \
    void hnx_ggml_vec_dot_##suffix(int n, float *s, size_t bs, const void *x, \
                                   size_t bx, const void *y, size_t by,       \
                                   int nrc) {                                 \
        /* bs/bx/by/nrc are the multi-row batching interface. These types  */ \
        /* declare nrows = 1, so ggml only ever calls with nrc == 1 and    */ \
        /* the strides unused; naming them keeps the signature honest      */ \
        /* without pretending to implement batching that was never        */  \
        /* written. */                                                        \
        (void) bs; (void) bx; (void) by; (void) nrc;                          \
        if (s == NULL) return;                                                \
        if (n <= 0) { *s = 0.0f; return; }                                    \
        const size_t nblocks = (size_t)n / HNX_BLOCK_SIZE;                    \
        *s = hnx_vec_dot((type_id), x, (const float *) y, nblocks);           \
    }

HNX_SHIM(iq0_9,  HNX_TYPE_IQ0_9)
HNX_SHIM(iq0_75, HNX_TYPE_IQ0_75)
HNX_SHIM(iq0_5,  HNX_TYPE_IQ0_5)
HNX_SHIM(iq0_25, HNX_TYPE_IQ0_25)
HNX_SHIM(int1,   HNX_TYPE_INT1)
HNX_SHIM(hnx1375, HNX_TYPE_1375)
HNX_SHIM(int4,   HNX_TYPE_INT4)
HNX_SHIM(fp2,    HNX_TYPE_FP2)

#undef HNX_SHIM
