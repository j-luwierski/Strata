// Experimental DFlash2 primitives. Layout follows llama.cpp a9d27ac / z-lab 07ebd93.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
// Out-of-place grouped causal convolution. Each block of block_rows rows has its own left padding.
// dynamic: [rows][2][kernel][hidden/group], base: [2][kernel][hidden].
void dflash2_convolve(const float* x, const float* dynamic, const float* base, float* out,
                     int rows, int hidden, int kernel, int group, int side, int block_rows, void* stream);
// Top-k unary candidates for each row, with deterministic smaller-token tie breaking.
// Optional caller-owned workspace enables parallel vocabulary tiles followed by
// an exact candidate merge. No allocation or synchronization inside the call.
std::size_t dflash2_topk_scratch_bytes(int rows, int vocab, int top_k);
void dflash2_topk(const float* logits, int32_t* ids, float* unary, int rows, int vocab, int top_k, void* stream,
                  void* scratch = nullptr, std::size_t scratch_bytes = 0);
// Greedy predecessor-conditioned walk over rows 1..rows-1 (row 0 is the anchor).
// Codebooks [vocab][rank], hidden projection [rows][rank]. Writes rows-1 proposals and optional
// conditional top-k softmax confidence (a gate, not the target sampling distribution).
void dflash2_select(const int32_t* candidates, const float* unary, const float* hidden,
                    const float* predecessor, const float* successor, const int32_t* anchor,
                    int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream);
// BF16 codebooks avoid duplicate full-vocabulary FP32 copies; arithmetic is identical.
void dflash2_select_bf16(const int32_t* candidates, const float* unary, const float* hidden,
                         const uint16_t* predecessor, const uint16_t* successor, const int32_t* anchor,
                         int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream,
                         bool reference = false);
// Completed target residual [rows][hc][hidden], before the next layer's learned HC read.
void dflash2_hc_mean(const float* residual, float* out, uint16_t* out_bf16,
                     int rows, int hidden, int hc, void* stream);
// Non-causal local attention for the official 32Q/8KV x 128 drafter, FP16 paged pools.
void dflash2_attention(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* table,
                       const int32_t* steps, float* out, float* scratch, int rows, int heads,
                       int kv_heads, int page_size, int active_cells, int window, void* stream);
}  // namespace strata::kernels
