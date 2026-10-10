// Equations 11-14 of STEPQuant (arXiv:2609.38169), independently implemented.
// Reference for numerical validation: Dreamer-Toby/STEPQuant at 61f24c9.
#include "strata/kernels/stepquant.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include <map>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;
struct Head { size_t offset; int bits; };
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("STEPQuant: ") + cudaGetErrorString(e));
}
__device__ float stored_scale(float x) {
    return __half2float(__float2half_rn(fminf(65504.f, fmaxf(0x1p-24f, x))));
}
__device__ int level(float x, int bits) {
    if (bits == 2) return (x >= 0.f ? 1 : -1) * (fabsf(x) >= 2.f ? 3 : 1);
    const int m = (1 << (bits - 1)) - 1;
    return int(fminf(float(m), fmaxf(float(-m), rintf(x))));
}
__global__ void fit_pack(const float* state, unsigned char* payload, const Head* heads,
                         const float* impact, int H) {
    const int h = blockIdx.x, t = threadIdx.x;
    const Head cfg = heads[h];
    unsigned char* p = payload + cfg.offset;
    if (cfg.bits == 16) {
        auto* fp = reinterpret_cast<__half*>(p);
        for (int c = 0; c < S; ++c) fp[t*S+c] = __float2half_rn(state[(size_t(t)*H+h)*S+c]);
        return;
    }
    const int groups = cfg.bits == 2 ? 4 : 1;
    const int width = S / groups;
    __shared__ float rows[S*4];
    __shared__ float columns[S];
    auto* r_half = reinterpret_cast<__half*>(p);
    auto* c_half = r_half + S*groups;
    auto* codes = reinterpret_cast<unsigned char*>(c_half + S);
    const float w = impact[h*S+t];
    for (int g = 0; g < groups; ++g) {
        float sum = 0;
        for (int c = g*width; c < (g+1)*width; ++c) sum += fabsf(state[(size_t(t)*H+h)*S+c]);
        const float r = stored_scale(sqrtf(fmaxf(sum / width, 1.e-20f) / w));
        rows[t*groups+g] = r;
        r_half[t*groups+g] = __float2half_rn(r);
    }
    __syncthreads();
    const int qmax = cfg.bits == 2 ? 3 : (1 << (cfg.bits-1))-1;
    float c0 = 0;
    for (int r = 0; r < S; ++r) {
        const float x = state[(size_t(r)*H+h)*S+t];
        c0 = fmaxf(c0, fabsf(x / rows[r*groups+t/width]) / qmax);
    }
    c0 = stored_scale(c0);
    float numerator = 0, denominator = 0;
    for (int r = 0; r < S; ++r) {
        const float x = state[(size_t(r)*H+h)*S+t];
        const float scale = rows[r*groups+t/width];
        const float z = scale * level(x / scale / c0, cfg.bits);
        const float wr = impact[h*S+r];
        numerator += wr*wr*z*x;
        denominator += (wr*wr)*(z*z);
    }
    const float fitted = stored_scale(denominator > 0 ? numerator / fmaxf(denominator, 1.e-30f) : c0);
    columns[t] = fitted;
    c_half[t] = __float2half_rn(fitted);
    __syncthreads();
    // Each thread owns a whole row, including all bytes straddled by INT6.
    // 128 values are divisible by every packing group: no padding or atomics.
    const int values_per_word = cfg.bits == 8 ? 1 : 4;
    const int bytes_per_word = values_per_word * cfg.bits / 8;
    for (int c = 0; c < S; c += values_per_word) {
        unsigned word = 0;
        for (int j = 0; j < values_per_word; ++j) {
            const float x = state[(size_t(t)*H+h)*S+c+j];
            const int z = level(x / rows[t*groups+(c+j)/width] / columns[c+j], cfg.bits);
            const unsigned u = cfg.bits == 2 ? unsigned((z+3)/2) : unsigned(z+qmax);
            word |= u << (j*cfg.bits);
        }
        const size_t start = size_t(t)*S*cfg.bits/8 + c*cfg.bits/8;
        for (int j = 0; j < bytes_per_word; ++j) codes[start+j] = (word >> (8*j)) & 255;
    }
}
__global__ void reconstruct(float* state, const unsigned char* payload, const Head* heads, int H) {
    const int h = blockIdx.x, col = threadIdx.x;
    const Head cfg = heads[h];
    const unsigned char* p = payload + cfg.offset;
    if (cfg.bits == 16) {
        const auto* fp = reinterpret_cast<const __half*>(p);
        for (int row = 0; row < S; ++row) state[(size_t(row)*H+h)*S+col] = __half2float(fp[row*S+col]);
        return;
    }
    const int groups = cfg.bits == 2 ? 4 : 1;
    const auto* r = reinterpret_cast<const __half*>(p);
    const auto* c = r + S*groups;
    const auto* codes = reinterpret_cast<const unsigned char*>(c + S);
    const float cs = __half2float(c[col]);
    for (int row = 0; row < S; ++row) {
        const size_t bit = size_t(row*S+col)*cfg.bits;
        const size_t byte = bit/8;
        unsigned u = codes[byte];
        if ((bit%8) + cfg.bits > 8) u |= unsigned(codes[byte+1]) << 8;
        u = (u >> (bit%8)) & ((1u << cfg.bits)-1);
        const int z = cfg.bits == 2 ? 2*int(u)-3 : int(u)-((1 << (cfg.bits-1))-1);
        state[(size_t(row)*H+h)*S+col] = __half2float(r[row*groups+col/(S/groups)]) * cs * z;
    }
}
__global__ void commit_prefix(unsigned char* slot, const unsigned char* temporary, size_t bytes,
                              const int32_t* keep, int prefix) {
    if (*keep != prefix) return;
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < bytes; i += gridDim.x*blockDim.x)
        slot[i] = temporary[i];
}
void launch_fit(const float* state, unsigned char* payload, const Head* heads, const float* impact, int H, void* stream) {
    fit_pack<<<H, S, 0, static_cast<cudaStream_t>(stream)>>>(state, payload, heads, impact, H);
    check(cudaGetLastError());
}
void launch_reconstruct(float* state, const unsigned char* payload, const Head* heads, int H, void* stream) {
    reconstruct<<<H, S, 0, static_cast<cudaStream_t>(stream)>>>(state, payload, heads, H);
    check(cudaGetLastError());
}
void launch_commit(unsigned char* slot, const unsigned char* temp, size_t bytes, const int32_t* keep, int prefix, void* stream) {
    commit_prefix<<<128, 256, 0, static_cast<cudaStream_t>(stream)>>>(slot, temp, bytes, keep, prefix);
    check(cudaGetLastError());
}
#include "../stepquant_runtime.inc"
