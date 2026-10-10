// STEPQuant on SYCL; shared host plan/session runtime, device fit with FP16 scales.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/stepquant.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include <map>
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
// Local runtime adapters used only by the shared STEPQuant implementation.
using cudaStream_t = sycl::queue*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaMemcpyHostToDevice=0, cudaMemcpyDeviceToHost=1;
int cudaGetDevice(int* d) { *d=dpct::get_current_device_id(); return 0; }
int cudaSetDevice(int d) { dpct::select_device(d); return 0; }
int cudaMalloc(void** p, size_t n) { *p=sycl::malloc_device(n, dpct::get_in_order_queue()); if (!*p) throw std::bad_alloc(); return 0; }
int cudaFree(void* p) { sycl::free(p,dpct::get_in_order_queue()); return 0; }
int cudaMemcpy(void* to, const void* from, size_t n, int) { dpct::get_in_order_queue().memcpy(to,from,n).wait_and_throw(); return 0; }
int cudaMemcpyAsync(void* to, const void* from, size_t n, int, cudaStream_t q) { strata::q_of(q)->memcpy(to,from,n); return 0; }
int cudaMemset(void* p,int v,size_t n) { dpct::get_in_order_queue().memset(p,v,n).wait_and_throw(); return 0; }
int cudaMemsetAsync(void* p,int v,size_t n,cudaStream_t q) { strata::q_of(q)->memset(p,v,n); return 0; }
int cudaStreamSynchronize(cudaStream_t q) { strata::q_of(q)->wait_and_throw(); return 0; }
void check(int) {}
constexpr int S=128;
struct Head { size_t offset; int bits; };
float stored_scale(float x) {
    return float(sycl::half(sycl::fmin(65504.f, sycl::fmax(0x1p-24f, x))));
}
int level(float x, int bits) {
    if (bits == 2) return (x >= 0.f ? 1 : -1) * (sycl::fabs(x) >= 2.f ? 3 : 1);
    const int m = (1 << (bits - 1)) - 1;
    return int(sycl::fmin(float(m), sycl::fmax(float(-m), sycl::rint(x))));
}
void fit_pack(const float* state, unsigned char* payload, const Head* heads,
                         const float* impact, int H, sycl::nd_item<1> item, float* rows, float* columns) {
    const int h = item.get_group(0), t = item.get_local_id(0);
    const Head cfg = heads[h];
    unsigned char* p = payload + cfg.offset;
    if (cfg.bits == 16) {
        auto* fp = reinterpret_cast<sycl::half*>(p);
        for (int c = 0; c < S; ++c) fp[t*S+c] = sycl::half(state[(size_t(t)*H+h)*S+c]);
        return;
    }
    const int groups = cfg.bits == 2 ? 4 : 1;
    const int width = S / groups;
    auto* r_half = reinterpret_cast<sycl::half*>(p);
    auto* c_half = r_half + S*groups;
    auto* codes = reinterpret_cast<unsigned char*>(c_half + S);
    const float w = impact[h*S+t];
    for (int g = 0; g < groups; ++g) {
        float sum = 0;
        for (int c = g*width; c < (g+1)*width; ++c) sum += sycl::fabs(state[(size_t(t)*H+h)*S+c]);
        const float r = stored_scale(sycl::sqrt(sycl::fmax(sum / width, 1.e-20f) / w));
        rows[t*groups+g] = r;
        r_half[t*groups+g] = sycl::half(r);
    }
    item.barrier(sycl::access::fence_space::local_space);
    const int qmax = cfg.bits == 2 ? 3 : (1 << (cfg.bits-1))-1;
    float c0 = 0;
    for (int r = 0; r < S; ++r) {
        const float x = state[(size_t(r)*H+h)*S+t];
        c0 = sycl::fmax(c0, sycl::fabs(x / rows[r*groups+t/width]) / qmax);
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
    const float fitted = stored_scale(denominator > 0 ? numerator / sycl::fmax(denominator, 1.e-30f) : c0);
    columns[t] = fitted;
    c_half[t] = sycl::half(fitted);
    item.barrier(sycl::access::fence_space::local_space);
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
void reconstruct(float* state, const unsigned char* payload, const Head* heads, int H, sycl::nd_item<1> item) {
    const int h = item.get_group(0), col = item.get_local_id(0);
    const Head cfg = heads[h];
    const unsigned char* p = payload + cfg.offset;
    if (cfg.bits == 16) {
        const auto* fp = reinterpret_cast<const sycl::half*>(p);
        for (int row = 0; row < S; ++row) state[(size_t(row)*H+h)*S+col] = float(fp[row*S+col]);
        return;
    }
    const int groups = cfg.bits == 2 ? 4 : 1;
    const auto* r = reinterpret_cast<const sycl::half*>(p);
    const auto* c = r + S*groups;
    const auto* codes = reinterpret_cast<const unsigned char*>(c + S);
    const float cs = float(c[col]);
    for (int row = 0; row < S; ++row) {
        const size_t bit = size_t(row*S+col)*cfg.bits;
        const size_t byte = bit/8;
        unsigned u = codes[byte];
        if ((bit%8) + cfg.bits > 8) u |= unsigned(codes[byte+1]) << 8;
        u = (u >> (bit%8)) & ((1u << cfg.bits)-1);
        const int z = cfg.bits == 2 ? 2*int(u)-3 : int(u)-((1 << (cfg.bits-1))-1);
        state[(size_t(row)*H+h)*S+col] = float(r[row*groups+col/(S/groups)]) * cs * z;
    }
}
void commit_prefix(unsigned char* slot, const unsigned char* temporary, size_t bytes,
                              const int32_t* keep, int prefix, sycl::nd_item<1> item) {
    if (*keep != prefix) return;
    for (size_t i = item.get_global_id(0); i < bytes; i += item.get_global_range(0))
        slot[i] = temporary[i];
}

void launch_fit(const float* state, unsigned char* payload, const Head* heads, const float* impact, int H, void* stream) {
    strata::q_of(stream)->submit([=](sycl::handler& cgh) {
        sycl::local_accessor<float,1> rows(sycl::range<1>(S*4),cgh), columns(sycl::range<1>(S),cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(H)*S,S),[=](sycl::nd_item<1> item) {
            fit_pack(state,payload,heads,impact,H,item,rows.get_multi_ptr<sycl::access::decorated::no>().get(),columns.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}
void launch_reconstruct(float* state, const unsigned char* payload, const Head* heads, int H, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>(size_t(H)*S,S),[=](sycl::nd_item<1> item) {
        reconstruct(state,payload,heads,H,item);
    });
}
void launch_commit(unsigned char* slot, const unsigned char* temp, size_t bytes, const int32_t* keep, int prefix, void* stream) {
    strata::q_of(stream)->parallel_for(sycl::nd_range<1>(32768,256),[=](sycl::nd_item<1> item) {
        commit_prefix(slot,temp,bytes,keep,prefix,item);
    });
}
#include "../../../../src/kernels/stepquant_runtime.inc"
