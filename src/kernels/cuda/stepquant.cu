// Equations 11-14 of STEPQuant (arXiv:2609.38169), independently implemented.
// Reference for numerical validation: Dreamer-Toby/STEPQuant at 61f24c9.
#include "strata/kernels/stepquant.hpp"
#include "strata/kernels/native_gdn.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
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
std::vector<std::unique_ptr<StepQuantState>> runtime;
int runtime_device = -1;
int runtime_heads = 0, runtime_interval = 0;
void* runtime_arena = nullptr;
size_t runtime_slot = 0;
std::vector<std::array<uint64_t, 2>> headers;
uint64_t plan_hash(const StepQuantPlan& plan, int layer) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint32_t x) { for (int j = 0; j < 4; ++j) { h ^= (x >> (8*j)) & 255; h *= 1099511628211ull; } };
    mix(1); mix(S); mix(uint32_t(layer));
    for (int b : plan.bits) mix(uint32_t(b));
    for (float w : plan.impact) { uint32_t x; std::memcpy(&x, &w, 4); mix(x); }
    return h;
}
} // namespace
struct StepQuantState::Impl {
    Head* heads = nullptr;
    float* impact = nullptr;
    unsigned char* payload = nullptr;
    size_t size = 0;
    int H = 0, device = -1;
    bool owns_payload = true;
    ~Impl() {
        int previous = -1;
        cudaGetDevice(&previous);
        if (device >= 0) cudaSetDevice(device);
        if (owns_payload) cudaFree(payload);
        cudaFree(impact); cudaFree(heads);
        if (previous >= 0) cudaSetDevice(previous);
    }
    void same_device() const {
        int current = -1; check(cudaGetDevice(&current));
        if (current != device) throw std::invalid_argument("STEPQuant: state belongs to another CUDA device");
    }
};
StepQuantState::StepQuantState(const StepQuantPlan& plan) : impl_(std::make_unique<Impl>()) {
    auto& m = *impl_;
    if (plan.bits.empty() || plan.bits.size() > 65535 || plan.impact.size() != plan.bits.size()*S)
        throw std::invalid_argument("STEPQuant: expected 1..65535 heads and head*128 impact factors");
    for (float w : plan.impact)
        if (!std::isfinite(w) || w <= 0) throw std::invalid_argument("STEPQuant: impact must be finite and positive");
    std::vector<Head> heads;
    for (int b : plan.bits) {
        if (b != 2 && b != 4 && b != 6 && b != 8 && b != 16)
            throw std::invalid_argument("STEPQuant: bits must be 2/4/6/8/16");
        heads.push_back({m.size, b});
        m.size += b == 16 ? S*S*2 : S*S*b/8 + (S*(b == 2 ? 4 : 1)+S)*2;
    }
    m.H = int(heads.size());
    check(cudaGetDevice(&m.device));
    check(cudaMalloc(reinterpret_cast<void**>(&m.heads), heads.size()*sizeof(Head)));
    check(cudaMalloc(reinterpret_cast<void**>(&m.impact), plan.impact.size()*sizeof(float)));
    check(cudaMalloc(reinterpret_cast<void**>(&m.payload), m.size));
    check(cudaMemcpy(m.heads, heads.data(), heads.size()*sizeof(Head), cudaMemcpyHostToDevice));
    check(cudaMemcpy(m.impact, plan.impact.data(), plan.impact.size()*sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemset(m.payload, 0, m.size));
}
StepQuantState::~StepQuantState() = default;
size_t StepQuantState::bytes() const { return impl_->size; }
void* StepQuantState::data() const { return impl_->payload; }
void StepQuantState::bind(void* payload) {
    impl_->same_device();
    if (reinterpret_cast<uintptr_t>(payload) % 2) throw std::invalid_argument("STEPQuant: unaligned borrowed payload");
    if (payload == impl_->payload) return;
    if (impl_->owns_payload) check(cudaFree(impl_->payload));
    impl_->payload = static_cast<unsigned char*>(payload);
    impl_->owns_payload = false;
}
void StepQuantState::pack(const float* state, void* stream) {
    impl_->same_device();
    if (!state || !impl_->payload) throw std::invalid_argument("STEPQuant: null state/payload");
    fit_pack<<<impl_->H, S, 0, static_cast<cudaStream_t>(stream)>>>(state, impl_->payload, impl_->heads, impl_->impact, impl_->H);
    check(cudaGetLastError());
}
void StepQuantState::unpack(float* state, void* stream) const {
    impl_->same_device();
    if (!state || !impl_->payload) throw std::invalid_argument("STEPQuant: null state/payload");
    reconstruct<<<impl_->H, S, 0, static_cast<cudaStream_t>(stream)>>>(state, impl_->payload, impl_->heads, impl_->H);
    check(cudaGetLastError());
}
void StepQuantState::step(float* scratch, const float* q, const float* k, const float* v,
                          const float* gate, const float* beta, float* output, int key_heads, void* stream) {
    // Validate geometry before unpack changes caller-owned scratch.
    if (key_heads < 1 || impl_->H % key_heads != 0 || !stream || !scratch || !q || !k || !v || !gate || !beta || !output)
        throw std::invalid_argument("STEPQuant: invalid recurrent step arguments");
    unpack(scratch, stream);
    native_gdn_step(scratch, q, k, v, gate, beta, output, {S, key_heads, impl_->H}, stream);
    pack(scratch, stream);
}
void StepQuantState::writeback(float* state, void* stream) { pack(state, stream); unpack(state, stream); }
void stepquant_configure(const std::string& path, int layers, int interval, int heads, int state_size) {
    if (!runtime.empty()) throw std::logic_error("STEPQuant: runtime already configured");
    if (state_size != S || heads < 1 || heads > 65535 || layers < 1 || interval < 1)
        throw std::invalid_argument("STEPQuant: unsupported model geometry");
    std::ifstream input(path);
    std::string magic;
    int version = 0, dim = 0, H = 0, n = 0;
    if (!(input >> magic >> version >> dim >> H >> n) || magic != "STRATA_STEPQUANT" || version != 1 ||
        dim != S || H != heads || n != layers-layers/interval)
        throw std::invalid_argument("STEPQuant: incompatible or truncated plan header");
    std::vector<std::unique_ptr<StepQuantState>> pending(layers);
    std::vector<std::array<uint64_t, 2>> pending_headers(layers);
    size_t slot = 0;
    for (int i = 0; i < n; ++i) {
        int layer = -1;
        if (!(input >> layer) || layer < 0 || layer >= layers || layer%interval == interval-1 || pending[layer])
            throw std::invalid_argument("STEPQuant: invalid/duplicate/QSA layer in plan");
        StepQuantPlan plan;
        plan.bits.resize(heads); plan.impact.resize(size_t(heads)*S);
        for (int& b : plan.bits) if (!(input >> b)) throw std::invalid_argument("STEPQuant: truncated bits");
        for (float& w : plan.impact) if (!(input >> w)) throw std::invalid_argument("STEPQuant: truncated impact");
        pending[layer] = std::make_unique<StepQuantState>(plan);
        pending_headers[layer] = {plan_hash(plan, layer), uint64_t(pending[layer]->bytes())};
        slot = std::max(slot, pending[layer]->bytes() + 16);
        pending[layer]->bind(nullptr); // session arena owns the persistent payload

    }
    std::string trailing;
    if (input >> trailing) throw std::invalid_argument("STEPQuant: trailing plan data");
    check(cudaGetDevice(&runtime_device));
    runtime = std::move(pending);
    headers = std::move(pending_headers);
    runtime_slot = (slot + 255) / 256 * 256;
    runtime_heads = heads; runtime_interval = interval;
}
bool stepquant_enabled() { return !runtime.empty(); }
bool stepquant_accepts_geometry(int layers, int interval, int heads, int state_size) {
    return runtime.empty() || (layers == int(runtime.size()) && interval == runtime_interval &&
                               heads == runtime_heads && state_size == S);
}
void stepquant_writeback(int layer, float* state, void* stream) {
    if (runtime.empty()) return;
    int device = -1; check(cudaGetDevice(&device));
    if (device != runtime_device || layer < 0 || size_t(layer) >= runtime.size() || !runtime[layer])
        throw std::invalid_argument("STEPQuant: layer/device has no calibrated plan");
    runtime[layer]->pack(state, stream);
}
void stepquant_read(int layer, float* scratch, void* stream) {
    if (runtime.empty()) return;
    if (layer < 0 || size_t(layer) >= runtime.size() || !runtime[layer])
        throw std::invalid_argument("STEPQuant: no GDN plan for read");
    runtime[layer]->unpack(scratch, stream);
}
size_t stepquant_recurrence_bytes() { return runtime_slot; }
bool stepquant_bind_session(void* arena, size_t convolution_bytes, int count, int ordinal) {
    if (runtime.empty()) return true;
    const int expected = int(std::count_if(runtime.begin(), runtime.end(), [](const auto& p) { return bool(p); }));
    if (!arena || count != expected || ordinal != 0 || (runtime_arena && runtime_arena != arena)) return false;
    size_t i = 0;
    for (auto& state : runtime) if (state) {
        state->bind(static_cast<unsigned char*>(arena) + i*(runtime_slot+convolution_bytes) + 16);
        ++i;
    }
    runtime_arena = arena;
    return true;
}
void stepquant_zero_session(void* arena, size_t convolution_bytes, void* stream) {
    if (runtime.empty()) return;
    if (arena != runtime_arena) throw std::invalid_argument("STEPQuant: zero of unbound session");
    size_t i = 0;
    for (size_t layer = 0; layer < runtime.size(); ++layer) if (runtime[layer]) {
        // The session already zeroed all payloads, padding and convolution history.
        // Header source addresses remain stable through all queued copies.
        check(cudaMemcpyAsync(static_cast<unsigned char*>(arena) + i*(runtime_slot+convolution_bytes),
                              headers[layer].data(), 16, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
        ++i;
    }
}
bool stepquant_zero_layer(void* slot, void* stream) {
    if (runtime.empty() || !slot) return false;
    for (size_t layer = 0; layer < runtime.size(); ++layer) if (runtime[layer] && runtime[layer]->data() &&
        static_cast<unsigned char*>(runtime[layer]->data()) == static_cast<unsigned char*>(slot)+16) {
        check(cudaMemsetAsync(slot, 0, runtime_slot, static_cast<cudaStream_t>(stream)));
        check(cudaMemcpyAsync(slot, headers[layer].data(), 16, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
        return true;
    }
    return false;
}
void stepquant_unbind_session(void* arena) {
    if (arena != runtime_arena || !runtime_arena) return;
    for (auto& state : runtime) if (state) state->bind(nullptr);
    runtime_arena = nullptr;
}
bool stepquant_validate_session(const void* host_arena, size_t bytes, size_t convolution_bytes,
                                int count, int ordinal, std::string& error) {
    if (runtime.empty()) return true;
    const size_t stride = runtime_slot + convolution_bytes;
    const int expected = int(std::count_if(runtime.begin(), runtime.end(), [](const auto& p) { return bool(p); }));
    if (!host_arena || ordinal != 0 || count != expected || bytes != size_t(count)*stride) {
        error = "STEPQuant: incompatible packed session geometry"; return false;
    }
    size_t i = 0;
    for (size_t layer = 0; layer < runtime.size(); ++layer) if (runtime[layer]) {
        std::array<uint64_t, 2> header;
        std::memcpy(header.data(), static_cast<const unsigned char*>(host_arena) + i*stride, 16);
        if (header != headers[layer]) { error = "STEPQuant: checkpoint uses a different calibration plan"; return false; }
        ++i;
    }
    return true;
}
void stepquant_release() {
    runtime.clear(); headers.clear(); runtime_slot = 0; runtime_arena = nullptr; runtime_device = -1; runtime_heads = 0; runtime_interval = 0;
}
} // namespace strata::kernels
