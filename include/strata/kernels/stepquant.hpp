#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels {
// Independent implementation of STEPQuant's GDN dual-axis fit. State addressing
// is Strata's (key, head, value), S=128; allocation is per whole value head.
struct StepQuantPlan {
    std::vector<int> bits;           // 2, 4, 6, 8, or 16 (FP16 pivot)
    std::vector<float> impact;       // [head, key], finite positive normalized w
};
// One packed state. Owns device metadata and tightly packed codes + FP16 scales.
// Construct outside CUDA graph capture, on the device used by all subsequent calls.
class StepQuantState {
public:
    explicit StepQuantState(const StepQuantPlan& plan);
    ~StepQuantState();
    StepQuantState(const StepQuantState&) = delete;
    StepQuantState& operator=(const StepQuantState&) = delete;
    size_t bytes() const; // payload only; excludes shared plan metadata
    void pack(const float* state, void* stream);
    void unpack(float* state, void* stream) const;
    // Full compressed recurrence: one shared FP32 scratch matrix, unquantized readout,
    // then packed persistent writeback. q is normalized but NOT scaled by 1/sqrt(128).
    // Fresh states start at zero. No allocation or host synchronization in these calls.
    void step(float* scratch, const float* q, const float* k, const float* v,
              const float* gate, const float* beta, float* output, int key_heads, void* stream);
    void writeback(float* state, void* stream); // pack then reconstruct in FP32
    // Borrow caller storage (bytes() bytes, >=2-byte aligned), or nullptr to
    // detach it. Caller keeps it alive through all queued work and captured graphs.
    void bind(void* payload);
    void* data() const; // device payload, for copies/checkpoints; not an upstream wire format
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Engine mode: one packed state per GDN layer, one shared FP32 scratch matrix.
// Frozen plan is loaded and all buffers allocated BEFORE session graph capture.
void stepquant_configure(const std::string& path, int layers, int qsa_interval, int heads, int state_size);
bool stepquant_enabled();
bool stepquant_accepts_geometry(int layers, int interval, int heads, int state_size);
void stepquant_writeback(int layer, float* state, void* stream); // pack only
void stepquant_read(int layer, float* scratch, void* stream);
size_t stepquant_recurrence_bytes(); // common 256-aligned slot, includes plan header
bool stepquant_bind_session(void* arena, size_t convolution_bytes, int count, int ordinal);
void stepquant_zero_session(void* arena, size_t convolution_bytes, void* stream);
bool stepquant_zero_layer(void* slot, void* stream);
void stepquant_unbind_session(void* arena);
bool stepquant_validate_session(const void* host_arena, size_t bytes, size_t convolution_bytes,
                                int count, int ordinal, std::string& error);
void stepquant_release();
} // namespace strata::kernels
