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
// Construct outside graph capture, on the device used by all subsequent calls.
class StepQuantState {
public:
    explicit StepQuantState(const StepQuantPlan& plan);
    ~StepQuantState();
    StepQuantState(const StepQuantState&) = delete;
    StepQuantState& operator=(const StepQuantState&) = delete;
    size_t bytes() const; // payload only; excludes shared plan metadata
    void pack(const float* state, void* stream);
    void unpack(float* state, void* stream) const;
    void pack_to(const float* state, void* payload, void* stream) const;
    void unpack_from(const void* payload, float* state, void* stream) const;
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
// Engine mode: per-session packed states, immutable plans replicated per device.
// Frozen plan is loaded and all buffers allocated BEFORE session graph capture.
void stepquant_configure(const std::string& path, int layers, int qsa_interval, int heads, int state_size);
bool stepquant_enabled();
bool stepquant_accepts_geometry(int layers, int interval, int heads, int state_size);
void stepquant_writeback(int layer, void* slot, const float* state, void* stream);
void stepquant_read(int layer, const void* slot, float* scratch, void* stream);
// Proposals leave slot untouched. A device keep counter commits exactly that prefix,
// including zero accepted tokens. Scratch and temporary belong to the caller's session.
void stepquant_verify(int layer, void* slot, float* scratch, void* temporary,
                      const float* h, int channels, const float* gate, const float* beta,
                      const float* z, const float* gamma, float eps, float* y,
                      int key_heads, int heads, int tokens, const int32_t* keep, void* stream);
size_t stepquant_recurrence_bytes(); // largest slot, for shared temporary storage only
size_t stepquant_recurrence_bytes(int layer); // this layer's 256-aligned slot + header
// Sum a contiguous GDN ordinal range. Also gives the offset to a layer when
// count is its ordinal minus the session's first ordinal.
size_t stepquant_session_bytes(size_t convolution_bytes, int count, int ordinal=0);
bool stepquant_bind_session(void* arena, size_t convolution_bytes, int count, int ordinal);
void stepquant_zero_session(void* arena, size_t convolution_bytes, void* stream);
bool stepquant_zero_layer(void* slot, void* stream);
void stepquant_unbind_session(void* arena);
bool stepquant_validate_session(const void* host_arena, size_t bytes, size_t convolution_bytes,
                                int count, int ordinal, std::string& error);
// Unquantized prompt trace for Strata's own offline calibration. Explicitly
// synchronous and bounded; never enable during serving or graph capture.
bool stepquant_trace_enabled();
void stepquant_trace_configure(const std::string& directory, int sample_every=8, int max_tokens=512);
void stepquant_observe(int layer, const float* h, const float* gate, const float* beta,
                       const float* state, int key_heads, int heads, void* stream);
void stepquant_release();
} // namespace strata::kernels
