// Shared host runtime for CUDA, HIP and SYCL. Included after backend launch helpers.
std::vector<StepQuantPlan> plans;
std::map<int, std::vector<std::unique_ptr<StepQuantState>>> devices;
int runtime_heads = 0, runtime_interval = 0;
struct Session { size_t conv; int count, ordinal, device; };
std::map<void*, Session> sessions;
size_t runtime_slot = 0;
std::vector<size_t> layer_slots, slot_prefix;
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
        if (current != device) throw std::invalid_argument("STEPQuant: state belongs to another GPU device");
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
void StepQuantState::pack(const float* state, void* stream) { pack_to(state, impl_->payload, stream); }
void StepQuantState::unpack(float* state, void* stream) const { unpack_from(impl_->payload, state, stream); }
void StepQuantState::pack_to(const float* state, void* payload, void* stream) const {
    impl_->same_device();
    if (!state || !payload) throw std::invalid_argument("STEPQuant: null state/payload");
    launch_fit(state, static_cast<unsigned char*>(payload), impl_->heads, impl_->impact, impl_->H, stream);
}
void StepQuantState::unpack_from(const void* payload, float* state, void* stream) const {
    impl_->same_device();
    if (!state || !payload) throw std::invalid_argument("STEPQuant: null state/payload");
    launch_reconstruct(state, static_cast<const unsigned char*>(payload), impl_->heads, impl_->H, stream);
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
    if (!plans.empty()) throw std::logic_error("STEPQuant: runtime already configured");
    if (state_size != S || heads < 1 || heads > 65535 || layers < 1 || interval < 1)
        throw std::invalid_argument("STEPQuant: unsupported model geometry");
    std::ifstream input(path);
    std::string magic;
    int version = 0, dim = 0, H = 0, n = 0;
    if (!(input >> magic >> version >> dim >> H >> n) || magic != "STRATA_STEPQUANT" || version != 1 ||
        dim != S || H != heads || n != layers-layers/interval)
        throw std::invalid_argument("STEPQuant: incompatible or truncated plan header");
    std::vector<StepQuantPlan> pending(layers);
    std::vector<std::array<uint64_t, 2>> pending_headers(layers);
    size_t slot = 0;
    for (int i = 0; i < n; ++i) {
        int layer = -1;
        if (!(input >> layer) || layer < 0 || layer >= layers || layer%interval == interval-1 || !pending[layer].bits.empty())
            throw std::invalid_argument("STEPQuant: invalid/duplicate/QSA layer in plan");
        StepQuantPlan plan;
        plan.bits.resize(heads); plan.impact.resize(size_t(heads)*S);
        for (int& b : plan.bits) if (!(input >> b)) throw std::invalid_argument("STEPQuant: truncated bits");
        for (float& w : plan.impact) if (!(input >> w)) throw std::invalid_argument("STEPQuant: truncated impact");
        StepQuantState validate(plan);
        pending_headers[layer] = {plan_hash(plan, layer), uint64_t(validate.bytes())};
        slot = std::max(slot, validate.bytes() + 16);
        pending[layer] = std::move(plan);

    }
    std::string trailing;
    if (input >> trailing) throw std::invalid_argument("STEPQuant: trailing plan data");
    std::vector<size_t> pending_slots(layers), pending_prefix{0};
    for (int l = 0; l < layers; ++l) if (!pending[l].bits.empty()) {
        pending_slots[l] = (pending_headers[l][1] + 16 + 255) / 256 * 256;
        pending_prefix.push_back(pending_prefix.back() + pending_slots[l]);
    }
    layer_slots = std::move(pending_slots);
    slot_prefix = std::move(pending_prefix);
    plans = std::move(pending);
    headers = std::move(pending_headers);
    runtime_slot = (slot + 255) / 256 * 256;
    runtime_heads = heads; runtime_interval = interval;
}
bool stepquant_enabled() { return !plans.empty(); }
bool stepquant_accepts_geometry(int layers, int interval, int heads, int state_size) {
    return plans.empty() || (layers == int(plans.size()) && interval == runtime_interval &&
                               heads == runtime_heads && state_size == S);
}
namespace {
StepQuantState& codec(int layer) {
    int device; check(cudaGetDevice(&device));
    auto it = devices.find(device);
    if (it == devices.end() || layer < 0 || size_t(layer) >= plans.size() || !it->second[layer])
        throw std::invalid_argument("STEPQuant: prepare session on this device before capture");
    return *it->second[layer];
}
std::vector<int> gdn_layers() {
    std::vector<int> result;
    for (size_t l = 0; l < plans.size(); ++l) if (!plans[l].bits.empty()) result.push_back(int(l));
    return result;
}

}
void stepquant_writeback(int layer, void* slot, const float* state, void* stream) {
    if (!plans.empty()) codec(layer).pack_to(state, static_cast<unsigned char*>(slot)+16, stream);
}
void stepquant_read(int layer, const void* slot, float* scratch, void* stream) {
    if (!plans.empty()) codec(layer).unpack_from(static_cast<const unsigned char*>(slot)+16, scratch, stream);
}
void stepquant_verify(int layer, void* slot, float* scratch, void* temporary,
                      const float* h, int channels, const float* gate, const float* beta,
                      const float* z, const float* gamma, float eps, float* y,
                      int key_heads, int heads, int tokens, const int32_t* keep, void* stream) {
    auto& c = codec(layer);
    if (heads != runtime_heads || key_heads < 1 || heads % key_heads || tokens < 1 || !temporary)
        throw std::invalid_argument("STEPQuant: invalid verification geometry");
    stepquant_read(layer, slot, scratch, stream);
    for (int t = 0; t < tokens; ++t) {
        const float* ht = h + size_t(t)*channels;
        fused_gdn_step_norm(scratch, ht, ht+S*key_heads, ht+2*S*key_heads,
                            gate+size_t(t)*heads, beta+size_t(t)*heads, z+size_t(t)*heads*S,
                            gamma, eps, y+size_t(t)*heads*S, key_heads, heads, stream);
        c.pack_to(scratch, temporary, stream);
        if (keep) {
            launch_commit(static_cast<unsigned char*>(slot)+16, static_cast<unsigned char*>(temporary), c.bytes(), keep, t+1, stream);
        }
        c.unpack_from(temporary, scratch, stream);
    }
}
size_t stepquant_recurrence_bytes() { return runtime_slot; }
size_t stepquant_recurrence_bytes(int layer) {
    if (layer < 0 || size_t(layer) >= layer_slots.size() || !layer_slots[layer])
        throw std::invalid_argument("STEPQuant: invalid recurrent layer");
    return layer_slots[layer];
}
size_t stepquant_session_bytes(size_t conv, int count, int ordinal) {
    if (count < 0 || ordinal < 0 || slot_prefix.empty() ||
        size_t(ordinal) >= slot_prefix.size() || size_t(count) > slot_prefix.size()-1-size_t(ordinal))
        throw std::invalid_argument("STEPQuant: invalid session range");
    const size_t slots = slot_prefix[ordinal+count] - slot_prefix[ordinal];
    if (count && conv > (SIZE_MAX-slots)/size_t(count))
        throw std::overflow_error("STEPQuant: session byte count overflow");
    return slots + size_t(count)*conv;
}
bool stepquant_bind_session(void* arena, size_t conv, int count, int ordinal) {
    if (plans.empty()) return true;
    const auto layers = gdn_layers();
    if (count < 0 || ordinal < 0 || size_t(ordinal+count) > layers.size() || (!arena && count)) return false;
    int device; check(cudaGetDevice(&device));
    if (!devices.count(device)) {
        std::vector<std::unique_ptr<StepQuantState>> states(plans.size());
        for (int l : layers) { states[l] = std::make_unique<StepQuantState>(plans[l]); states[l]->bind(nullptr); }
        devices.emplace(device, std::move(states));
    }
    sessions[arena] = {conv, count, ordinal, device};
    return true;
}
void stepquant_zero_session(void* arena, size_t conv, void* stream) {
    if (plans.empty()) return;
    auto it = sessions.find(arena);
    if (it == sessions.end() || it->second.conv != conv) throw std::invalid_argument("STEPQuant: unregistered session");
    auto layers = gdn_layers();
    for (int i = 0; i < it->second.count; ++i)
        check(cudaMemcpyAsync(static_cast<unsigned char*>(arena) + stepquant_session_bytes(conv, i, it->second.ordinal),
                              headers[layers[it->second.ordinal+i]].data(), 16, cudaMemcpyHostToDevice,
                              static_cast<cudaStream_t>(stream)));
}
bool stepquant_zero_layer(void* slot, void* stream) {
    if (plans.empty() || !slot) return false;
    const auto layers = gdn_layers();
    int current_device; check(cudaGetDevice(&current_device));
    const uintptr_t address = reinterpret_cast<uintptr_t>(slot);
    for (auto& entry : sessions) {
        if (entry.second.device != current_device) continue;
        const uintptr_t base = reinterpret_cast<uintptr_t>(entry.first);
        if (address < base) continue;
        for (int i = 0; i < entry.second.count; ++i) {
            if (address-base != stepquant_session_bytes(entry.second.conv, i, entry.second.ordinal)) continue;
            const int layer = layers[entry.second.ordinal+i];
            check(cudaMemsetAsync(slot, 0, layer_slots[layer], static_cast<cudaStream_t>(stream)));
            check(cudaMemcpyAsync(slot, headers[layer].data(), 16, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
            return true;
        }
    }
    return false;
}
void stepquant_unbind_session(void* arena) { sessions.erase(arena); }
bool stepquant_validate_session(const void* host, size_t bytes, size_t conv, int count, int ordinal, std::string& error) {
    if (plans.empty()) return true;
    const auto layers = gdn_layers();
    if (count < 0 || ordinal < 0 || size_t(ordinal+count) > layers.size() || (!host && count) || bytes != stepquant_session_bytes(conv, count, ordinal)) {
        error = "STEPQuant: incompatible packed session geometry"; return false;
    }
    for (int i = 0; i < count; ++i) {
        std::array<uint64_t, 2> header;
        std::memcpy(header.data(), static_cast<const unsigned char*>(host)+stepquant_session_bytes(conv, i, ordinal), 16);
        if (header != headers[layers[ordinal+i]]) { error = "STEPQuant: checkpoint uses a different calibration plan"; return false; }
    }
    return true;
}
namespace {
std::filesystem::path trace_directory;
int trace_every=8, trace_limit=512;
struct Trace { std::ofstream file; int count=0, hk=0, heads=0; };
std::map<int, Trace> traces;
}
bool stepquant_trace_enabled() { return !trace_directory.empty(); }
void stepquant_trace_configure(const std::string& directory, int every, int limit) {
    if (directory.empty() || every < 1 || limit < every || !plans.empty())
        throw std::invalid_argument("STEPQuant: calibration needs an unquantized run and positive sample limits");
    std::filesystem::create_directories(directory);
    trace_directory=directory; trace_every=every; trace_limit=limit; traces.clear();
}
void stepquant_observe(int layer, const float* h, const float* gate, const float* beta,
                       const float* state, int hk, int H, void* stream) {
    if (trace_directory.empty()) return;
    auto& trace=traces[layer];
    if (trace.count >= trace_limit) return;
    if (hk < 1 || H < 1 || H % hk || !plans.empty()) throw std::invalid_argument("STEPQuant: invalid calibration run");
    if (!trace.file.is_open()) {
        auto path=trace_directory/("layer-"+std::to_string(layer)+".bin");
        if (std::filesystem::exists(path)) throw std::runtime_error("STEPQuant: trace file already exists: "+path.string());
        trace.file.open(path,std::ios::binary);
        const int32_t header[]={0x53515452,1,H,hk,S,trace_every,trace_limit};
        trace.file.write(reinterpret_cast<const char*>(header),sizeof(header));
        trace.hk=hk; trace.heads=H;
    }
    if (hk != trace.hk || H != trace.heads) throw std::invalid_argument("STEPQuant: trace geometry changed");
    check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    const int32_t sampled=(++trace.count % trace_every)==0;
    trace.file.write(reinterpret_cast<const char*>(&sampled),4);
    auto copy=[&](const float* source, size_t count) {
        std::vector<float> host(count);
        check(cudaMemcpy(host.data(),source,count*4,cudaMemcpyDeviceToHost));
        trace.file.write(reinterpret_cast<const char*>(host.data()),count*4);
    };
    copy(h,size_t(2)*hk*S); copy(gate,H); copy(beta,H);
    if (sampled) copy(state,size_t(H)*S*S);
    trace.file.flush();
    if (!trace.file) throw std::runtime_error("STEPQuant: calibration trace write failed");
}
void stepquant_release() {
    traces.clear(); trace_directory.clear();
    layer_slots.clear(); slot_prefix.clear();
    devices.clear(); plans.clear(); headers.clear(); sessions.clear(); runtime_slot = 0; runtime_heads = 0; runtime_interval = 0;
}
} // namespace strata::kernels
