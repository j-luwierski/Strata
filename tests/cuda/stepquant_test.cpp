#include "strata/kernels/stepquant.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/core/session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace strata::kernels;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc((void**)&p, n*sizeof(float))); }
    ~Buffer() { cudaFree(p); }
    void upload(const std::vector<float>& x) { ck(cudaMemcpy(p,x.data(),x.size()*4,cudaMemcpyHostToDevice)); }
    std::vector<float> download(size_t n) const {
        std::vector<float> x(n); ck(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost)); return x;
    }
};
template<typename T> std::vector<T> read(std::ifstream& f, size_t n) {
    std::vector<T> x(n);
    require(bool(f.read(reinterpret_cast<char*>(x.data()),n*sizeof(T))),"truncated fixture"); return x;
}
float compare(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    float max_error = 0;
    require(actual.size()==expected.size(), "reference size mismatch");
    for (size_t i=0; i<actual.size(); ++i) {
        require(std::isfinite(actual[i]), "nonfinite result");
        float error=std::abs(actual[i]-expected[i]); max_error=std::max(max_error,error);
        if (error > tolerance*(1+std::abs(expected[i]))) {
            std::fprintf(stderr,"element %zu: actual %.9g expected %.9g error %.9g\n",i,actual[i],expected[i],error);
            throw std::runtime_error("reference mismatch");
        }
    }
    return max_error;
}
void reference(const char* path, const char* trace_path, cudaStream_t stream) {
    std::ofstream trace(trace_path, std::ios::binary);
    require(bool(trace), "cannot create trace");
    auto record = [&](const std::vector<float>& x) {
        require(bool(trace.write(reinterpret_cast<const char*>(x.data()),x.size()*4)), "trace write failed");
    };
    std::ifstream f(path, std::ios::binary);
    auto header=read<int32_t>(f,3);
    const int H=header[0], HK=header[1], steps=header[2];
    require(H>0 && H<=128 && HK>0 && H%HK==0 && steps>=0 && steps<=4096,"invalid fixture header");
    StepQuantPlan plan;
    auto bits=read<int32_t>(f,H); plan.bits.assign(bits.begin(),bits.end()); plan.impact=read<float>(f,H*128);
    const size_t N=size_t(H)*128*128;
    Buffer state(N),q(HK*128),k(HK*128),v(H*128),gate(H),beta(H),output(H*128);
    StepQuantState packed(plan);
    auto record_payload = [&] {
        std::vector<unsigned char> p(packed.bytes());
        ck(cudaMemcpy(p.data(),packed.data(),p.size(),cudaMemcpyDeviceToHost));
        require(bool(trace.write(reinterpret_cast<const char*>(p.data()),p.size())), "payload trace failed");
    };
    state.upload(read<float>(f,N));
    packed.writeback(state.p,stream); ck(cudaStreamSynchronize(stream));
    record(state.download(N)); record_payload();
    for(int t=0;t<steps;++t) {
        q.upload(read<float>(f,HK*128)); k.upload(read<float>(f,HK*128)); v.upload(read<float>(f,H*128));
        gate.upload(read<float>(f,H)); beta.upload(read<float>(f,H));
        packed.step(state.p,q.p,k.p,v.p,gate.p,beta.p,output.p,HK,stream);
        ck(cudaStreamSynchronize(stream));
        record(output.download(H*128));
        record(state.download(N)); // full-precision update actually passed to the fit
        packed.unpack(state.p,stream); ck(cudaStreamSynchronize(stream));
        record(state.download(N)); record_payload();
    }
    require(f.peek()==EOF,"trailing fixture data");
    std::printf("packed recurrent trace: heads=%d steps=%d payload=%zu\n",H,steps,packed.bytes());
}
void session_storage(cudaStream_t stream, bool canonical) {
    using namespace strata::core;
    const auto path = std::filesystem::temp_directory_path()/
        ("strata-stepquant-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".plan");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { stepquant_release(); std::error_code e; std::filesystem::remove(path,e); } } cleanup{path};
    ModelGeometry g;
    if (!canonical) {
        g.n_layers=2; g.qsa_interval=2; g.ssm_v_heads=5; g.ssm_k_heads=1;
        g.ssm_value_dim=5*128; g.ssm_conv_channels=7*128;
    }
    const size_t H=(size_t)g.ssm_v_heads, layers=(size_t)g.n_gdn_layers(), elements=H*128*128;
    {
        std::ofstream f(path);
        f << "STRATA_STEPQUANT 1 128 " << H << " " << layers << "\n";
        for(int l=0;l<g.n_layers;++l) if (!is_qsa_layer(g,l)) {
            f << l << "\n";
            for(size_t h=0;h<H;++h) f << std::vector<int>{2,4,6,8,16}[h%5] << " ";
            f << "\n";
            for(size_t i=0;i<H*128;++i) f << "1 ";
            f << "\n";
        }
    }
    const size_t dense_bytes=session_bytes(g,16,10);
    stepquant_configure(path.string(),(int)g.n_layers,(int)g.qsa_interval,(int)H,128);
    require(stepquant_accepts_geometry((int)g.n_layers,(int)g.qsa_interval,(int)H,128),"plan rejected matching geometry");
    require(!stepquant_accepts_geometry((int)g.n_layers,(int)g.qsa_interval,(int)H+1,128),"plan accepted foreign heads");
    const size_t packed_bytes=session_bytes(g,16,10);
    require(packed_bytes < dense_bytes, "session allocation did not shrink");
    Buffer arena((packed_bytes+3)/4);
    SessionState ss;
    const auto used=session_init(g,16,10,arena.p,ss);
    require(used>0 && used<=packed_bytes,"packed session carve out of bounds");
    session_zero(ss,g,nullptr,stream); ck(cudaStreamSynchronize(stream));
    ConversationStateSizes sizes; std::string error;
    require(conversation_session_sizes(g,ss,sizes,error),error);
    const size_t conv=(size_t)g.ssm_conv_channels*(g.ssm_d_conv-1)*4;
    require(sizes.gdn==layers*(stepquant_recurrence_bytes()+conv),"snapshot byte estimate ignored packed stride");
    Buffer dense(elements);
    stepquant_read(0,dense.p,stream); ck(cudaStreamSynchronize(stream));
    compare(dense.download(elements),std::vector<float>(elements,0.f),0.f);
    std::vector<float> x(elements);
    for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*0.01f)*0.1f;
    dense.upload(x); stepquant_writeback(0,dense.p,stream); stepquant_read(0,dense.p,stream); ck(cudaStreamSynchronize(stream));
    const auto expected=dense.download(x.size());
    ConversationCheckpoint checkpoint; checkpoint.ids={1};
    require(conversation_checkpoint_save(checkpoint,ss,g,error),error);
    require(conversation_checkpoint_validate(checkpoint,ss,g,error),error);
    auto corrupted=checkpoint; corrupted.gdn[0]^=1;
    require(!conversation_checkpoint_validate(corrupted,ss,g,error),"foreign plan header accepted");
    auto b=ss.gdn; b.state=ss.gdn_state; b.conv_state=ss.gdn_state+stepquant_recurrence_bytes()/4;
    gdn_buffers_zero_state(b,g,stream); ck(cudaStreamSynchronize(stream));
    stepquant_read(0,dense.p,stream); ck(cudaStreamSynchronize(stream));
    compare(dense.download(x.size()),std::vector<float>(x.size(),0.f),0.f);
    require(conversation_checkpoint_restore(checkpoint,ss,g,error),error);
    stepquant_read(0,dense.p,stream); ck(cudaStreamSynchronize(stream));
    compare(dense.download(x.size()),expected,0.f);
    session_release(ss); delete[] ss.qsa_states;
    std::printf("packed session + checkpoint restore: dense=%zu packed=%zu saved=%zu bytes\n",dense_bytes,packed_bytes,dense_bytes-packed_bytes);
}
void basic(cudaStream_t stream) {
    StepQuantPlan plan{{2,4,6,8,16},std::vector<float>(5*128,1.f)};
    StepQuantState packed(plan);
    const size_t N=5*128*128;
    require(packed.bytes()==76544,"tight mixed-precision payload byte count");
    Buffer state(N), restored(N);
    std::vector<float> x(N);
    for (int row=0;row<128;++row) for(int h=0;h<5;++h) for(int col=0;col<128;++col)
        x[(row*5+h)*128+col]=h==4 ? 1.5f : 0.f;
    state.upload(x);
    packed.pack(state.p,stream); packed.unpack(restored.p,stream); ck(cudaStreamSynchronize(stream));
    auto actual=restored.download(N);
    for (int row=0;row<128;++row) for(int h=0;h<5;++h) for(int col=0;col<128;++col) {
        const size_t i=(row*5+h)*128+col;
        require(h==4 ? actual[i]==1.5f : std::abs(actual[i])<1.e-12f,"zero/pivot reconstruction");
    }
    // A captured fit/reconstruction must agree with ordinary launches, and allocate nothing on replay.
    for(size_t i=0;i<N;++i) x[i]=std::sin(float(i)*0.018f)*0.2f;
    state.upload(x); packed.writeback(state.p,stream); ck(cudaStreamSynchronize(stream));
    auto expected=state.download(N); state.upload(x);
    cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    packed.writeback(state.p,stream); ck(cudaStreamEndCapture(stream,&graph));
    ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)); ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
    compare(state.download(N),expected,0.f);
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
    for(int b : {0,3,5,7,32}) {
        bool rejected=false;
        try { StepQuantState invalid({{b},std::vector<float>(128,1.f)}); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid bits accepted");
    }
    bool rejected=false;
    try { StepQuantState invalid({{4},std::vector<float>(128,-1.f)}); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid impact accepted");
    std::printf("packed storage, zero states, pivots, CUDA graphs and invalid plans: PASS\n");
}
}
int main(int argc,char** argv) {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 77;
    cudaStream_t stream=nullptr;
    try {
        ck(cudaStreamCreate(&stream)); basic(stream); session_storage(stream,false); session_storage(stream,true);
        if(argc==5 && std::string(argv[1])=="--reference" && std::string(argv[3])=="--trace") reference(argv[2],argv[4],stream);
        else require(argc==1,"usage: stepquant_test [--reference fixture.bin --trace results.bin]");
        ck(cudaStreamDestroy(stream)); return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"STEPQuant test: %s\n",e.what()); cudaStreamDestroy(stream); return 1;
    }
}
