#pragma once
// CUDA-shaped test operations backed by the same DPCT queues/graphs as Strata.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
using cudaError_t = int;
using cudaStream_t = sycl::queue*;
using cudaGraph_t = dpct::experimental::command_graph_ptr;
using cudaGraphExec_t = dpct::experimental::command_graph_exec_ptr;
constexpr int cudaSuccess=0, cudaMemcpyDeviceToHost=0, cudaMemcpyHostToDevice=1;
constexpr int cudaStreamCaptureModeGlobal=0;
inline const char* cudaGetErrorString(int) { return "SYCL operation failed"; }
inline int cudaGetDeviceCount(int* count) { *count=0; for (auto& device: sycl::device::get_devices()) if (device.is_gpu()) ++*count; return 0; }
inline int cudaMalloc(void** p, size_t bytes) { *p=sycl::malloc_device(bytes,dpct::get_in_order_queue()); return *p ? 0 : 1; }
inline int cudaFree(void* p) { sycl::free(p,dpct::get_in_order_queue()); return 0; }
inline int cudaMemcpy(void* dst,const void* src,size_t bytes,int) { dpct::get_in_order_queue().memcpy(dst,src,bytes).wait_and_throw(); return 0; }
inline int cudaMemcpyAsync(void* dst,const void* src,size_t bytes,int,cudaStream_t q) { strata::q_of(q)->memcpy(dst,src,bytes); return 0; }
inline int cudaStreamCreate(cudaStream_t* q) { *q=dpct::get_current_device().create_queue(true); return 0; }
inline int cudaStreamDestroy(cudaStream_t q) { if(q) dpct::get_current_device().destroy_queue(q); return 0; }
inline int cudaStreamSynchronize(cudaStream_t q) { strata::q_of(q)->wait_and_throw(); return 0; }
inline int cudaStreamBeginCapture(cudaStream_t q,int) { dpct::experimental::begin_recording(q); return 0; }
inline int cudaStreamEndCapture(cudaStream_t q,cudaGraph_t* graph) { dpct::experimental::end_recording(q,graph); return 0; }
inline int cudaGraphInstantiate(cudaGraphExec_t* out,cudaGraph_t graph,void*,void*,int) {
    *out=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize()); return 0;
}
inline int cudaGraphLaunch(cudaGraphExec_t graph,cudaStream_t q) { q->ext_oneapi_graph(*graph); return 0; }
inline int cudaGraphExecDestroy(cudaGraphExec_t graph) { delete graph; return 0; }
inline int cudaGraphDestroy(cudaGraph_t graph) { delete graph; return 0; }
