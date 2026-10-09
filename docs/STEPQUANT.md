# STEPQuant GDN state quantization (experimental CUDA port)

[STEPQuant](https://github.com/Dreamer-Toby/STEPQuant) quantizes the recurrent
state of Delta-rule attention. It does not quantize model weights or the QSA
K/V cache. This port implements the GDN dual-axis fit from
[the paper](https://arxiv.org/abs/2609.38169), with per-head INT2/4/6/8 allocation,
FP16 pivot heads, FP16 row and column scales, and groups of 32 values for INT2.
Allocation and row impact come from a frozen upstream calibration plan. No
uniform allocation is substituted for calibration.

The CUDA implementation is written independently. The external reference used
for validation is STEPQuant commit `61f24c9c2bc188b60a6c7525e7f86c7459bc737d`.
Upstream Python/Triton/SGLang code is not bundled with Strata.

## Current scope

There are two interfaces:

- `strata::kernels::StepQuantState` owns a packed device state. Its `step`
  unpacks into caller-owned FP32 scratch, runs the native GDN recurrence,
  computes the readout from the FP32 update, and packs the next persistent
  state. INT6 codes use six bits per element, rather than an INT8 surrogate.
  Construction allocates the storage; graph replay allocates nothing.
- `strata --stepquant-plan FILE` uses packed persistent GDN state slots in the
  session arena, with one shared FP32 matrix for the layer currently running.
  It quantizes state after each decode update and at each batched-prefill chunk
  boundary. Readouts within that prompt chunk stay FP32. With token-by-token
  prompt processing, every prompt token is quantized. Prefill chunk size can
  therefore change outputs in this experimental mode.

Session allocation, convolution offsets, checkpoint sizes and snapshot/read
limits use the packed slot size. Slots use the largest layer payload in the
plan, rounded to 256 bytes, so layers with a smaller payload have padding.
Convolution, PLE and QSA states retain their existing formats and precision.
Checkpoints and disk sessions preserve packed bytes. Each slot carries the
plan fingerprint and payload size; restore rejects a different calibration
plan before writing device state. A checkpoint from the FP32 mode is not a
checkpoint for this mode. No speed or task-quality result is claimed.

Only native CUDA is implemented. HIP, gfx906 and SYCL builds reject the build
option. One active session on one device with plain `generate` decode is supported;
speculation, batch decoding, layer splits and pipeline windows are rejected.
This includes the verifier API, so unsupported callers cannot silently verify
using FP32 state. `--serve` is also rejected: the current server requires that
verifier even when it emits one token at a time. The checkpoint/snapshot APIs
understand packed states; server scheduling and speculative verification still
need their own port before the Python server can enable this option.

## Build and use

The build option defaults to off, so release builds contain no added kernels,
allocations, arguments or token-path calls.

```sh
cmake -S . -B build-stepquant \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_STEPQUANT=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-stepquant --target strata stepquant_test -j 8
```

Choose the architecture for your GPU. The command above was built for an RTX
4070 Ti SUPER (`sm_89`) on Linux with CUDA 13.4.

Calibrate **the exact checkpoint** using upstream STEPQuant, then export its
version-2 `.pt` artifact. The exporter requires PyTorch; the engine does not.
A calibration from Qwen3.8-27B is not a calibration for Flash-Next. The supplied
upstream adapters are not a Strata trace collector. Check the attention's value
head ordering as well as its geometry when preparing a calibration: Strata pairs
value heads with key heads by modulo. Geometry validation alone cannot identify
a foreign checkpoint or a differently ordered plan.

```sh
python tools/stepquant_plan.py checkpoint-plan.pt checkpoint-plan.strata \
  --layers 48 --qsa-interval 4 --heads 48
build-stepquant/strata generate --pack /path/to/pack --tokens-file prompt.ids \
  --spec 0 --stepquant-plan checkpoint-plan.strata
```

The exporter and engine reject missing, duplicate or QSA layer entries, wrong
shapes, unsupported precision, nonpositive/nonfinite impact factors and
truncated plans. Allocation is global in upstream calibration: do not calibrate
and allocate each layer independently if reproducing its nominal 4/6-bit budget.
FP16 pivots and scales add storage beyond that nominal budget.

The text plan starts with `STRATA_STEPQUANT 1 128 HEADS GDN_LAYERS`. Each layer
then has its zero-based model index, one precision per value head, and
`HEADS * 128` impact factors in `(head, key)` order. Layer indices are model
indices, not compressed GDN ordinals. Runtime state addressing is Strata's
`(key, head, value)`, not upstream's `(head, key, value)`.

## Validation

```sh
ctest --test-dir build-stepquant -R '^stepquant_test$' --output-on-failure
python tools/test_stepquant_plan.py
python tools/test_stepquant_default.py
python tools/test_stepquant_reference.py \
  --upstream /path/to/STEPQuant --binary build-stepquant/stepquant_test
compute-sanitizer --tool memcheck --error-exitcode 1 build-stepquant/stepquant_test
```

The external-reference check covers all five precisions, nonuniform impact,
INT2 group boundaries, outliers, zero rows and 256 recurrent updates. Each update
uses the actual previous CUDA packed state as the starting state for both
implementations. It checks the FP32 update/readout, fitted FP16 scales, nearest
integer codes and exact reconstruction from the stored payload. Parallel
reductions can select different FP16 scales at rounding boundaries; the checker
reports reconstruction differences instead of promising byte-identical fits.
The native CUDA update/readout is checked with `atol=1e-6/1e-5, rtol=1e-4`; row
and column scales allow one FP16 ulp plus subnormal spacing. A level may differ
from upstream at a FP16 scale decision boundary; exact reconstruction and
nearest-code selection are checked against the scales actually stored.

In the synthetic five-head test (one head each at 2/4/6/8/16 bits, 128 by 128),
the packed payload is 76,544 bytes, versus 327,680 bytes of FP32 state. Those
counts exclude shared plan metadata and FP32 scratch.

On Linux, CUDA 13.4, RTX 4070 Ti SUPER 16 GB, the synthetic session test with
Strata's full default geometry, 16 K/V cells, and a repeating 2/4/6/8/16-bit
head allocation measured these arena sizes:

| Session arena | Bytes |
| --- | ---: |
| FP32 GDN state | 127,387,392 |
| Packed GDN state | 39,918,336 |
| Difference | 87,469,056 |

This is an allocation test without model weights, not a full-model serving
benchmark or a nominal STEPQuant@4/@6 calibration. Both arenas include the
same shared scratch, QSA buffers and other session state. The test also saves,
validates, corrupts and restores a running checkpoint. CUDA graph replay,
invalid plans and zero/pivot states pass; compute-sanitizer reports no memory
errors.

No full-model generation, task-quality benchmark or other GPU backend has been
validated. With the STEPQuant blocks disabled, every changed engine source and
the GDN buffer declaration are byte-identical to this branch's `origin/main`
base; `test_stepquant_default.py` checks that explicitly.
