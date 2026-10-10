# STEPQuant GDN state quantization (opt-in)

[STEPQuant](https://github.com/Dreamer-Toby/STEPQuant) quantizes the recurrent
state of Delta-rule attention. It does not quantize model weights or the QSA
K/V cache. This port implements the GDN dual-axis fit from
[the paper](https://arxiv.org/abs/2609.38169), with per-head INT2/4/6/8 allocation,
FP16 pivot heads, FP16 row and column scales, and groups of 32 values for INT2.
Allocation and row impact come from a frozen calibrated plan. Strata can collect
its own unquantized traces and fit the plan, or import an upstream artifact.

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

The engine supports plain decode, speculative verification, the Python server,
batched independent sessions, layer splits and pipeline snapshots. Verification
reconstructs the initial packed state and quantizes after every proposed token.
Proposals leave persistent state untouched. Commit replays the recurrence and
copies exactly the accepted prefix, with a device counter captured in the graph;
zero accepted tokens leave the slot unchanged. A one-token self-commit uses the
same rule. This path uses the native fused GDN step/norm and quantizes its output
separately when QFUSE is enabled.

Each session owns its FP32 scratch and temporary packed payload. Plans are
replicated on each participating GPU before graph capture; no mutable codec is
bound globally to a session. Range sessions map their global GDN ordinals to the
same layer plan and checkpoint fingerprint.

CUDA and HIP use the same device kernel. SYCL has a work-group implementation
of the fit and reconstruction with the same shared host runtime. CUDA was built
and tested here. HIP/MI50 and SYCL source support has been added, but their builds
could not be run in this environment: ROCm and `icx`/`icpx` are absent. They need
backend compilation and device validation before being considered validated.

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

Calibrate **the exact checkpoint** and its head ordering. Strata pairs value
heads with key heads by modulo. Geometry validation alone cannot identify a
foreign checkpoint. The collector saves post-convolution normalized q/k, decay,
beta and FP32 state samples from the model actually loaded by Strata. It runs
prompt recurrence one token at a time to collect intermediate states. Collection
requires `--native` and batched prefill; IQ packs also require `--spec 2` or
higher. Calibration collects the prompt, rather than speculative proposal
states. Use representative prompts with at least eight tokens processed by
prefill (normally at least nine prompt tokens, because decode handles the last).

```sh
build-stepquant/strata --pack /path/to/pack --native /path/to/shard1.gguf \
  --tokens-file calibration.ids --prefill 128 --spec 2 --max-new 1 \
  --stepquant-trace /path/to/new-trace-directory
python tools/stepquant_calibrate.py /path/to/new-trace-directory checkpoint.plan \
  --bits 6 --pivots 32 --layers 48 --qsa-interval 4 --heads 48
build-stepquant/strata --pack /path/to/pack --native /path/to/shard1.gguf \
  --tokens-file prompt.ids --prefill 128 --spec 4 --stepquant-plan checkpoint.plan
```

The collector is an opt-in synchronous calibration path. It retains at most
512 tokens per layer and samples a state every eight tokens, yielding at most
64 snapshots per layer. It refuses existing layer trace files and cannot run
with quantized feedback, serving, batching or pipeline windows. Generation
with no trace uses the existing prompt kernel. The offline calibrator requires
NumPy and independently implements transported row impact, lifetime weights,
weighted candidate distortions, exact global head dynamic programming, residual
risk ranking and the second allocation after selecting FP16 pivots. It writes
a versioned text plan and a JSON report with trace/plan hashes and token counts.
The nominal 4/6-bit budget counts each FP16 pivot as eight bits, as STEPQuant
does; actual payload storage also includes FP16 pivots and scales.

An upstream version-2 artifact can still be imported with PyTorch:

```sh
python tools/stepquant_plan.py checkpoint-plan.pt checkpoint.plan \
  --layers 48 --qsa-interval 4 --heads 48
```

For the Python server, add `"stepquant_plan": "/absolute/path/checkpoint.plan"`
to its config and use the engine built with the option enabled. An explicit
`--stepquant-plan` in `args` takes precedence. Speculation, batch scheduling,
prompt-cache checkpoints, reset and disk sessions use the packed state. The
normal configuration remains unchanged when no plan is supplied.

The exporter and engine reject missing, duplicate or QSA layer entries, wrong
shapes, unsupported precision, nonpositive/nonfinite impact factors and
truncated plans. Allocation is global: do not calibrate
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
python tools/test_stepquant_calibrate.py
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
| Packed GDN state | 40,634,368 |
| Difference | 86,753,024 |

This is an allocation test without model weights, not a full-model serving
benchmark or a nominal STEPQuant@4/@6 calibration. Both arenas include the
same FP32 scratch, QSA buffers and other session state; packed mode also includes
a temporary packed slot for speculative recurrence. The test also saves,
validates, corrupts and restores a running checkpoint. GPU graph replay, independent/range sessions, every accepted-prefix length,
invalid plans, bounded traces and zero/pivot states pass.

No task-quality benchmark or other GPU backend has been validated. With the STEPQuant blocks disabled, every changed engine source and
the GDN buffer declaration are byte-identical to this branch's `origin/main`
base; `test_stepquant_default.py` checks that explicitly.

CPU calibration tests compare the global optimum with exhaustive assignments,
check pivot budgets, parse the engine trace layout and reject truncation. Impact,
lifetime, exact DP and nominal @4/@6 allocations also matched the external
STEPQuant reference on a two-layer synthetic calibration fixture.

A full-model CUDA smoke test used the local IQ3_XXS Flash-Next pack on the RTX
4070 Ti SUPER with CUDA 13.4. An 18-token prompt produced traces for all 36 GDN
layers and an @6 plan (32 FP16 pivots, 424 INT4, 880 INT6 and 392 INT8 heads).
With that plan, 12 generated tokens completed through captured MTP verification
windows; 8 of 9 draft tokens were accepted. This small calibration validates
integration, not task quality or speed relative to FP32. It is not a production
calibration corpus.

The Python server smoke test ran on `127.0.0.1` with the same model and plan,
`parallel: 2`, a 1,024-token context and 64-token prefill chunks. Two simultaneous
OpenAI chat requests returned HTTP 200 with 16 tokens each. The engine captured
windows over both slots and reused a 64-token prompt checkpoint. The replies
reached the test's short token limit during reasoning, so this is a scheduling
and state-storage check, not answer-quality evidence. The temporary server was
stopped after the test. All 273 Python server tests passed.
