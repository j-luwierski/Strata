# STEPQuant full-model quality check, 2026-10-10

This compares the full Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS model with FP32 GDN
state, STEPQuant@6 and STEPQuant@4. All 48 model layers run, including all 36
GDN layers. The weights are IQ3_XXS in every run; “FP32” refers to recurrent
state, not model weights.

Hardware: NVIDIA RTX 4070 Ti SUPER, 16,376 MiB, driver 610.57.04, CUDA 13.4;
AMD Ryzen 9 5900X, 125 GiB system RAM, Linux. The binary SHA-256 and complete
commands are in the comparison JSON files. CUDA was built with STEPQuant
compiled in; FP32 runs omit the runtime plan.

## Method

The installer collected 512 unquantized prompt updates per GDN layer from
`data/stepquant-calibration.txt`, sampling a matrix every eight tokens. The
calibrator used a nominal budget of 4 or 6 bits, 32 FP16 pivot heads and a
2,048-token lifetime horizon. It calibrated the actual model, rather than a
small replacement model. The calibration metadata records corpus and plan
hashes and precision counts.

The separate, original held-out corpus is `bench/stepquant/heldout.json`:
English narrative, Polish narrative, code, science, arithmetic and a chat
continuation. Six documents contain 1,078 input tokens. Their 1,072 next-token
positions were teacher-forced through the native verifier. The first 32
positions of each document were excluded from scoring, leaving 880 positions.
Every consumed token updates and packs the GDN state in the STEPQuant runs.
This exercises quantized feedback repeatedly, rather than comparing only the
last logit of an FP32 prefill chunk.

Each pair uses the same executable, weights, tokens, context, expert profile
and fixed cache budget (`--expert-cache 4000`, 5,357 actual profile slots).
`--pcie-frac 0` keeps uncached experts on the CPU; cached experts run on the
GPU. `--short-read 2048` forces verifier windows, `--prompt-cache 0` prevents
prefix reuse, and `--spec 2` enables the native verifier. There is no MTP
checkpoint in this test configuration. The tool verifies the target token and
position of every recorded probability and refuses incomplete/nonfinite data.

Automatic cache sizing was used in an initial exploratory run. That was a
confound: packing state changes free VRAM and hence expert placement, and CPU
and GPU experts round differently. Those exploratory results are excluded
here. With fixed placement, the two independent FP32 logprob files are
byte-identical (SHA-256
`a77f4bfc445c637c0a98573c23061ec322e0401588856fef54e969abcac24b54`).

## Results

Lower perplexity is better. Top-1 agreement compares the most likely token
with FP32 on the same 880 positions. Next-token accuracy compares it with the
corpus's actual next token; it is not a multiple-choice task score.

| GDN state | Mean NLL | Perplexity | Change in perplexity | Top-1 agreement with FP32 | Next-token accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1.637611 | 5.142867 | — | 100% | 58.30% |
| STEPQuant@6 | 1.635210 | 5.130533 | -0.24% | 95.11% | 58.64% |
| STEPQuant@4 | 1.648816 | 5.200818 | +1.13% | 89.55% | 57.84% |

On this small corpus @6 stays close to FP32; @4 changes more predictions and
has higher perplexity. The small @6 decrease does not establish better general
quality. No MMLU, HellaSwag, coding pass rate, long-context accuracy or
statistical confidence claim follows from these six documents. The longest
input is 212 tokens; the 2,048-token configured context is not a long-context
test. There are no quality measurements for HIP or SYCL.

Raw probabilities, engine logs, per-run JSON and calibration metadata are
included beside this report. The diagnostic D2H copies and host probability
calculation add work; wall times in the JSON are not production speed
benchmarks.

## Repeat

Use a config pointing to a STEPQuant-enabled CUDA executable and these exact
IQ3_XXS weights. For each plan:

```sh
.venv/bin/python tools/stepquant_quality.py \
  --config /path/to/model-config.json --plan /path/to/calibrated.plan \
  --output /path/to/results --expert-cache 4000
```

For a smaller card lower `--expert-cache`, using the same value for every run.
The config's `stepquant_plan` is ignored for the FP32 half of the pair. The
chosen plan is used only for its STEPQuant half. The tool does not change the
installed config. Calibrate through `./setup.sh --stepquant on --yes --no-start`
(or `--stepquant-bits 4 --stepquant-recalibrate`), as described in
`docs/STEPQUANT.md`.

## Validation and limits

The installer STEPQuant tests (14) and Intel adapter tests (15) pass without a
GPU or downloads. All 273 Python server tests pass. CUDA kernel checks pass for
packed storage, pivots, graph replay, trace bounds, independent sessions,
accepted speculative prefixes and checkpoint restore. `setup.sh` passes shell
syntax checking and exposes the new flags in `--help`.

The full installer suite runs 433 tests: 430 pass, one is skipped, and two
existing `test_setup_engine_hash.WhatARefusalDoesToTheCaller` tests fail. Both
failures were reproduced using `setup.py` from commit `3530b742`, before this
installer change. They are not counted as successful validation. The unchanged
engine default-source check passes for all 13 guarded engine/header files.

No HIP, SYCL or CUDA 12 device validation was possible on this NVIDIA/CUDA 13
machine. The installer keeps STEPQuant enabled when rebuilding or migrating to
the CUDA 12 engine, but that migration was tested with mocks only.
