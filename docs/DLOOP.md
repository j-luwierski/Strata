# DLoop (experimental, opt-in)

[NAVER DLoop](https://arxiv.org/abs/2610.07659) repeats complete drafting stages before one target verification.
After each block, it sums the natural logarithms of the draft probabilities. If the latest block's score is at
least the gate, another block follows. The score resets at each block; an uncertain token cannot be compensated
by a confident one. The target still verifies every emitted token using the existing acceptance rule.

This implementation uses the available **stock MTP weights**. The paper also retrains the drafter to consume its
own features across additional stages. The [official repository](https://github.com/naver-ai/DLoop) has not yet
released that implementation or those weights. Strata does not reproduce the paper's trained checkpoints or
claim its speedups. A selected draft vocabulary normalizes confidence over that vocabulary, not the full head.

## Setup

`setup.sh` passes its arguments to `setup.py`, which owns the installer questions and saves the engine flags in
`strata-<model>.json`. Fresh interactive setup asks whether to use DLoop (default **no**), then the block size,
maximum loop count and gate. It uses the existing MTP files and compiles a DLoop-capable engine when necessary.
No extra draft weights are downloaded. These noninteractive commands use the same path:

```sh
# New installation or explicit reconfiguration:
./setup.sh --setup --dloop on --dloop-block-size 3 --dloop-max-loops 2 --dloop-gate=-0.5 --yes --no-start

# Configure an installed MTP model, without downloading its model again:
./setup.sh --dloop on --yes --no-start

# Compare loop settings with the same drafter without loops and save the result:
./setup.sh --dloop-calibrate --yes --no-start

# Restore the earlier speculation flags:
./setup.sh --dloop off --yes --no-start
```

With several models installed, setup asks which one to edit; `--yes` selects the first. Enabling a different
kind of drafter is refused. The separate DFlash integration described below is needed for a DFlash installation.
The model's existing config is backed up before a change. Turning DLoop off restores the earlier `--spec`,
`--spec-min-p` and `--mtp-max-t`, while retaining unrelated settings changed in the meantime.

## Parameters and limits

| Setup option | Engine option | Meaning |
| --- | --- | --- |
| `--dloop on/off` | `--dloop` | Enable the block loop; default off |
| `--dloop-block-size N` | same | Drafts per full block; default 3 |
| `--dloop-max-loops N` | same | Upper bound on blocks per verification; default 2 |
| `--dloop-gate G` | same | Continue when the latest block's `sum(log(q)) >= G`; default -0.5 |
| `--dloop-calibrate` | none | Measure choices and save the result |

The existing native verifier has at most eight rows: one anchor and **seven drafts**. Therefore
`block_size * max_loops <= 7`. Three drafts and two loops permit six drafts, verified with the anchor in a
seven-row window. More negative gates permit less confident extensions; zero extends only a probability-one
block. A one-loop configuration is useful for checking the unchanged single-block behavior.

DLoop supersedes the per-token `--spec-min-p` gate, which otherwise could cut a complete block. Prompt lookup
keeps its existing policy. Use `--suffix-draft 0` when measuring the loop alone. Serial CLI generation and the
serial server path are supported. Batch slots, `--pipeline-windows 2`, lookup chains and oracle drafts are
refused. A last verification can consume a shorter prefix at the target context boundary.

To build manually, add `-DSTRATA_ENABLE_DLOOP=ON` to the normal CUDA, HIP or SYCL build command. It defaults to OFF;
the engine sources with the DLoop blocks removed were checked against main at `fb58e0db` and were identical.
CUDA was built and run. HIP and SYCL source support is present, but neither could be compiled on the validation
PC: ROCm and `icpx` were absent. An Intel setup checks the actual container binary's capability without touching
a running serving container. Windows HIP needs an engine already built with this option.

## Calibration

Setup offers DLoop calibration after enabling it. `--calibrate` also runs it after the usual hardware tuning
when DLoop is enabled. The loop calibration loads an engine for each arm, warms its graphs, then runs three
prompts twice in alternating arm order. It tries blocks of two and three with several gates, plus the requested
configuration. Maximum loop counts are bounded by the verifier's capacity.

Each arm uses a fixed expert-cache byte budget, disables adaptive swaps and prefix sharing, and reads the
prompt with `STRATA_PREFILL_CPU_SHARE=0` so placement and prompt scheduling cannot decide the comparison. For an
automatically sized cache, a largest-window probe determines a fixed budget that fits. A winning DLoop
configuration saves that fixed cache budget alongside the loop options.

An arm must reproduce **all baseline greedy token IDs**, including early end-of-turn behavior, on every trial.
It must also exceed baseline median speed by more than 3%. Otherwise calibration chooses `off` and restores
the earlier speculation flags. Results, rejected trials and engine facts are saved under `dloop_calibration`
in the model config. An interrupted or failed calibration leaves its previous settings in place.

This is a small parity and performance check, not a general language-model benchmark. The underlying mixed
CPU/GPU engine can produce different floating-point results with different verification window sizes; keeping
the acceptance rule does not establish bit-identical output across those window sizes. Do not assume that an
uncalibrated loop preserves the exact legacy continuation.

## Measurements

Measurements and commands are in [bench/results/2026-10-10-dloop](../bench/results/2026-10-10-dloop/README.md).
They include matched greedy and sampled checks with the same six-draft window, plus a single-block
DFlash adapter check in the separate validation checkout. They use the complete 48-layer Qwen3.8-Flash-Next IQ3_XXS model, a Ryzen 9 5900X, 125 GiB RAM and an RTX 4070 Ti
SUPER with 16 GiB VRAM. Stock MTP loops were slower than the installer-style adaptive MTP baseline on the tested
prompts, and two of three prompts changed greedy tokens. **DLoop stays off by default.** Calibration rejected
all tested loop settings when they differed from baseline. Some preliminary measurements overlapped a
separate compilation; use the clean validation runs identified in the results README for timing comparisons.

## DFlash integration after its merge

This branch does not import the separate DFlash implementation. The delta for connecting DLoop to that
implementation is [integrations/dloop-dflash.patch](../integrations/dloop-dflash.patch). It is prepared against a
local validation merge of DLoop and `feat/dflash` at `73b3db6b`. Apply it only after both implementations are
present, resolving any intervening upstream changes:

```sh
git apply --check integrations/dloop-dflash.patch
git apply integrations/dloop-dflash.patch
```

The patch reuses DFlash's normal setup choice, artifact and block graphs. After a confident block, it projects
the drafter's final normalized hidden states directly into its own per-layer context K/V, bypassing the target
feature fusion. The previous block's last draft becomes the next anchor. Committed target taps replace those
unverified features after verification; later rejected cells remain outside the next attention bound. All
accumulated blocks receive one target verification. The patch also connects setup, restores `--dflash-block`
on disable, and uses the same calibration policy. Stock DFlash weights receive no loop-aware retraining.

A trained seven-query DFlash block cannot be repeated in full with the current eight-row verifier.
Two loops therefore use at most three queries per block; using all seven permits one loop only.
