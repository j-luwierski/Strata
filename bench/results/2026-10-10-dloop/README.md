# DLoop validation — 2026-10-10

Complete Qwen3.8-Flash-Next IQ3_XXS, all 48 layers and stock MTP; Ryzen 9 5900X (12 cores), about 125 GiB RAM,
RTX 4070 Ti SUPER 16 GiB, CUDA 13.4, driver 610.57.04, Linux. Each JSON records the commands, GPU/CPU facts,
output token IDs, hashes and timings. Baseline is a same-day build of main `fb58e0db` with DLoop compiled OFF.
The feature build is main plus the guarded DLoop changes. No installed model config was changed for these runs.

Common controls: fixed `--expert-cache 4000` byte budget (the native profile produces 5357 actual slots,
8.68 GiB), expert profile `data/expert-profile.bin`, `--pcie-frac 0`, 12 pool workers, adaptive swaps off,
prompt lookup off, RAM PLE, 4096-token context, 64-token prefill chunks and `STRATA_PREFILL_CPU_SHARE=0`.
These settings isolate the loop; they are not the best-performing settings on every PC.

## Clean comparisons

The following runs had no compilation running alongside them. Decode includes graph capture costs; the CLI
runs below are short continuations, not a steady-state serving throughput benchmark.

| Run | Baseline | DLoop | Greedy token parity |
| --- | --- | --- | --- |
| MTP, English, 128 tokens, two paired repeats; adaptive baseline (`spec=4`, min-p=.5), DLoop block=3, loops=2, gate=-.5 | 72.86 tok/s (main); 72.61 (feature OFF) | 64.01 tok/s | Feature OFF matches main in both runs; loop changes token IDs |
| MTP, English, 128 tokens, same six-draft window (`spec=7`, min-p=0); DLoop two blocks of three, gate=-100 forces extension | 58.74 tok/s (main); 58.83 (feature OFF) | 58.56 tok/s | All 128 tokens identical, including both loop blocks |
| DFlash Q4_0 integration patch, English, 128 tokens; fixed three-query block, DLoop limited to one block | 41.72 tok/s; repeated OFF 41.84 | 41.97 tok/s | All 128 tokens identical |
| DFlash Q4_0 integration patch, English, 128 tokens; baseline three-query block, DLoop two blocks of three, gate=-100 forces extension | 41.12 tok/s; repeated OFF 41.92 | 32.46 tok/s | Loop changes token IDs |

The clean MTP default comparison is about **12% slower** with loops. The identical longer-window comparison
shows that grouping the same six MTP drafts into two gated blocks preserves that existing path's output. The
legacy six-draft path itself changes the English continuation relative to the adaptive baseline, first at
output index 110; the gated variable-window run first differs at index 8. These observations establish window
sensitivity in this mixed CPU/GPU engine, not the exact cause of every changed token. No bit-exactness claim
against all legacy window policies is made.

The DFlash patch was compiled and run in a separate checkout containing a local merge of DLoop and DFlash at
`73b3db6b`. DFlash's implementation is not included in this branch. A single seven-query trained DFlash block
cannot be repeated in full within the native verifier's limit of seven drafts. These tests use three-query
blocks. The second block consumes the first block's final normalized draft features, without a target pass.
All accumulated drafts still receive target verification.

Raw data:

- [mtp-clean-final.json](mtp-clean-final.json): clean paired default comparison.
- [mtp-long-window-parity.json](mtp-long-window-parity.json): existing six-draft MTP vs two three-draft stages.
- [dflash-single-block.json](dflash-single-block.json): one-stage adapter parity.
- [dflash-multi-block.json](dflash-multi-block.json): forced second stage, with seven-row verification.

## Calibration and additional checks

[calibration.json](calibration.json) records a real serial-server calibration with three prompts, two rounds,
and five unique loop choices. Baseline median was 49.55 tok/s; every loop candidate differed in greedy tokens
on at least one prompt and was rejected. The selected result was **off**. A separate compilation overlapped
part of this sweep, so its timing numbers are not used as a clean speed comparison. Token comparisons remain
useful, and the clean comparisons above independently confirm the lack of a speed gain on the tested case.

[mtp-adaptive.json](mtp-adaptive.json) contains the preliminary English/code/Polish test: two runs of 128 tokens
per arm. Feature OFF matched main on all six continuations (768 tokens). The loop matched code and differed
on English/Polish. Compilation also overlapped part of this test; its times are preliminary.

139 CPU/setup regression tests passed, including gate equality, invalid probabilities, latest-block scoring,
verifier bounds, reversible config edits, unsafe-mode rejection, calibration parity filtering, the 3% margin,
engine cleanup and a fixed-budget cache probe. `setup.sh --help` exposes all DLoop options; malformed settings
are rejected before GPU detection or model downloads.

DLoop compiled OFF removes the added engine/header source blocks exactly back to main for CUDA/HIP and SYCL
sources. CUDA compiled ON and OFF and ran the full model. HIP configuration failed because ROCm was absent;
SYCL configuration failed because `icpx` was absent. Neither backend's execution was tested. A 64-token sampled check at temperature 0.7, seed 123 and the same six-draft window also matched all token IDs
(main 56.29, feature OFF 56.54, DLoop 56.53 tok/s); see [mtp-sampling-parity.json](mtp-sampling-parity.json).
Other sampling settings, long contexts, multiple GPUs, end-to-end HTTP clients and broad quality metrics were
not measured here.
