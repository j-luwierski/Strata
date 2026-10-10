# STEPQuant per-layer allocation, 2026-10-10

This changes storage layout, not quantization or calibration. The full model,
plans, held-out corpus and hardware are the same as in the
[quality report](../2026-10-10-stepquant-rtx4070ti-super/README.md).

Each GDN layer now reserves its own payload plus a 16-byte plan header, rounded
to 256 bytes. Prefix sums locate layers in full and split sessions. Only the
session's shared speculative temporary buffer retains the largest layer size.
Convolution, QSA and model weights keep their existing formats.

## Allocation

These counts come from the actual calibrated plans and the allocator. They are
not measurements of total process VRAM. The 108 MiB baseline is the persistent
FP32 recurrent matrices. Shared FP32 computation scratch exists in both paths.
The STEPQuant columns include slot headers/alignment, the temporary slot and
immutable device metadata (shared between sessions on that GPU). Unchanged
convolution and other session buffers are excluded from this comparison.

| Plan | Previous STEPQuant allocation | Compact allocation | Additional saving | Saving versus FP32 |
| --- | ---: | ---: | ---: | ---: |
| @6 | 32.802 MiB | 23.320 MiB | 9.482 MiB | 84.680 MiB |
| @4 | 29.587 MiB | 17.016 MiB | 12.570 MiB | 90.984 MiB |

The CUDA allocation test uses the full model geometry, 16 QSA cells and k=10;
it allocates the arena and device plan metadata, zeros/reads all recurrent
layers and saves/restores a checkpoint with each real plan. With @6 the arena
falls from the FP32 127,387,392 bytes to 37,681,664; with @4 to 31,071,744.
These arena counts exclude separately allocated immutable device metadata.
The complete test output is in `six-allocation.txt` and `four-allocation.txt`.

## Quality and verification

Full IQ3_XXS model teacher-forced evaluation was repeated for both plans using
the same fixed expert placement as the earlier report. Both FP32 and STEPQuant
logprob TSV files are byte-identical to their corresponding earlier files, for
all 1,072 positions. Thus scored perplexity remains 5.142867 (FP32), 5.130533
(@6), and 5.200818 (@4), on 880 positions after warmup. The earlier report
contains the identical raw TSVs; their hashes and this binary's hash are in
`summary.json`, and the new run metadata is in the comparison JSON files.
This verifies layout parity on that small corpus, not general model accuracy
or production decode speed.

CUDA build and tests pass: heterogeneous-layer sizes, compact offsets,
convolution offsets, full and split checkpoints, rejection of the old padded
layout, independent sessions, zeroing without damaging a neighbouring layer,
proposal isolation, every accepted speculative prefix and captured commits.
The standalone conversation snapshot and validation tests also pass. The 13
OFF-path source checks remain byte-identical to origin/main.

HIP and SYCL sources have matching layout changes. HIP configuration cannot
find ROCm, and SYCL configuration with the required `icpx` compiler cannot
find that compiler. Neither backend was built or device-tested here.

Previously saved heterogeneous STEPQuant checkpoints with the padded layout
are rejected by the new byte-count/header validation; recalibrating the plan
is unnecessary. The STEPQuant-disabled checkpoint path is unchanged.

## Repeat

```sh
cmake --build build-stepquant --target strata stepquant_test -j6
build-stepquant/stepquant_test --allocation-plan /path/to/calibrated.plan
python tools/test_stepquant_default.py
```

Repeat quality with `tools/stepquant_quality.py` and compare the TSV hashes as
shown in the earlier report. The installer accepts the same existing plans.
