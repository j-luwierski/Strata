# DLoop / DFlash integration delta

`dloop-dflash.patch` changes only the DLoop adapter hooks in the DFlash header/runtime, the shared decoder, DLoop's setup helper and its config tests. It does not add the DFlash implementation to the DLoop branch.

Prerequisites: DLoop and the DFlash implementation from `feat/dflash` (validated base: `73b3db6b`) must already
be merged. The validation checkout used a local merge, outside this branch. The patch was checked against its
clean index with `git apply --cached --check`. See [docs/DLOOP.md](../docs/DLOOP.md) for behavior and commands.

CUDA compilation and full-model runs passed for both one-block and two-block paths. The one-block path
matched all 128 baseline token IDs. A forced second block was slower and changed the continuation relative
to the shorter window, so calibration must check it before enabling it.
