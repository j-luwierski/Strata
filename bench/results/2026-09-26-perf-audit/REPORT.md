# Strata performance audit — 2026-09-26

Systematic audit of the local Strata checkout: hot-path analysis, one-optimization-at-a-time
experiments, and a fresh baseline-vs-final benchmark. Every number below was measured on this
machine; where a claim rests on the repository's own paper (`docs/paper/Strata-Paper.pdf`) it is
marked as such.

## 1. Repository overview

Strata is a single-model inference engine for Qwen3.8-Flash-Next (48 layers: 36 GDN + 12 QSA mixers,
512 experts/layer with top-10 routing, an n-gram PLE table, and an MTP draft layer). The production
path is `engine/strata --serve`: prompts are processed in 2,048-token chunks (prefill), then decode
runs as speculative **verify windows** of up to 4 tokens (`Verifier::run`, a captured CUDA graph per
window size). Per layer the router's top-10 ids reach the host through a pinned doorbell; the host
splits them into VRAM-resident experts (GPU hit kernel), a PCIe DMA share, and CPU-computed misses
(23→11-worker spin pool over the 40 GiB pinned expert arena). The two custom features:

- **history cache** — serve-mode reuse of the consumed token sequence (`generate.cpp` GEN loop),
- **temperature support** — per-request sampling keys on the GEN line feeding `sampler.cu`.

## 2. Baseline (original tree, commit f2a08d4)

Machine: RTX 4070 16 GB (CUDA sm_89 build, driver 610.57.04), AMD Ryzen 9 5900X (12C/24T, **AVX2
only, no AVX-512**), 125 GB RAM, Linux. Build: `./build.sh` (Release, `-O3`, CUDA arch 89).
Workload: `bench/serve_bench.py` over the engine's GEN protocol; code-review prompt of 762 tokens,
256 generated tokens, greedy, `--expert-cache 2300` (pinned so residency cannot drift between
runs). Prefill workload: 4,371-token prompt, 8 generated. Sampled workload: `temperature=0.7
top_p=0.95 top_k=20 seed=3`.

| Metric | Baseline (measured) |
|---|---:|
| Decode (greedy, 256 tok) | **54.33 tok/s** median of 3 (47.2 / 57.0 / 54.3) |
| Decode (sampled, T=0.7) | **1.50 tok/s** |
| Tokens per verify window | 2.46 |
| Prompt processing (4,371 tok) | **678.7 tok/s** median of 3 |
| Prompt processing (762 tok) | ~470 tok/s |

## 3. Hot-path analysis (measured, final build, fresh session)

Per verify round (up to 4 tokens, 2.72 tokens/round realized; the engine's own counters, non-serve
`--stats` run):

| Stage | ms/round | share |
|---|---:|---:|
| CPU expert pool (`pool multi`) | 25.0 | 51% |
| — gate/up rows | 16.0 | |
| — down rows | 7.1 | (27.5 GB/s of expert weights streamed) |
| GPU rings (wait for layer graphs) | 15.2 | 31% |
| Host loop (plan/actq/jobs + staging) | 6.9 | 14% |
| Commit graph + sync | 0.63 | 1% |
| MTP drafting | 1.49 | 3% |

Findings that drove the work:

1. **The sampled path decoded at 1.5 tok/s**: `sampler_kernel` ran one CUDA thread per token doing
   `top_k` as 20 sequential full-vocabulary scans with an inner sweep over the taken list
   (~1.6 s per verify window). The greedy path used a proper block-parallel reduction; the sampled
   path never got the same treatment. And because the HTTP server never forwarded sampling keys,
   every served request was greedy, so nobody could see it.
2. **The Linux pool enumerated logical CPUs**, so this box ran 23 workers + host over 12 physical
   cores (the Windows branch groups by physical core; the paper's 6-core machine ran 6 workers).
3. Prefill is bound by per-expert dequantize-to-f16 traffic and per-expert kernel launches (paper
   finding 11 confirms the same); no algorithmic waste was found in it that a small change fixes.

## 4. Optimization experiments

### #1 Block-parallel top-k sampler — KEEP (commit 1eb83d0)

- `src/kernels/cuda/sampler.cu`: one block per token; `top_k` as `k` block-argmax rounds (two-level
  warp/block reduction, ties to the lowest index — the serial scan's strict `>` rule), then the same
  top_p / temperature / Philox chain in the serial order.
- Correctness: `sampler_parity` 0 failures; a 248,320-vocabulary GPU-vs-CPU draw check matches the
  serial reference exactly on five parameter sets including injected duplicate logits.
- Measured: sampled decode 1.50 → **42.7 tok/s** (128-token probe); greedy unchanged (53.3 vs 54.3,
  within noise — greedy launches the untouched kernel).

### #2 Forward sampling keys from the HTTP server — KEEP (commit 0c9dfbd)

- `serve/server.py` (`StrataEngine.generate`): the request's `temperature/top_p/top_k/seed` ride the
  GEN line in the engine's own spelling. Absent temperature stays greedy; `temperature=0` is not
  forwarded; `top_k` outside the sampled path's 1..64 is dropped.
- End-to-end over the real server: temperature=0 == absent (token-identical); temperature=0.8
  differs from greedy; two temperature=0.8 seed=99 requests return identical text.
- This completes the documented temperature feature; it is only usable because #1 made the sampled
  path fast.

### #3 One pool worker per physical core on Linux — KEEP (commit e072373)

- `src/kernels/cpu/pool.cpp`: the Linux branch of `physical_cores()` keeps the first sibling of each
  sysfs `thread_siblings_list` group (fallback to the old enumeration when sysfs is missing).
- Measured (greedy 256, 5 reps each): 23 logical workers **53.43 tok/s** median vs one-per-core
  **58.43** (+9.4%); an independent 3-rep A/B agreed (+8.9%). Prompt processing unchanged (471).

### #4 DMA share (pcie-frac) retuned for this box — KEEP (local run config, gitignored)

- With the faster CPU pool, the 55% DMA share (tuned for the paper's 6-core machine) was stealing
  RAM bandwidth from twelve AVX2 workers. Measured: 0.40 → **62.22 tok/s** median (5 reps) vs 0.55 →
  56.96 and 0.70 → 52.89. The compiled-in default stays 0.55; `strata-iq3_xxs.json` now passes
  `--pcie-frac 0.40` (the file is machine-specific by design).

### #5 Spec-window sweep — REJECTED

- `--spec 5` → 59.3 tok/s; `--spec 6 --spec-min-p 0.4` → 54.1 vs 62.2 for the existing spec 4 /
  min-p 0.5. Every extra window token routes its own CPU experts (paper finding 2); no change made.

## 5. Successful optimizations (retained)

| Commit | Optimization | Decode (greedy) | Decode (sampled) | Prompt | Status |
|---|---|---:|---:|---:|---|
| 1eb83d0 | Block-parallel top-k sampler | n/a (greedy untouched) | 1.5 → 42.7 tok/s (probe) | — | KEEP |
| 0c9dfbd | Server forwards sampling keys | — (feature completion) | enables #1 over HTTP | — | KEEP |
| e072373 | One worker per physical core | +9.4% | — | ±0% | KEEP |
| (local) | `--pcie-frac 0.40` | +9.2% | — | ±0% | KEEP |

## 6. Rejected / no change

| Idea | Result | Reason |
|---|---|---|
| `--spec 5`, `--spec 6 --spec-min-p 0.4` | 59.3 / 54.1 vs 62.2 tok/s | Slower; extra window tokens cost more CPU expert work than they save |
| `--pcie-frac 0.70` | 52.9 vs 57.0 tok/s | Regression at the current CPU speed |
| Prefill micro-changes (host grouping loops, per-layer allocations) | ns-scale | Not measurable against a 1.5 ms/chunk kernel-traffic bound |
| T-MAC-style LUT i-quant CPU kernels | not attempted | New kernel family; highest-value *remaining* work, needs its own parity campaign |

## 7. Custom features after the audit

- **History cache**: verified on a real three-turn conversation — turn 2 reused **826 tokens and
  processed the 14-token delta** (prompt 2,932 ms → **357 ms**). Note: a turn's continuation is not
  bit-identical to a from-scratch re-prefill of the same conversation; the reused seam carries
  verify-window (sequential) arithmetic while a fresh prefill uses chunked-scan kernels — the same
  batch-vs-sequential rounding difference every batched-prefill engine has, not a cache defect.
  Reuse still requires the client to echo the assistant turn the engine consumed (the bundled chat
  page strips reasoning, so its continuing chats re-prefill).
- **Temperature**: now honored end to end (it was silently dropped at the HTTP layer), and fast.

## 8. Final cumulative benchmark (fresh runs, same workloads)

| Metric | Original (f2a08d4) | Final (this audit) | Difference |
|---|---:|---:|---:|
| Decode greedy, 256 tok | 54.33 tok/s | **62.90 tok/s** (5 reps: 54.5–63.7) | **+15.8%** |
| Decode sampled (T=0.7), 256 tok | 1.50 tok/s | **57.47 tok/s** (5 reps) | **38×** |
| Prompt processing, 4,371 tok | 678.7 tok/s | 681.1 tok/s | +0.4% (unchanged) |
| Tokens per verify window | 2.46 | 2.42–2.91 | ~unchanged |

Honest caveat: the pool/DMA changes alter the mix of experts computed on the CPU vs on the GPU. The
engine's own R4 note says the GPU hit path is not bit-exact against the CPU kernels, so greedy
continuations can differ beyond the early tokens when that mix changes (demonstrated: a turn
continued from a reused session diverged from a from-scratch session at token 18 with the adaptive
tier). Timing claims above are unaffected; output-exactness claims are scoped accordingly.

## 9. Git history

```
Original SHA: f2a08d4 (main)
Final SHA:    594714e, plus the untracked pre-existing TODO-GLM untouched

1eb83d0 perf: sample the verify window's head with a block per token
0c9dfbd serve: forward temperature/top_p/top_k/seed to the engine
e072373 pool: one worker per physical core on Linux, like the Windows path
94b0b93 bench: engine-level benchmark harness over the GEN protocol
594714e docs: the v1 limits line still said temperature was ignored
```

## 10. Remaining opportunities (ranked)

1. **Lookup-table i-quant CPU kernels (T-MAC style)** — the CPU pool is still 51% of a decode round
   at 27.5 GB/s on AVX2; the i-quant dot products are arithmetic-bound (~5 GB/s/core, paper finding
   7). A LUT kernel or load-time conversion of the hottest experts could lift the CPU half toward
   the Q2_0 speed. Highest remaining ceiling, needs a full parity campaign.
2. **Fused 8-bit grouped prefill kernel** (paper §7) — prefill is bound by per-expert f16 dequant
   traffic; a Marlin-style grouped kernel would lift the 680 tok/s prompt path, most on short/medium
   prompts.
3. **Auto-tune the DMA share at load** — a 20-round calibration probe (0.3/0.55/0.7) would pick the
   pcie-frac per machine instead of per-config tuning; this audit measured ±9% between settings.
4. **Make the GPU hit path bit-exact (R4)** — would make residency changes output-neutral, which
   makes the adaptive tier reproducible and the history-cache seam exact against fresh prefill.
5. **Next-layer expert prediction** to start DMA before the router runs (paper §7) — architectural.

## 11. Final assessment

- The primary decode bottleneck on this box was the **CPU expert pool**, and its biggest
  self-inflicted cost was thread placement (two workers per core); the pool now matches the
  Windows/reference policy and streams experts at 27.5 GB/s.
- The second finding was not on the paper's list: the **temperature feature was dead end to end** —
  dropped at the HTTP server and unusably slow (1.5 tok/s) in the kernel. Both halves are fixed and
  verified; sampled and greedy decode now sit within ~9% of each other.
- Prompt processing was confirmed bound by kernel-traffic architecture, not by anything a small
  change fixes; it is unchanged, as expected.
- Next highest-value target: the i-quant CPU kernels (item 1 above).
