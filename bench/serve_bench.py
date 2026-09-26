#!/usr/bin/env python3
"""bench/serve_bench.py - repeatable engine-level benchmark over the `strata --serve` stdin protocol.

Spawns the engine once with the production arguments (from the run config, with a fixed expert-cache
size so residency does not drift between runs), feeds it GEN requests, timestamps every token line,
and reports per-request prompt/decode throughput plus the verify-window grouping (tokens per window,
window latency percentiles).  Greedy requests are deterministic; the same prompt always produces the
same tokens, so runs are comparable.

    .venv/bin/python bench/serve_bench.py --label baseline --reps 3 --new 256 --prompt-file bench/prompt_1k.txt
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def load_tokenizer(pack_dir: Path):
    import strata_tokenizer as ST
    tdir = pack_dir / "tokenizer"
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def engine_args(cfg: dict, expert_cache: int | None, vision: bool) -> list[str]:
    args = list(cfg["args"])
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--expert-cache" and expert_cache is not None:
            out += ["--expert-cache", str(expert_cache)]
            i += 2
            continue
        if a == "--vision" and not vision:
            i += 1
            continue
        out.append(a)
        i += 1
    if expert_cache is not None and "--expert-cache" not in out:
        out += ["--expert-cache", str(expert_cache)]
    return out


def run_session(exe: str, args: list[str], cwd: str, env_extra: dict, requests: list[dict], log_path: str):
    """requests: [{ids, new}] in order.  Returns one result dict per request."""
    import os
    env = dict(os.environ)
    env.update(env_extra)
    log = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", bufsize=1, env=env)
    ready = False
    for line in proc.stdout:
        if line.startswith("READY"):
            ready = True
            break
    if not ready:
        raise RuntimeError("engine never became READY (see " + log_path + ")")

    results = []
    for req in requests:
        ids = req["ids"]
        proc.stdin.write(f"GEN {req['new']} {','.join(str(int(t)) for t in ids)}\n")
        proc.stdin.flush()
        t_lines = []          # (monotonic, token id)
        done = None
        t0 = time.monotonic()
        stopped = False
        for line in proc.stdout:
            if line.startswith("T "):
                t_lines.append((time.monotonic(), int(line[2:])))
            elif line.startswith("DONE"):
                f = line.split()
                done = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                        "decode_ms": float(f[4]), "finish": f[5], "wall_s": time.monotonic() - t0}
                break
            elif line.startswith("ERR"):
                raise RuntimeError("engine error: " + line.strip())
            if req.get("stop_after") and len(t_lines) >= req["stop_after"]:
                stopped = True
                break
        if done is None and stopped:
            # drain to DONE without consuming further
            for line in proc.stdout:
                if line.startswith("DONE"):
                    f = line.split()
                    done = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                            "decode_ms": float(f[4]), "finish": f[5], "wall_s": time.monotonic() - t0}
                    break
                elif line.startswith("ERR"):
                    raise RuntimeError("engine error: " + line.strip())
        if done is None:
            raise RuntimeError("engine ended without DONE")
        # window grouping: consecutive T lines <3 ms apart belong to one verify window
        windows = []
        for ts, _tid in t_lines:
            if windows and ts - windows[-1][-1] < 0.003:
                windows[-1].append(ts)
            else:
                windows.append([ts])
        wlat = [w[-1] - w[0] for w in windows]                    # span of each window (ms / 1000)
        wtok = [len(w) for w in windows]
        gaps = [windows[i][0] - windows[i - 1][-1] for i in range(1, len(windows))]
        done["windows"] = len(windows)
        done["tokens_per_window"] = round(sum(wtok) / len(wtok), 3) if wtok else 0.0
        done["window_gap_ms_p50"] = round(1000 * statistics.median(gaps), 2) if gaps else None
        done["window_gap_ms_p95"] = round(1000 * sorted(gaps)[int(0.95 * (len(gaps) - 1))], 2) if gaps else None
        done["window_gap_ms_p99"] = round(1000 * sorted(gaps)[int(0.99 * (len(gaps) - 1))], 2) if gaps else None
        done["decode_tok_s"] = round(1000.0 * done["generated"] / done["decode_ms"], 2) if done["decode_ms"] else None
        done["prompt_tok_s"] = round(1000.0 * done["prompt_tokens"] / done["prompt_ms"], 2) if done["prompt_ms"] else None
        done["first_tokens"] = [tid for _ts, tid in t_lines[:8]]
        results.append(done)
        print("  request: %d prompt tokens in %.0f ms (%.1f tok/s), %d generated in %.0f ms (%.2f tok/s), "
              "%.2f tok/window" % (done["prompt_tokens"], done["prompt_ms"], done["prompt_tok_s"],
                                   done["generated"], done["decode_ms"], done["decode_tok_s"],
                                   done["tokens_per_window"]), flush=True)
    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default=str(ROOT / "strata-iq3_xxs.json"))
    ap.add_argument("--pack", default=str(ROOT / "packs/iq3_xxs"))
    ap.add_argument("--prompt-file", required=True, help="text file, one prompt per run (separated by a line '===')")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3, help="times each prompt is sent")
    ap.add_argument("--expert-cache", type=int, default=None, help="pin the VRAM tier size (slots)")
    ap.add_argument("--vision", action="store_true", help="keep the engine's --vision flag")
    ap.add_argument("--engine-arg", action="append", default=[], help="extra engine argument (repeatable)")
    ap.add_argument("--drop-engine-arg", action="append", default=[], help="remove an engine argument")
    ap.add_argument("--out", default=str(ROOT / "bench/results/serve_bench.jsonl"))
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = load_tokenizer(Path(a.pack))
    sections = [s.strip() for s in Path(a.prompt_file).read_text(encoding="utf-8").split("\n===")]
    prompts = [tok.encode(s, parse_special=True) for s in sections if s]
    print("label %s: %d prompt(s), lengths %s, new=%d, reps=%d" %
          (a.label, len(prompts), [len(p) for p in prompts], a.new, a.reps), flush=True)

    args = engine_args(cfg, a.expert_cache, a.vision)
    for d in a.drop_engine_arg:
        while d in args:
            args.remove(d)
    args += []
    for ea in a.engine_arg:
        # the engine's parser takes "--opt value" as two tokens; accept the "--opt=value" spelling anyway
        args.extend(ea.split("=", 1) if ea.startswith("--") and "=" in ea else [ea])

    requests = []
    for p in prompts:
        for _ in range(a.reps):
            requests.append({"ids": p, "new": a.new})

    log_path = str(ROOT / f"bench/results/engine-{a.label}.log")
    t0 = time.time()
    results = run_session(cfg["exe"], args, cfg.get("cwd", str(ROOT)), {}, requests, log_path)
    for r in results:
        r["label"] = a.label
    with open(a.out, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    dec = [r["decode_tok_s"] for r in results if r["generated"] > 1]
    pre = [r["prompt_tok_s"] for r in results if r["prompt_tokens"] > 1]
    print("label %s done in %.0f s: decode tok/s %s (median %.2f), prompt tok/s %s (median %.2f)" %
          (a.label, time.time() - t0, dec, statistics.median(dec) if dec else 0,
           pre, statistics.median(pre) if pre else 0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
