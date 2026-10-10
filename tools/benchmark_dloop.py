"""Same-day native-model DLoop A/B. No downloads, no installed config edits.

python tools/benchmark_dloop.py --baseline /path/to/main/strata --engine /path/to/dloop/strata \
  --pack /path/to/pack --native /path/to/model.gguf --mtp /path/to/mtp/rt --out /tmp/dloop-report
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer

PROMPTS = {
    "english": "Explain why matrix multiplication processes a long prompt faster than generating its tokens one at a time.",
    "code": "Write a Python function that merges two sorted lists without changing the inputs. Explain its time complexity.",
    "polish": "Wyjaśnij po polsku, jak działa pamięć podręczna procesora. Podaj prosty przykład z programu.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("baseline", "engine", "pack", "native", "mtp", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--profile", type=Path, default=Path("data/expert-profile.bin"))
    p.add_argument("--cache", type=int, default=4000)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument("--max-loops", type=int, default=2)
    p.add_argument("--gate", type=float, default=-0.5)
    p.add_argument("--prompt", choices=tuple(PROMPTS), action="append")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_gguf(a.native)
    report = {"date": time.strftime("%Y-%m-%d"), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "platform": platform.uname()._asdict(), "runs": []}
    for command, key in [(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], "gpu"),
                         (["lscpu"], "cpu")]:
        report[key] = subprocess.run(command, capture_output=True, text=True).stdout
    common = ["--pack", str(a.pack.resolve()), "--native", str(a.native.resolve()), "--mtp", str(a.mtp.resolve()),
              "--expert-profile", str(a.profile.resolve()), "--expert-cache", str(a.cache), "--max-context", "4096",
              "--max-new", str(a.tokens), "--prefill", "64", "--ple-io", "ram", "--pcie-frac", "0",
              "--adapt-every", "0", "--pool-workers", "12", "--suffix-draft", "0", "--spec", "4", "--spec-min-p", "0.5"]
    reference = {}
    for repeat in range(a.repeats):
        for name in a.prompt or PROMPTS:
            prompt = f"<|im_start|>user\n{PROMPTS[name]}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ids = tok.encode(prompt, parse_special=True)
            arms = [("main", a.baseline, []), ("off", a.engine, []),
                    ("dloop", a.engine, ["--dloop", "--dloop-block-size", str(a.block_size), "--dloop-max-loops", str(a.max_loops),
                                         "--dloop-gate", str(a.gate)])]
            if repeat % 2:
                arms.reverse()
            for arm, engine, flags in arms:
                stem = f"{name}-{repeat}-{arm}"
                command = [str(engine.resolve()), *common, "--tokens", ",".join(map(str, ids)),
                           "--window-hashes", str((a.out / (stem + '.windows')).resolve()), *flags]
                print(f"{stem}: running full native model", flush=True)
                start = time.monotonic()
                run = subprocess.run(command, capture_output=True, text=True, timeout=1800)
                (a.out / (stem + ".log")).write_text(run.stdout + '\n' + run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{stem} failed ({run.returncode}); see its log")
                outputs = list(map(int, re.search(r"^output  :(.*)$", run.stdout, re.M).group(1).split()))
                speed = float(re.search(r"^decode\s+.*?->\s+([\d.]+) tok/s", run.stdout, re.M).group(1))
                reference.setdefault(name, outputs)
                result = {"name": name, "repeat": repeat, "arm": arm, "command": command, "ids": outputs,
                          "same_tokens": outputs == reference[name], "tok_s": speed, "wall_s": time.monotonic() - start,
                          "output_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest()}
                report["runs"].append(result)
                (a.out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(f"{stem}: {speed:.2f} tok/s; same_tokens={result['same_tokens']}", flush=True)
    return 0 if all(r["same_tokens"] for r in report["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
