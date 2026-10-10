#!/usr/bin/env python3
"""Paired full-model teacher-forced quality measurement, using native verifier windows.

No task accuracy is inferred from perplexity. Uses the same executable, weights,
cache capacity and prompt tokens in both runs. STRATA_LOGPOS records every actual
next-token probability; --short-read routes the text through decode recurrence,
so STEPQuant is applied after every token rather than only after prompt chunks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.stepquant_setup import drop, engine_env, validate_plan
from tools.strata_tokenizer import Tokenizer


def read_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        values = line.split('\t')
        if len(values) < 8:
            raise ValueError('truncated logprob record')
        pos, token, logp, top, top_logp, correct, extra, without = values[:8]
        if not math.isfinite(float(logp)):
            raise ValueError('nonfinite target probability')
        rows.append(dict(position=int(pos), token=int(token), logp=float(logp), top=int(top),
                         correct=int(correct), without_eot=float(without)))
    return rows


def evaluate(cfg, items, out, label, plan, warmup, cache):
    args = list(cfg['args'])
    for flag in ('--stepquant-plan', '--spec', '--prefill', '--max-context', '--max-new', '--batch', '--short-read', '--prompt-cache', '--expert-cache', '--pcie-frac'):
        drop(args, flag)
    args += ['--serve', '--spec', '2', '--prefill', '64', '--max-context', '2048',
             '--short-read', '2048', '--prompt-cache', '0', '--expert-cache', str(cache), '--pcie-frac', '0']
    if plan:
        validate_plan(Path(plan)); args += ['--stepquant-plan', str(Path(plan).resolve())]
    logpos = out / (label + '.logprobs.tsv')
    logpos.unlink(missing_ok=True)
    env = engine_env(cfg); env['STRATA_LOGPOS'] = str(logpos)
    command = [cfg['exe'], *args]
    messages = queue.Queue()
    start = time.monotonic()
    all_rows, lengths = [], []
    with (out / (label + '.engine.log')).open('w') as stderr:
        proc = subprocess.Popen(command, cwd=cfg.get('cwd'), env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        def read():
            for line in proc.stdout:
                messages.put(line.strip())
            messages.put(None)
        threading.Thread(target=read, daemon=True).start()
        def until(prefix):
            while True:
                line = messages.get(timeout=900)
                if line is None or line.startswith('ERR'):
                    raise RuntimeError(f'{label}: engine ended or failed: {line}; see engine log')
                if line.startswith(prefix):
                    return line
        try:
            until('READY')
            for item in items:
                tokens = item['tokens']
                proc.stdin.write('GEN 1 temp=0 top_k=1 ckpt=0 ' + ','.join(map(str, tokens)) + '\n')
                proc.stdin.flush()
                until('DONE')
                rows = read_rows(logpos)
                fresh = rows[len(all_rows):]
                expected = len(tokens) - 1
                if len(fresh) != expected or [r['position'] for r in fresh] != list(range(expected)) or \
                        [r['token'] for r in fresh] != tokens[1:]:
                    raise ValueError(f'{label}/{item["id"]}: incomplete teacher forcing: {len(fresh)} rows, expected {expected}')
                all_rows.extend(fresh)
                lengths.append(expected)
                print(f'{label}/{item["id"]}: {expected} teacher-forced tokens', flush=True)
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    scored, at = [], 0
    for n in lengths:
        scored.extend(all_rows[at + warmup:at + n]); at += n
    if not scored:
        raise ValueError('no tokens remain after warmup')
    nll = -sum(r['logp'] for r in scored) / len(scored)
    without = [r['without_eot'] for r in scored if math.isfinite(r['without_eot'])]
    report = dict(label=label, command=command, seconds=time.monotonic()-start, tokens=len(all_rows),
                  scored_tokens=len(scored), warmup_per_document=warmup, mean_nll=nll, perplexity=math.exp(nll),
                  next_token_accuracy=sum(r['correct'] for r in scored)/len(scored),
                  perplexity_without_eot=math.exp(-sum(without)/len(without)) if without else None,
                  plan_sha256=hashlib.sha256(Path(plan).read_bytes()).hexdigest() if plan else None)
    (out / (label + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    return report, scored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--corpus', type=Path, default=ROOT / 'bench/stepquant/heldout.json')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expert-cache', type=int, default=4000, help='fixed expert cache slots in both runs (default 4000; reduce for a smaller GPU)')
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=512)
    a = p.parse_args()
    if a.expert_cache < 1:
        p.error('expert-cache must be positive')
    if not 9 <= a.max_tokens <= 2048 or not 0 <= a.warmup < a.max_tokens - 1:
        p.error('max-tokens must be 9..2048 and warmup smaller than max-tokens - 1')
    cfg = json.loads(a.config.read_text())
    native = cfg['args'][cfg['args'].index('--native') + 1]
    tk = Tokenizer.from_gguf(native)
    items = json.loads(a.corpus.read_text())
    for item in items:
        item['tokens'] = tk.encode(item['text'], parse_special=True)[:a.max_tokens]
        if len(item['tokens']) <= a.warmup + 1:
            raise ValueError('a document is shorter than warmup')
    a.output.mkdir(parents=True, exist_ok=True)
    baseline, b = evaluate(cfg, items, a.output, 'fp32', None, a.warmup, a.expert_cache)
    quant, q = evaluate(cfg, items, a.output, 'stepquant', a.plan, a.warmup, a.expert_cache)
    if [(r['position'], r['token']) for r in b] != [(r['position'], r['token']) for r in q]:
        raise ValueError('runs did not score identical positions')
    report = dict(baseline=baseline, stepquant=quant,
                  delta_nll=quant['mean_nll']-baseline['mean_nll'],
                  perplexity_ratio=quant['perplexity']/baseline['perplexity'],
                  top1_agreement=sum(x['top']==y['top'] for x,y in zip(b,q))/len(b),
                  corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
                  native=str(Path(native).resolve()), executable_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
    (a.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
