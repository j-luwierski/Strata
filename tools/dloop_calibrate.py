"""Measure DLoop against the same model drafter without loops.

Candidates must reproduce every greedy token on the calibration prompts and beat
baseline by at least 3%. This small check is not a general model-quality evaluation.
"""
import copy
import json
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools import dloop_setup as D
import calibrate as CAL


CANDIDATES = [
    {'enabled': True, 'block_size': b, 'max_loops': 7 // b, 'gate': g}
    for b, g in [(3, -0.25), (3, -0.5), (3, -1), (2, -0.5), (2, -1)]
]


def key(choice):
    return 'off' if not choice['enabled'] else f"b{choice['block_size']}-n{choice['max_loops']}-g{choice['gate']:g}"


def select(runs, choices, min_gain=0.03):
    """Every repeat must match baseline and finish, including the end-of-turn token."""
    rates = {name: statistics.median(row['tok_s'] for row in rows) for name, rows in runs.items()
             if rows and all(row['same_tokens'] and row['tok_s'] > 0 for row in rows)}
    if 'off' not in rates:
        raise RuntimeError('baseline did not produce stable greedy tokens; calibration cannot compare candidates')
    best = max(rates, key=rates.get)
    if rates[best] <= rates['off'] * (1 + min_gain):
        best = 'off'
    return choices[best], rates


def measure(cfg, ids_list, start_engine, say=print, repeats=2, max_new=96, candidates=None):
    if not (cfg.get('dloop') or {}).get('enabled'):
        raise ValueError('enable DLoop before calibrating it')
    baseline = D.apply(copy.deepcopy(cfg), {'enabled': False})
    choices = {'off': {'enabled': False}}
    for choice in [cfg['dloop'], *(CANDIDATES if candidates is None else candidates)]:
        clean = {k: choice[k] for k in ('enabled', 'block_size', 'max_loops', 'gate')}
        D.validate(clean)
        choices[key(clean)] = clean
    runs = {name: [] for name in choices}
    reference = {}
    placements = {}
    # Alternating engine-start order reduces drift. Prefix sharing, adaptive tier
    # swaps and automatic expert-cache sizing must not change the comparison.
    base_args = CAL.engine_args(baseline)
    for flag, value in [('--prompt-cache', '0'), ('--conversation-cache-mib', '0'), ('--adapt-every', '0')]:
        base_args = CAL.with_arg(base_args, flag, value)
    if CAL.arg_value(base_args, '--expert-cache') in (None, 'auto'):
        # Probe with the largest candidate window: a fixed byte budget then leaves
        # room for every arm. The engine knows the largest native expert blob.
        probe_cfg = copy.deepcopy(cfg)
        D.apply(probe_cfg, {'enabled': True, 'block_size': 1, 'max_loops': 7, 'gate': -0.5})
        probe = start_engine(CAL.engine_args(probe_cfg))
        try:
            budget = int(probe.info.get('dloop_cache_budget', 0))
            if budget < 1:
                raise ValueError('engine does not report dloop_cache_budget; rebuild the DLoop engine')
        finally:
            probe.close()
        base_args = CAL.with_arg(base_args, '--expert-cache', str(budget))
        say(f'  Fixed expert-cache byte budget for every arm: --expert-cache {budget}')
    for repeat in range(repeats):
        order = list(choices) if repeat % 2 == 0 else list(choices)[::-1]
        for name in order:
            arm = copy.deepcopy(baseline)
            arm['args'] = list(base_args)
            if name != 'off':
                D.apply(arm, choices[name])
            engine = None
            try:
                engine = start_engine(arm['args'])
                placements[name] = dict(getattr(engine, 'info', {}))
                # Identical warmup for graph capture and page faults; it is not scored.
                list(engine.generate(ids_list[0], 16, {'temperature': 0}, threading.Event()))
                per_prompt, matches = [], []
                for i, ids in enumerate(ids_list):
                    output = [t for t in engine.generate(ids, max_new, {'temperature': 0}, threading.Event()) if t is not None]
                    ms = (engine.last or {}).get('decode_ms', 0)
                    if not output or ms <= 0:
                        raise RuntimeError('engine returned no measurable decode')
                    if name == 'off' and i not in reference:
                        reference[i] = output
                    matches.append(output == reference.get(i))
                    per_prompt.append(len(output) * 1000 / ms)
                row = {'tok_s': statistics.median(per_prompt), 'same_tokens': all(matches), 'per_prompt_tok_s': per_prompt,
                       'matches': matches, 'repeat': repeat}
                runs[name].append(row)
                say(f"  DLoop {name}: {row['tok_s']:.2f} tok/s; greedy tokens {'match' if row['same_tokens'] else 'DIFFER'}")
            except Exception as e:
                if name == 'off':
                    raise
                runs[name].append({'tok_s': 0, 'same_tokens': False, 'error': str(e), 'repeat': repeat})
                say(f'  DLoop {name}: rejected ({e})')
            finally:
                if engine is not None:
                    engine.close()
    winner, rates = select(runs, choices)
    return {'choice': winner, 'report': {'date': time.strftime('%Y-%m-%d'), 'runs': runs, 'rates': rates,
            'winner': key(winner), 'min_gain': 0.03, 'repeats': repeats, 'max_new': max_new,
            'engine_info': placements, 'cpu': platform.processor(), 'platform': platform.platform(),
            'base_args': base_args, 'env': {'STRATA_PREFILL_CPU_SHARE': '0'},
            'scope': 'small greedy-token parity and speed check; stock MTP/DFlash, no loop-aware training'}}


def run(cfg, say=print, start_engine=None):
    from strata_tokenizer import Tokenizer
    tpath = Path(cfg['tokenizer'])
    vocab = json.loads((tpath / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for token, i in vocab.items():
        tokens[i] = token
    tokenizer = Tokenizer(tokens, (tpath / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                          json.loads((tpath / 'token_type.json').read_text()))
    ids = [CAL.chat_ids(tokenizer, prompt) for prompt in CAL.PROMPTS]
    if start_engine is None:
        from serve.server import StrataEngine, child_env
        env = child_env(cfg)
        env['STRATA_PREFILL_CPU_SHARE'] = '0'
        def start_engine(args):
            return StrataEngine(cfg['exe'], args, cwd=cfg.get('cwd'), log=cfg.get('log'), env=env)
    return measure(cfg, ids, start_engine, say)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: dloop_calibrate.py strata-model.json (report only; setup saves the choice)')
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), indent=2))
