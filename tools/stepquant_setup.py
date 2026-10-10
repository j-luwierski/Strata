"""STEPQuant choices and calibration used by the normal Strata installer.

No downloads or GPU work at import time. Calibration runs the installed engine
without a packed-state plan, then installs a completed plan atomically.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'data' / 'stepquant-calibration.txt'


def arguments(parser):
    parser.add_argument('--stepquant', choices=('on', 'off'), help='opt-in STEPQuant GDN state compression (default: off)')
    parser.add_argument('--stepquant-plan', type=Path, help='use an existing Strata STEPQuant plan instead of calibrating')
    parser.add_argument('--stepquant-bits', type=int, choices=(4, 6), help='calibration nominal bit budget (default: 6)')
    parser.add_argument('--stepquant-pivots', type=int, help='number of FP16 pivot heads (default: 32)')
    parser.add_argument('--stepquant-horizon', type=int, help='calibration lifetime horizon in tokens (default: 2048)')
    parser.add_argument('--stepquant-corpus', type=Path, help='UTF-8 calibration text (default: data/stepquant-calibration.txt)')
    parser.add_argument('--stepquant-recalibrate', action='store_true', help='collect fresh full-model traces and replace the plan')


def requested(a):
    return any(getattr(a, k, None) is not None for k in
               ('stepquant', 'stepquant_plan', 'stepquant_bits', 'stepquant_pivots', 'stepquant_horizon', 'stepquant_corpus')) or a.stepquant_recalibrate


def validate_arguments(a, parser):
    if a.stepquant_pivots is not None and a.stepquant_pivots < 0:
        parser.error('--stepquant-pivots must be zero or more')
    if a.stepquant_horizon is not None and a.stepquant_horizon < 1:
        parser.error('--stepquant-horizon must be positive')
    if a.stepquant == 'off' and any(getattr(a, k, None) is not None for k in
            ('stepquant_plan', 'stepquant_bits', 'stepquant_pivots', 'stepquant_horizon', 'stepquant_corpus')):
        parser.error('--stepquant off cannot be combined with STEPQuant calibration options')
    if a.stepquant == 'off' and a.stepquant_recalibrate:
        parser.error('--stepquant off cannot be combined with --stepquant-recalibrate')
    if a.stepquant_plan and (a.stepquant_recalibrate or any(getattr(a, k, None) is not None for k in
            ('stepquant_bits', 'stepquant_pivots', 'stepquant_horizon', 'stepquant_corpus'))):
        parser.error('choose either --stepquant-plan or STEPQuant calibration options')


def path_input(S, question, default, yes):
    if yes:
        return str(default)
    S.flush_typed_ahead()
    return input(f'  {question} [{default}]: ').strip() or str(default)


def choose(S, a, old=None):
    old = dict(old or {})
    if not old.get('stepquant_plan') and '--stepquant-plan' in old.get('args', []):
        old['stepquant_plan'] = old['args'][old['args'].index('--stepquant-plan') + 1]
    saved = old.get('stepquant') or {}
    enabled = a.stepquant == 'on' if a.stepquant is not None else (True if requested(a) else
        S.ask('Use STEPQuant for GDN state compression? (experimental; requires a source build)',
              ['y', 'n'], 'y' if old.get('stepquant_plan') else 'n', a.yes) == 'y')
    if not enabled:
        return None
    S.say('  STEPQuant compresses recurrent GDN state; the model weights and attention KV format keep their chosen precision.')
    source = 'plan' if a.stepquant_plan else 'calibrate'
    if not requested(a):
        source = S.ask('STEPQuant plan: calibrate on this model, or use an existing plan?', ['calibrate', 'plan'],
                       'plan' if old.get('stepquant_plan') else 'calibrate', a.yes)
    # An unchanged installed plan is reused, including for --stepquant on.
    plan = a.stepquant_plan or (old.get('stepquant_plan') if source == 'plan' or not a.stepquant_recalibrate else None)
    if source == 'plan' and not plan:
        plan = path_input(S, 'STEPQuant plan file', '', a.yes)
    if source == 'plan':
        validate_plan(Path(plan).expanduser().resolve())
        return dict(bits=saved.get('bits', 6), pivots=saved.get('pivots', 32),
                    horizon=saved.get('horizon', 2048), corpus=str(saved.get('corpus') or CORPUS),
                    plan=str(Path(plan).expanduser().resolve()),
                    plan_model=saved.get('model') if not a.stepquant_plan else None)
    bits = a.stepquant_bits or int(S.ask('STEPQuant nominal bit budget', ['4', '6'], str(saved.get('bits', 6)), a.yes))
    pivots = a.stepquant_pivots
    if pivots is None:
        default = str(saved.get('pivots', 32))
        pivots = int(S.ask('STEPQuant FP16 pivot heads', list(dict.fromkeys(['0', '16', '32', '64', default])), default, a.yes))
    horizon = a.stepquant_horizon
    if horizon is None:
        default = str(saved.get('horizon', 2048))
        horizon = int(S.ask('STEPQuant lifetime horizon (tokens)', list(dict.fromkeys(['512', '2048', '8192', default])), default, a.yes))
    corpus = a.stepquant_corpus or saved.get('corpus') or CORPUS
    if not a.stepquant_corpus and not a.yes:
        if S.ask('STEPQuant calibration text', ['default', 'custom'], 'default', a.yes) == 'custom':
            corpus = path_input(S, 'UTF-8 text file', corpus, a.yes)
    changed_values = bool(saved) and any(v != saved.get(k, default) for k, v, default in
        [('bits', bits, 6), ('pivots', pivots, 32), ('horizon', horizon, 2048), ('corpus', str(corpus), str(CORPUS))])
    changed = any(getattr(a, k, None) is not None for k in
                  ('stepquant_bits', 'stepquant_pivots', 'stepquant_horizon', 'stepquant_corpus'))
    if a.stepquant_recalibrate or ((changed or changed_values) and not a.stepquant_plan) or source == 'calibrate' and not requested(a):
        plan = None
    if plan:
        validate_plan(Path(plan).expanduser().resolve())
    else:
        corpus = Path(corpus).expanduser().resolve()
        if not corpus.is_file():
            raise ValueError(f'calibration text does not exist: {corpus}')
    return dict(bits=bits, pivots=pivots, horizon=horizon, corpus=str(corpus), plan_model=saved.get('model') if not a.stepquant_plan else None, plan=str(Path(plan).expanduser().resolve()) if plan else None)


def validate_plan(path):
    """Reject truncated/nonfinite/invalid plans before changing the installed config."""
    words = path.read_text(encoding='utf-8').split()
    if len(words) < 5 or words[:3] != ['STRATA_STEPQUANT', '1', '128']:
        raise ValueError(f'{path}: not a Strata STEPQuant plan (import upstream .pt plans with tools/stepquant_plan.py)')
    heads, layers = map(int, words[3:5])
    if not 1 <= heads <= 65535 or not 1 <= layers <= 65535 or len(words) != 5 + layers * (1 + heads * 129):
        raise ValueError(f'{path}: invalid plan dimensions or length')
    seen, at = set(), 5
    for _ in range(layers):
        layer = int(words[at]); at += 1
        if layer < 0 or layer in seen:
            raise ValueError(f'{path}: duplicate or invalid layer')
        seen.add(layer)
        bits = list(map(int, words[at:at + heads])); at += heads
        if any(b not in (2, 4, 6, 8, 16) for b in bits):
            raise ValueError(f'{path}: invalid precision')
        weights = list(map(float, words[at:at + heads * 128])); at += heads * 128
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError(f'{path}: invalid impact factors')
    return heads, seen


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def model_geometry(native):
    from tools.gguf_reader import GGUFFile
    md = GGUFFile(Path(native)).metadata
    prefix = md.get('general.architecture', '') + '.'
    count = int(md.get(prefix + 'block_count', 0))
    interval = int(md.get(prefix + 'full_attention_interval', 0))
    heads = int(md.get(prefix + 'ssm.time_step_rank', 0))
    if count < 1 or interval < 1 or heads < 1 or md.get(prefix + 'ssm.state_size') != 128:
        raise ValueError('this model has no supported 128-element GDN state geometry')
    layers = {i for i in range(count) if i % interval != interval - 1}
    if not layers:
        raise ValueError('this model has no GDN layers')
    return heads, layers


def drop(args, flag, value=True):
    while flag in args:
        i = args.index(flag)
        del args[i:i + 1 + int(value)]


def engine_env(cfg):
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (cfg.get('env') or {}).items()})
    if cfg.get('lib_dirs'):
        key = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[key] = os.pathsep.join([*cfg['lib_dirs'], env.get(key, '')])
    # Diagnostics from a caller must not pollute a calibration or quality run.
    for k in ('STRATA_LOGPOS', 'STRATA_DUMP_FIRST_LOGITS'):
        env.pop(k, None)
    return env


def apply(S, cfg, choice, directory):
    working = copy.deepcopy(cfg)
    _apply(S, working, choice, directory)
    cfg.clear()
    cfg.update(working)


def _apply(S, cfg, choice, directory):
    # Remove an old raw-argument plan too, otherwise it overrides the server config.
    args = list(cfg['args'])
    drop(args, '--stepquant-plan')
    cfg['args'] = args
    cfg.pop('stepquant_plan', None); cfg.pop('stepquant', None)
    if choice is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = None
    if choice['plan']:
        source = Path(choice['plan'])
        if choice.get('plan_model') and '--native' in args and Path(choice['plan_model']).resolve() != Path(args[args.index('--native')+1]).resolve():
            raise ValueError('saved STEPQuant plan belongs to another model; use --stepquant-recalibrate')
        heads, layers = validate_plan(source)
        if '--native' in args:
            if (heads, layers) != model_geometry(args[args.index('--native') + 1]):
                raise ValueError("STEPQuant plan does not cover this model's GDN geometry")
        # Keep imported plans local to this model's installation.
        data = source.read_bytes()
        target = directory / ('state-' + hashlib.sha256(data).hexdigest()[:16] + '.plan')
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
            f.write(data); temporary = Path(f.name)
        os.replace(temporary, target)
        S.ok(f'STEPQuant plan: {target}')
    else:
        from tools.strata_tokenizer import Tokenizer
        native = args[args.index('--native') + 1] if '--native' in args else None
        if native is None:
            raise ValueError('STEPQuant calibration requires a native GDN model (--native)')
        expected_heads, expected_layers = model_geometry(native)
        total_heads = expected_heads * len(expected_layers)
        minimum_bits = 2 if choice['bits'] == 4 else 4
        if not 0 <= choice['pivots'] <= total_heads or choice['bits'] * total_heads - 8 * choice['pivots'] < minimum_bits * (total_heads-choice['pivots']):
            raise ValueError('too many FP16 pivots for this model and bit budget')
        tk = Tokenizer.from_gguf(native)
        text = Path(choice['corpus']).read_text(encoding='utf-8')
        tokens = tk.encode(text)[:513]   # observer records at most 512 updates per layer
        if len(tokens) < 9:
            raise ValueError('STEPQuant calibration text must contain at least 9 tokens')
        S.say(f'  Calibrating STEPQuant on the full model ({len(tokens)-1} prompt tokens, FP32 states) ...')
        # Separate single-GPU trace process: no rejected drafts, batching, or checkpoint reuse.
        with tempfile.TemporaryDirectory(prefix='stepquant-', dir=directory) as tmp:
            tmp = Path(tmp)
            ids = tmp / 'prompt.ids'; ids.write_text(' '.join(map(str, tokens)))
            trace = tmp / 'traces'
            command = list(args)
            for flag in ('--prefill', '--spec', '--max-context', '--max-new', '--tokens-file', '--stepquant-trace',
                         '--gpus', '--layer-split', '--pipeline-windows', '--batch', '--prompt-cache', '--kv-resident'):
                drop(command, flag)
            for flag in ('--serve',):
                drop(command, flag, False)
            command += ['--prefill', '64', '--spec', '2', '--max-context', '1024', '--max-new', '1',
                        '--tokens-file', str(ids), '--stepquant-trace', str(trace)]
            gpu = cfg.get('gpu')
            if isinstance(gpu, list):
                gpu = gpu[0]
            if gpu is not None:
                drop(command, '--gpu'); command += ['--gpu', str(gpu)]
            command = S.stepquant_trace_command(cfg, command) if hasattr(S, 'stepquant_trace_command') else [cfg['exe'], *command]
            result = S.run(command, cwd=cfg.get('cwd'), env=engine_env(cfg), check=False)
            if result is not None and result.returncode != 0:
                raise ValueError(f'calibration engine exited with code {result.returncode}; see its output above')
            paths = sorted(trace.glob('layer-*.bin'))
            if not paths:
                raise ValueError('the engine produced no GDN calibration traces')
            from tools.stepquant_calibrate import read_statistics, calibrate
            statistics = {int(p.stem.split('-')[1]): read_statistics(p) for p in paths}
            heads = {s['heads'] for s in statistics.values()}
            if len(heads) != 1:
                raise ValueError('inconsistent calibration head geometry')
            h = heads.pop()
            if h != expected_heads or set(statistics) != expected_layers or any(s['tokens'] != len(tokens)-1 for s in statistics.values()):
                raise ValueError('incomplete calibration: expected every GDN layer and every prompt update')
            plans = calibrate(statistics, choice['bits'], choice['pivots'], choice['horizon'])
            out = tmp / 'state.plan'
            with out.open('w') as f:
                f.write(f'STRATA_STEPQUANT 1 128 {h} {len(plans)}\n')
                for layer, (bits, impact) in sorted(plans.items()):
                    f.write(str(layer) + '\n' + ' '.join(map(str, bits)) + '\n')
                    f.write(' '.join(format(float(v), '.9g') for v in impact.flat) + '\n')
            validate_plan(out)
            report = {**{k: v for k, v in choice.items() if k not in ('plan', 'plan_model')},
                      'model': str(Path(native).resolve()), 'tokens': len(tokens)-1,
                      'corpus_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'plan_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                      'trace_sha256': {p.stem: file_sha256(p) for p in paths},
                      'head_counts': {str(b): sum(int((bits == b).sum()) for bits, _ in plans.values()) for b in (2, 4, 6, 8, 16)}}
            # The config is written by setup only after both files exist.
            report_tmp = tmp / 'state.plan.json'; report_tmp.write_text(json.dumps(report, indent=2) + '\n')
            target = directory / ('state-' + report['plan_sha256'][:16] + '.plan')
            os.replace(out, target)
            os.replace(report_tmp, target.with_suffix('.plan.json'))
        S.ok(f'STEPQuant calibrated: {target}')
    cfg['stepquant_plan'] = str(target.resolve())
    cfg['stepquant'] = {k: v for k, v in choice.items() if k not in ('plan', 'plan_model')}
    if '--native' in args:
        cfg['stepquant']['model'] = str(Path(args[args.index('--native')+1]).resolve())
