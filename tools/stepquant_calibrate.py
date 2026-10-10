#!/usr/bin/env python3
"""Calibrate Strata GDN traces using the STEPQuant equations and global head DP.

Independent NumPy implementation; no upstream code or model adapter is imported.
The trace stores post-convolution q/k and the engine's unquantized FP32 update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def read_statistics(path):
    with Path(path).open('rb') as f:
        header = f.read(28)
        if len(header) != 28:
            raise ValueError(f'{path}: truncated header')
        magic, version, heads, keys, size, every, limit = struct.unpack('<7i', header)
        if magic != 0x53515452 or version != 1 or size != 128 or not 0 < keys <= heads <= 65535 or heads % keys or every < 1 or limit < every:
            raise ValueError(f'{path}: invalid trace geometry')
        omega = np.zeros((heads, size), np.float64)
        decay = np.zeros(heads, np.float64)
        samples, tokens = [], 0
        def floats(n):
            raw = f.read(n * 4)
            if len(raw) != n * 4:
                raise ValueError(f'{path}: truncated record')
            x = np.frombuffer(raw, dtype='<f4')
            if not np.isfinite(x).all():
                raise ValueError(f'{path}: nonfinite calibration data')
            return x
        while flag := f.read(4):
            if len(flag) != 4 or tokens >= limit:
                raise ValueError(f'{path}: invalid record count')
            sampled, = struct.unpack('<i', flag)
            if sampled not in (0, 1):
                raise ValueError(f'{path}: invalid sample flag')
            qk = floats(2 * keys * size).reshape(2, keys, size)
            q, k = qk[:, np.arange(heads) % keys]
            q = q / np.float32(np.sqrt(size))
            gate, beta = floats(heads), floats(heads)
            if (gate > 0).any() or (beta < 0).any() or (beta > 1).any():
                raise ValueError(f'{path}: invalid decay/beta')
            transported = np.exp(gate)[:, None] * (q - beta[:, None] * k * (k * q).sum(-1, keepdims=True))
            omega += transported ** 2
            decay += gate
            tokens += 1
            if sampled:
                samples.append(floats(heads * size * size).reshape(size, heads, size).transpose(1, 0, 2).copy())
        if not tokens or not samples:
            raise ValueError(f'{path}: no sampled states; collect at least {every} direct decode tokens')
        return dict(omega=omega / tokens, decay=decay / tokens, samples=samples, tokens=tokens, heads=heads)


def impact_factors(omega):
    log_w = np.log(np.maximum(omega, 1e-20)) / 8
    return np.exp(log_w - log_w.mean(-1, keepdims=True)).astype(np.float32)


def lifetime(log_decay, horizon):
    if horizon < 1 or not np.isfinite(log_decay).all() or (log_decay > 0).any():
        raise ValueError('invalid lifetime inputs')
    x = 2 * log_decay
    result = np.full_like(x, float(horizon))
    np.divide(np.expm1(horizon * x), np.expm1(x), out=result, where=x != 0)
    return result


def half_scale(x):
    return np.clip(x, 2**-24, 65504).astype(np.float16).astype(np.float32)


def fit(x, w, bits):
    """Dual-axis fit with separately rounded FP16 factors and grouped INT2 rows."""
    groups = 4 if bits == 2 else 1
    grouped = x.reshape(x.shape[0], 128, groups, 128 // groups)
    rows = half_scale(np.sqrt(np.maximum(np.abs(grouped).mean(-1), 1e-20) / w[:, :, None]))
    rows = np.repeat(rows, 128 // groups, axis=-1)
    qmax = 3 if bits == 2 else (1 << (bits - 1)) - 1
    scaled = x / rows
    columns = half_scale(np.max(np.abs(scaled) / qmax, axis=-2, keepdims=True))
    def quant(c):
        y = scaled / c
        if bits == 2:
            return np.where(y >= 0, 1, -1).astype(np.float32) * np.where(np.abs(y) >= 2, 3, 1).astype(np.float32)
        return np.clip(np.rint(y), -qmax, qmax)
    rz = rows * quant(columns)
    weights = w[:, :, None] ** 2
    numerator = (weights * rz * x).sum(-2, keepdims=True)
    denominator = (weights * (rz * rz)).sum(-2, keepdims=True)
    fitted = np.divide(numerator, np.maximum(denominator, 1e-30))
    columns = half_scale(np.where(denominator > 0, fitted, columns))
    return rows * columns * quant(columns)


def allocate(cost, candidates, budget):
    """Exact dynamic programming across whole heads, including unused budget."""
    cost = np.asarray(cost, np.float64)
    candidates = np.asarray(candidates, np.int64)
    n = len(cost)
    if budget < n * candidates.min():
        raise ValueError('infeasible integer budget; reduce the number of FP16 pivots')
    if n == 0:
        return np.empty(0, np.int64)
    budget = min(int(budget), n * int(candidates.max())) // 2
    costs = candidates // 2
    previous = np.full(budget + 1, np.inf); previous[0] = 0
    parents = np.full((n, budget + 1), -1, np.int8)
    for i in range(n):
        current = np.full_like(previous, np.inf)
        for choice, units in enumerate(costs):
            if units > budget: continue
            trial = previous[:budget + 1 - units] + cost[i, choice]
            better = trial < current[units:]
            current[units:][better] = trial[better]
            parents[i, units:][better] = choice
        previous = current
    remaining = int(previous.argmin())
    if not np.isfinite(previous[remaining]):
        raise ValueError('no feasible assignment')
    result = np.empty(n, np.int64)
    for i in range(n - 1, -1, -1):
        choice = parents[i, remaining]
        result[i] = candidates[choice]; remaining -= costs[choice]
    return result


def calibrate(statistics, nominal=6, pivots=32, horizon=2048):
    if nominal not in (4, 6) or not statistics:
        raise ValueError('provide statistics and nominal bits 4 or 6')
    candidates = [2, 4, 6, 8] if nominal == 4 else [4, 6, 8]
    names = sorted(statistics)
    weights, losses = {}, []
    for name in names:
        stat = statistics[name]
        w = impact_factors(stat['omega']); weights[name] = w
        loss = []
        for bits in candidates:
            total = np.zeros(stat['heads'], np.float64)
            for state in stat['samples']:
                total += (((fit(state, w, bits) - state) ** 2) * w[:, :, None] ** 2).mean((1, 2))
            loss.append(total / len(stat['samples']))
        losses.append(np.stack(loss, axis=-1) * lifetime(stat['decay'], horizon)[:, None])
    cost = np.concatenate(losses)
    if not 0 <= pivots <= len(cost):
        raise ValueError('pivot count exceeds the number of heads')
    initial = allocate(cost, candidates, nominal * len(cost))
    risk = cost[np.arange(len(cost)), np.searchsorted(candidates, initial)]
    pivot = np.zeros(len(cost), bool)
    pivot[np.argsort(-risk, kind='stable')[:pivots]] = True
    assignment = np.full(len(cost), 16, np.int64)
    assignment[~pivot] = allocate(cost[~pivot], candidates, nominal * len(cost) - 8 * pivots)
    offset, plans = 0, {}
    for name in names:
        heads = statistics[name]['heads']
        plans[name] = (assignment[offset:offset + heads], weights[name])
        offset += heads
    return plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('traces', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--bits', type=int, choices=(4, 6), default=6)
    p.add_argument('--pivots', type=int, default=32); p.add_argument('--horizon', type=int, default=2048)
    p.add_argument('--layers', type=int, default=48); p.add_argument('--qsa-interval', type=int, default=4)
    p.add_argument('--heads', type=int, default=48)
    args = p.parse_args()
    if args.layers < 1 or args.qsa_interval < 1 or args.heads < 1:
        p.error('geometry must be positive')
    expected = [i for i in range(args.layers) if i % args.qsa_interval != args.qsa_interval - 1]
    paths = {i: args.traces / f'layer-{i}.bin' for i in expected}
    statistics = {i: read_statistics(path) for i, path in paths.items()}
    if any(s['heads'] != args.heads for s in statistics.values()):
        raise ValueError('trace heads differ from requested geometry')
    plans = calibrate(statistics, args.bits, args.pivots, args.horizon)
    with args.output.open('w') as f:
        f.write(f'STRATA_STEPQUANT 1 128 {args.heads} {len(plans)}\n')
        for layer, (bits, impact) in plans.items():
            f.write(str(layer) + '\n' + ' '.join(map(str, bits)) + '\n')
            f.write(' '.join(format(float(v), '.9g') for v in impact.flat) + '\n')
    counts = {str(b): sum(int((bits == b).sum()) for bits, _ in plans.values()) for b in (2, 4, 6, 8, 16)}
    report = dict(nominal_bits=args.bits, pivots=args.pivots, horizon=args.horizon, head_counts=counts,
                  tokens={str(i): s['tokens'] for i, s in statistics.items()},
                  trace_sha256={str(i): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() for i, path in paths.items()},
                  plan_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    main()
