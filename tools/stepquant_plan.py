#!/usr/bin/env python3
"""Export an upstream calibrated GDN plan to Strata's versioned text format.

Requires PyTorch only for reading .pt; STEPQuant itself is not imported. Never
calibrate on another model/checkpoint and reuse that allocation in Strata.
"""
import argparse
import math
from pathlib import Path
import re


def export_plan(artifact, layers, interval, heads):
    if artifact.get("format_version") != 2:
        raise ValueError("expected upstream format_version=2")
    if layers < 1 or interval < 1 or heads < 1:
        raise ValueError("positive layer, interval and head counts required")
    expected = {i for i in range(layers) if i % interval != interval - 1}
    plans = {}
    for name, plan in artifact["plans"].items():
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None and name.isdigit():
            layer = int(name)
        elif match:
            layer = int(match.group(1))
        else:
            raise ValueError(f"cannot determine Strata layer index from {name!r}")
        if layer not in expected or layer in plans:
            raise ValueError(f"unexpected/duplicate/QSA layer {layer}")
        if plan["architecture"] != "gdn" or plan.get("value_group_size", 32) != 32:
            raise ValueError("Strata supports GDN with INT2 value groups of 32")
        bits, impact = plan["bits"], plan["impact"]
        if tuple(bits.shape) != (heads, 128) or tuple(impact.shape) != (heads, 128):
            raise ValueError(f"layer {layer}: expected shape ({heads}, 128)")
        bit_rows = bits.tolist()
        precision = []
        for row in bit_rows:
            if any(b != row[0] for b in row) or row[0] not in (2, 4, 6, 8, 16):
                raise ValueError("GDN precision must be 2/4/6/8/16 per whole head")
            precision.append(int(row[0]))
        weights = [float(w) for row in impact.tolist() for w in row]
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("impact factors must be finite and positive")
        plans[layer] = (precision, weights)
    if set(plans) != expected:
        raise ValueError(f"missing GDN layers: {sorted(expected - set(plans))}")
    lines = [f"STRATA_STEPQUANT 1 128 {heads} {len(plans)}"]
    for layer in sorted(plans):
        bits, weights = plans[layer]
        lines.extend([str(layer), " ".join(map(str, bits)), " ".join(format(w, ".9g") for w in weights)])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--qsa-interval", type=int, default=4)
    parser.add_argument("--heads", type=int, default=48)
    args = parser.parse_args()
    import torch
    # Upstream v2 plans contain tensors and ordinary dictionaries; no pickle globals needed.
    artifact = torch.load(args.input, map_location="cpu", weights_only=True)
    text = export_plan(artifact, args.layers, args.qsa_interval, args.heads)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}; calibrated plan must match the exact Strata checkpoint.")


if __name__ == "__main__":
    main()
