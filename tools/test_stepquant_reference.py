#!/usr/bin/env python3
"""Compare the CUDA port against an external STEPQuant checkout (no model).

python tools/test_stepquant_reference.py --upstream /path/to/STEPQuant \
    --binary build-stepquant/stepquant_test
Requires torch and numpy. Uses upstream math as the independent oracle; nothing
from that repository is copied into Strata. Covers all supported precisions,
nonuniform row impact, groups of 32, pivots and recurrent quantization feedback.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    if not 0 <= args.steps <= 4096:
        parser.error("--steps must be between 0 and 4096")
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    from stepquant import QuantizationPlan, StateCodec, delta_step
    from stepquant.quantization import fit_state

    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    heads, key_heads, dim = 5, 1, 128
    bits = torch.tensor([2, 4, 6, 8, 16])[:, None].expand(heads, dim).clone()
    weights = torch.exp(torch.linspace(-1.5, 1.5, heads*dim)).reshape(heads, dim)
    plan = QuantizationPlan("gdn", bits, weights, 32)
    codec = StateCodec(plan)
    # Flatten Strata's (key,head,value), rather than the reference's (head,key,value).
    def state_bytes(x):
        return x[0].permute(1, 0, 2).contiguous().numpy().astype("<f4").tobytes()
    def floats(x):
        return x.contiguous().numpy().astype("<f4").tobytes()

    import numpy as np
    initial = torch.randn(1, heads, dim, dim)*0.1
    initial[:, 0, 0] = 0
    initial[:, 0, 1, 31:33] = torch.tensor([8., -8.])
    updates = []
    with tempfile.TemporaryDirectory() as directory:
        path, trace = Path(directory) / "reference.bin", Path(directory) / "trace.bin"
        with path.open("wb") as f:
            f.write(struct.pack("<iii", heads, key_heads, args.steps))
            f.write(bits[:, 0].numpy().astype("<i4").tobytes())
            f.write(floats(weights))
            f.write(state_bytes(initial))
            for _ in range(args.steps):
                q = torch.randn(1, key_heads, dim)
                k = torch.randn(1, key_heads, dim)
                q /= (q.square().sum(-1, keepdim=True)+1.e-6).sqrt()
                k /= (k.square().sum(-1, keepdim=True)+1.e-6).sqrt()
                v = torch.randn(1, heads, dim)*0.2
                gate = -torch.rand(1, heads)*0.05
                beta = torch.sigmoid(torch.randn(1, heads))
                updates.append((q, k, v, gate, beta))
                for x in updates[-1]:
                    f.write(floats(x))
        subprocess.run([str(args.binary.resolve()), "--reference", str(path), "--trace", str(trace)], check=True)
        raw = trace.read_bytes()
        offset = 0
        def take(count):
            nonlocal offset
            x = torch.from_numpy(np.frombuffer(raw, dtype="<f4", count=count, offset=offset).copy())
            offset += count*4
            return x
        def take_state():
            return take(heads*dim*dim).reshape(dim, heads, dim).permute(1, 0, 2)[None].contiguous()
        def check_payload(x, decoded):
            nonlocal offset
            _, ref_rows, ref_cols, ref_groups, _ = fit_state(x, plan)
            boundary_codes = 0
            for head, b in enumerate(bits[:, 0].tolist()):
                if b == 16:
                    nbytes = dim*dim*2
                    fp = np.frombuffer(raw, dtype="<f2", count=dim*dim, offset=offset).copy()
                    offset += nbytes
                    torch.testing.assert_close(decoded[0, head], torch.from_numpy(fp.astype("float32")).reshape(dim, dim), atol=0, rtol=0)
                    torch.testing.assert_close(decoded[0, head], x[0, head].half().float(), atol=0, rtol=0)
                    continue
                groups = 4 if b == 2 else 1
                nscale = dim*groups + dim
                scales = np.frombuffer(raw, dtype="<f2", count=nscale, offset=offset).copy().astype("float32")
                offset += nscale*2
                rows = torch.from_numpy(scales[:dim*groups]).reshape(dim, groups)
                cols = torch.from_numpy(scales[dim*groups:])
                reference_rows = ref_groups[0, 0].float() if b == 2 else ref_rows[0, head, :, None].float()
                # One FP16 ulp, plus the subnormal spacing. Different parallel
                # reductions can land on opposite sides of a FP16 rounding boundary.
                torch.testing.assert_close(rows, reference_rows, atol=2**-24, rtol=1e-3)
                # Fit columns using the rows actually stored by CUDA. This
                # isolates column fitting from a one-ulp row-scale change.
                expanded = rows.repeat_interleave(dim//groups, -1)
                scaled = x[0, head]/expanded
                qmax = 3 if b == 2 else 2**(b-1)-1
                def fp16_scale(a):
                    return a.clamp(2**-24, 65504).half().float()
                c0 = fp16_scale((scaled.abs()/qmax).amax(0))
                y0 = scaled/c0
                z0 = (torch.where(y0 >= 0, 1., -1.)*torch.where(y0.abs() >= 2, 3., 1.)
                      if b == 2 else y0.round().clamp(-qmax, qmax))
                fitted_values = expanded*z0
                weight = weights[head, :, None].square()
                numerator = (weight*fitted_values*x[0, head]).sum(0)
                denominator = (weight*fitted_values.square()).sum(0)
                fitted_cols = fp16_scale(torch.where(denominator > 0,
                                       numerator/denominator.clamp_min(1e-30), c0))
                torch.testing.assert_close(cols, fitted_cols, atol=2**-24, rtol=1e-3)
                nbytes = dim*dim*b//8
                payload = np.frombuffer(raw, dtype="uint8", count=nbytes, offset=offset)
                offset += nbytes
                unpacked = np.unpackbits(payload, bitorder="little").reshape(-1, b)
                codes = torch.from_numpy((unpacked @ (1 << np.arange(b))).astype("int32")).reshape(dim, dim)
                levels = codes*2-3 if b == 2 else codes-(2**(b-1)-1)
                expanded = rows.repeat_interleave(dim//groups, -1)
                reconstructed = expanded*cols[None, :]*levels
                torch.testing.assert_close(decoded[0, head], reconstructed, atol=0, rtol=0)
                y = x[0, head]/expanded/cols[None, :]
                if b == 2:
                    nearest = torch.where(y >= 0, 1, -1)*torch.where(y.abs() >= 2, 3, 1)
                    distance = (y.abs()-2).abs()
                else:
                    nearest = y.round().clamp(-(2**(b-1)-1), 2**(b-1)-1).int()
                    distance = ((y.abs() % 1)-0.5).abs()
                different = nearest != levels
                assert not (different & (distance > 1e-5)).any(), "codes are not nearest stored-scale levels"
                boundary_codes += int(different.sum())
            return boundary_codes
        actual = take_state()
        rounding_boundaries = check_payload(initial, actual)
        expected = codec.decode(codec.encode(initial))
        # Upstream packed CUDA tests use atol=5e-3, rtol=5e-3 for fits. FP16
        # scale boundaries make reduction-order differences visible in the codes.
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
        max_fit = float((actual-expected).abs().max())
        max_readout = 0.
        for step, (q, k, v, gate, beta) in enumerate(updates):
            # Both implementations start this update from the ACTUAL packed state
            # carried by CUDA. This tests feedback without confusing accumulated
            # quantizer-boundary drift with an incorrect recurrence equation.
            output, updated = delta_step(actual, q.expand(-1, heads, -1)/(dim**0.5),
                                          k.expand(-1, heads, -1), v, gate, beta)
            actual_output = take(heads*dim).reshape(1, heads, dim)
            actual_updated = take_state()
            torch.testing.assert_close(actual_updated, updated, atol=1e-6, rtol=1e-4, msg=f"update step {step}")
            actual = take_state()
            rounding_boundaries += check_payload(actual_updated, actual)
            expected = codec.decode(codec.encode(actual_updated))
            torch.testing.assert_close(actual_output, output, atol=1e-5, rtol=1e-4, msg=f"readout step {step}")
            # Exact codes/reconstruction and fitted scales were checked above.
            # A different FP16 scale can legitimately select the other nearest
            # level at a decision boundary; report, rather than mask, that drift.
            max_fit = max(max_fit, float((actual-expected).abs().max()))
            max_readout = max(max_readout, float((actual_output-output).abs().max()))
        assert offset == len(raw), "trailing/missing trace data"
        print(f"upstream parity PASS: {args.steps} updates, max readout error {max_readout:.9g}, "
              f"max fit error {max_fit:.9g}, stored-scale rounding ties {rounding_boundaries}")


if __name__ == "__main__":
    main()
