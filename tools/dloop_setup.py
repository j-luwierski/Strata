"""NAVER DLoop choices shared by fresh setup and installed-config edits."""
import math

FLAGS = ("--dloop", "--dloop-block-size", "--dloop-gate", "--dloop-max-loops")
MANAGED = ("--spec", "--spec-min-p", "--mtp-max-t")


def arguments(parser):
    parser.add_argument("--dloop", choices=("on", "off"), help="experimental NAVER DLoop with stock MTP weights")
    parser.add_argument("--dloop-block-size", type=int, help="drafts per block (default 3)")
    parser.add_argument("--dloop-gate", type=float, help="latest block log probability threshold (default -0.5)")
    parser.add_argument("--dloop-max-loops", type=int, help="blocks per verification (default 2; block * loops <= 7)")


def requested(a):
    return any(getattr(a, n, None) is not None for n in ("dloop", "dloop_block_size", "dloop_gate", "dloop_max_loops"))


def validate(c):
    b, n, g = c["block_size"], c["max_loops"], c["gate"]
    if not 1 <= b <= 7 or not 1 <= n <= 7 // b:
        raise ValueError("DLoop requires positive block-size * max-loops <= 7 (the verifier's limit)")
    if not math.isfinite(g) or g > 0:
        raise ValueError("DLoop gate must be finite and <= 0")
    return c


def choose(a, ask, say, previous=None):
    previous = previous or {}
    enabled = a.dloop
    if enabled is None and requested(a):
        enabled = "on"
    if enabled is None:
        say("  Experimental NAVER DLoop: draft complete blocks before one target verification.")
        say("  Stock MTP weights, without the paper's loop-aware training. Speed must be measured on your PC.")
        enabled = "on" if ask("Use DLoop?", ["y", "n"], "y" if previous.get("enabled") else "n", a.yes) == "y" else "off"
    if enabled == "off":
        if any(getattr(a, n) is not None for n in ("dloop_block_size", "dloop_gate", "dloop_max_loops")):
            raise ValueError("DLoop parameters cannot be combined with --dloop off")
        return {"enabled": False}
    b = a.dloop_block_size
    if b is None:
        b = int(ask("Drafts per DLoop block (1..7)?", [str(i) for i in range(1, 8)], str(previous.get("block_size", 3)), a.yes))
    if not 1 <= b <= 7:
        raise ValueError("DLoop block-size must be 1..7")
    n = a.dloop_max_loops
    if n is None:
        n = int(ask(f"Maximum DLoop blocks (1..{7 // b})?", [str(i) for i in range(1, 7 // b + 1)],
                    str(min(previous.get("max_loops", 2), 7 // b)), a.yes))
    g = a.dloop_gate
    if g is None:
        say("  Gate: closer to zero means fewer extensions; more negative permits less confident blocks.")
        gates = ["-0.25", "-0.5", "-1", "-2", str(previous.get("gate", -0.5))]
        g = float(ask("DLoop gate (sum of natural logs)?", gates, str(previous.get("gate", -0.5)), a.yes))
    return validate({"enabled": True, "block_size": b, "max_loops": n, "gate": g})


def value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def drop(args, flags):
    result, i = [], 0
    while i < len(args):
        if args[i] in flags:
            i += 1 if args[i] == "--dloop" else 2
        else:
            result.append(args[i]); i += 1
    return result


def apply(cfg, choice):
    args = list(cfg["args"])
    old = cfg.get("dloop") or {}
    if choice["enabled"]:
        validate(choice)
        if "--mtp" not in args or "--native" not in args:
            raise ValueError("DLoop needs a native model with --mtp; this installation uses a different drafter")
        if cfg.get("parallel", 1) > 1 or any(f in args for f in ("--batch", "--slots", "--lookup-chain", "--spec-oracle")) or \
                int(value(args, "--pipeline-windows") or 0) >= 2:
            raise ValueError("DLoop currently supports serial decoding; turn off parallel/batch, lookup-chain and pipeline-windows >= 2")
        previous = old.get("previous") if old.get("enabled") else None
        if previous is None:
            previous = {f: value(args, f) for f in MANAGED}
        args = drop(args, FLAGS + MANAGED)
        args += ["--dloop", "--dloop-block-size", str(choice["block_size"]), "--dloop-gate", str(choice["gate"]),
                 "--dloop-max-loops", str(choice["max_loops"]), "--spec", str(choice["block_size"] * choice["max_loops"] + 1),
                 "--spec-min-p", "0", "--mtp-max-t", str(choice["block_size"] * choice["max_loops"] + 1)]
        cfg["dloop"] = {**choice, "previous": previous}
    elif old.get("enabled") or "--dloop" in args:
        args = drop(args, FLAGS + MANAGED)
        args += [x for f, v in old.get("previous", {"--spec": "4", "--spec-min-p": "0.5"}).items() if v is not None for x in (f, v)]
        cfg.pop("dloop", None)
    else:
        cfg.pop("dloop", None)
    cfg["args"] = args
    return cfg


def capability(exe, run):
    """Use the installed engine's help, never assume a stock archive supports DLoop."""
    result = run([str(exe), "--help"], capture_output=True, text=True, timeout=60)
    return result.returncode == 0 and "--dloop-block-size" in result.stdout + result.stderr
