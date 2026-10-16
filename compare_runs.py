"""
Where do two runs of the same settings (e.g. two machines) part ways? Reads the per-sample
<id>_fingerprint.json and <id>_block_ortho.json that infer_flux2_mice_block_ortho.py writes.

    python compare_runs.py <folder A> <folder B> [--samples 38,90] [--tol 0.02]

Per sample, in run order:
    masks     token count + hash of every instance mask. Any difference = the inputs differ (masking), not rounding
    noise     the initial latent. A difference = seed / generator device
    text      every 200-token prompt segment (global, then each local prompt). A difference = text encoder
              (transformers version, attention backend, tokenizer / chat template)
    steps     per denoising step: mean |input| and |velocity| on each instance and the background
    blocks    the steering stats of every block (edit size, copy fraction, ...), in call order
The first entry whose relative difference exceeds --tol is reported. Inputs that match while steps drift
apart gradually = rounding amplified by the trajectory; a jump at one step/block = where it flips.
Also prints the two env_*.json files side by side where they differ.
"""
import argparse
import json
from pathlib import Path


def rel(a, b):
    if a is None or b is None:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / max(abs(a), abs(b), 1e-6)


def flat(x, prefix=""):
    """Nested dict/list -> {path: leaf}."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = x
    return out


def first_diff(a, b, tol):
    fa, fb = flat(a), flat(b)
    for k in fa:
        va, vb = fa[k], fb.get(k, "<missing>")
        if isinstance(va, bool) or not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            if va != vb:
                return k, va, vb
        elif rel(va, vb) > tol:
            return k, va, vb
    extra = set(fb) - set(fa)
    return (sorted(extra)[0], "<missing>", fb[sorted(extra)[0]]) if extra else None


def env_diff(a: Path, b: Path):
    ea = sorted(a.glob("env_*.json"))
    eb = sorted(b.glob("env_*.json"))
    if not ea or not eb:
        print("(no env_*.json in one of the folders)")
        return
    A, B = json.load(open(ea[0])), json.load(open(eb[0]))
    print(f"env: {ea[0].name}  vs  {eb[0].name}")
    for k in sorted(set(A) | set(B)):
        if A.get(k) != B.get(k):
            print(f"  {k:<22} {str(A.get(k))[:60]:<62} {str(B.get(k))[:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=str)
    parser.add_argument("b", type=str)
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--tol", type=float, default=0.02, help="Relative difference that counts as diverged")
    args = parser.parse_args()
    A, B = Path(args.a), Path(args.b)
    env_diff(A, B)

    ids = sorted({f.name.split("_fingerprint")[0] for f in A.glob("*_fingerprint.json")}
                 & {f.name.split("_fingerprint")[0] for f in B.glob("*_fingerprint.json")}, key=lambda x: int(x))
    if args.samples:
        ids = [i for i in ids if i in args.samples.split(",")]
    if not ids:
        print("No sample has a fingerprint in both folders (rerun with this version of the script).")
        return
    for sid in ids:
        fa, fb = json.load(open(A / f"{sid}_fingerprint.json")), json.load(open(B / f"{sid}_fingerprint.json"))
        print(f"\nsample {sid}")
        verdict = None
        for part in ("masks", "noise", "text"):
            d = first_diff(fa.get(part), fb.get(part), args.tol)
            print(f"  {part:<6} " + ("same" if d is None else f"DIFFERENT at {d[0]}: {d[1]} vs {d[2]}"))
            if d is not None and verdict is None:
                verdict = {"masks": "inputs differ: the masks", "noise": "inputs differ: the initial noise",
                           "text": "inputs differ: the prompt embeddings"}[part]
        for sa, sb in zip(fa.get("steps", []), fb.get("steps", [])):
            d = first_diff(sa, sb, args.tol)
            worst = max((rel(x, y) for k, x in flat(sa).items() if isinstance(x, (int, float))
                         for y in [flat(sb).get(k)] if isinstance(y, (int, float))), default=0.0)
            print(f"  step {sa['step']}: max rel diff {worst:.4f}" + ("" if d is None else f"  (first > tol: {d[0]})"))
        ba, bb = A / f"{sid}_block_ortho.json", B / f"{sid}_block_ortho.json"
        if ba.exists() and bb.exists():
            sa, sb = json.load(open(ba)), json.load(open(bb))
            sa, sb = (sa.get("steering", sa) if isinstance(sa, dict) else sa), (sb.get("steering", sb) if isinstance(sb, dict) else sb)
            for i, (ra, rb) in enumerate(zip(sa, sb)):
                d = first_diff(ra, rb, args.tol)
                if d is not None:
                    print(f"  blocks: first divergence at call {i} (step {ra.get('step')}, {ra.get('layer')}, "
                          f"{ra.get('stream')}): {d[0]} {d[1]} vs {d[2]}")
                    break
            else:
                print(f"  blocks: all {min(len(sa), len(sb))} block records within tol")
        da, db = A / f"{sid}_diag.json", B / f"{sid}_diag.json"
        if da.exists() and db.exists():  # --diagnose on both machines
            ga, gb = json.load(open(da)), json.load(open(db))
            for part in ("processor_state_at_start", "instance_tokens", "binding", "text_embed", "text_mass", "progress"):
                d = first_diff(ga.get(part), gb.get(part), args.tol)
                print(f"  diag {part:<24} " + ("same" if d is None else f"DIFFERENT at {d[0]}: {d[1]} vs {d[2]}"))
                if d is not None and verdict is None and part in ("processor_state_at_start", "instance_tokens", "binding"):
                    verdict = f"inputs differ: {part} (see the diag line above)"
        print("  -> " + (verdict or "inputs identical: any difference is numerical (rounding), amplified by the steps"))


if __name__ == "__main__":
    main()
