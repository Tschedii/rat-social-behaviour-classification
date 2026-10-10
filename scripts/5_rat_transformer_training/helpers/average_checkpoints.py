"""SWA-style weight averaging across several checkpoints of the same run.

Averaging weights (not predictions) is the cheap version of this idea — one forward pass at
inference time instead of N — and is well-suited to exactly the situation this project hit:
eval macro-F1 oscillating around a plateau (0.43-0.49) rather than trending anywhere, across
several genuinely different local optima rather than one clear best epoch. Averaging their
weights tends to land in a flatter, better-generalizing region between them.

    python helpers/average_checkpoints.py \
        --checkpoints training_output/top_checkpoints/*.pt \
        --out training_output/averaged_top5.pt

All input checkpoints must share the exact same architecture (same run, just different
epochs) — this is checked before averaging, not assumed. The optimizer state is deliberately
NOT carried into the output checkpoint: an average of several epochs' Adam momentum buffers
isn't meaningful, and dropping the key means a future --resume correctly starts fresh rather
than silently loading something nonsensical.
"""
import argparse
import glob
from pathlib import Path

import torch


def average_state_dicts(state_dicts):
    """Elementwise mean across a list of state_dicts with identical keys/shapes. Only
    floating-point tensors are averaged; any non-float tensor (integer buffers, if a future
    architecture change ever adds one) is instead cross-checked for exact equality across all
    inputs and copied from the first — averaging e.g. a step counter would be meaningless, and
    silently doing so would hide a real problem rather than surface one."""
    keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if list(sd.keys()) != keys:
            raise ValueError(f"checkpoint {i} has different parameter keys than checkpoint 0 — "
                              f"these aren't from compatible architectures, refusing to average")

    averaged = {}
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].is_floating_point():
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            averaged[key] = stacked.mean(dim=0).to(tensors[0].dtype)
        else:
            if not all(torch.equal(t, tensors[0]) for t in tensors[1:]):
                raise ValueError(f"non-float parameter {key!r} differs across checkpoints — "
                                  f"can't average it and copying an arbitrary one would be wrong; "
                                  f"inspect these checkpoints before proceeding")
            averaged[key] = tensors[0]
    return averaged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                     help="checkpoint paths to average, or shell globs (e.g. "
                          "training_output/top_checkpoints/*.pt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = []
    for pattern in args.checkpoints:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    paths = sorted(set(Path(p).resolve() for p in paths), key=lambda p: p.name)
    if len(paths) < 2:
        raise ValueError(f"need at least 2 checkpoints to average, got {len(paths)}: {paths}")

    print(f"averaging {len(paths)} checkpoints:")
    ckpts = [torch.load(p, map_location="cpu") for p in paths]
    for p, c in zip(paths, ckpts):
        print(f"  epoch {c['epoch']:>4d}  eval_macro_f1 {c['eval_macro_f1']:.4f}  {p.name}")

    # sanity: refuse to average checkpoints from genuinely different architectures/label sets —
    # the state_dict key check in average_state_dicts() catches shape/name drift, this catches
    # the case where keys+shapes happen to coincide but the classes they mean don't
    label_maps = [c["label_map"] for c in ckpts]
    if any(lm != label_maps[0] for lm in label_maps[1:]):
        raise ValueError(f"checkpoints have different label_maps, refusing to average: {label_maps}")

    averaged_sd = average_state_dicts([c["model"] for c in ckpts])

    out_ckpt = {
        "model": averaged_sd,
        "args": ckpts[0]["args"],
        "label_map": ckpts[0]["label_map"],
        "epoch": -1,  # not a real epoch — this is an average of several
        "eval_macro_f1": None,  # unknown until re-evaluated; deliberately not copied from any source
        "averaged_from": [
            {"path": str(p), "epoch": c["epoch"], "eval_macro_f1": c["eval_macro_f1"]}
            for p, c in zip(paths, ckpts)
        ],
        # no "optimizer" key on purpose — see module docstring
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.out)
    print(f"\nwrote {args.out}")
    print("run evaluate.py against it next — averaging weights doesn't guarantee a better "
          "eval score, only a more likely one; it needs to be checked like any other checkpoint.")


if __name__ == "__main__":
    main()
