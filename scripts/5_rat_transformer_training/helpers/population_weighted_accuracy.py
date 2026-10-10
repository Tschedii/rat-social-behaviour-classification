"""Recomputes accuracy weighted by each class's true share of the full dataset, instead of
its accidental share of the eval split.

Why this exists: the train/eval split is session-based, not class-stratified (see
build_index.py), so a class's eval-split frequency can differ wildly from its true frequency.
Exploring is the single most common class in the full dataset (~21%) but only ~4% of eval
bouts — so the standard "accuracy" reported by evaluate.py implicitly discounts how the model
does on your most common real class, and a checkpoint that happens to do badly on Exploring
specifically but well on eval-overrepresented classes can look like the best checkpoint by the
standard number while actually being one of the worst for real-world use.

Reads sklearn's classification_report text output (classification_report_eval.txt, as written
by evaluate.py) — works directly on that file, no need for the raw predictions.

    python helpers/population_weighted_accuracy.py \
        training_output/eval_best/classification_report_eval.txt \
        training_output/eval_epoch152_f10.4855/classification_report_eval.txt \
        ...

Pass as many report paths as you want; each gets its own row in the comparison table.
"""
import argparse
import re
from pathlib import Path

# True full-dataset (train+eval combined) bout counts per class, from build_index.py's class
# counts on window_index.csv — Immobile excluded (dropped from the model entirely; see
# run_on_server.sh). Source: the rebuild_check run, 2026-08-30ish. If the dataset composition
# ever changes (new sessions, relabeling), regenerate these from a fresh
# `python helpers/build_index.py` run's printed "class counts" and update here.
POPULATION_COUNTS = {
    "Air sniffing": 424,
    "Approaching/Following": 316,
    "Body touch": 143,
    "Exploring": 446,
    "Grooming": 102,
    "Investigation": 276,
    "Locomotion": 101,
    "Rearing": 145,
    "Tail investigation": 159,
}

_ROW_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z/ ]+?)\s+(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+"
    r"(?P<f1>[\d.]+)\s+(?P<support>\d+)\s*$"
)


def parse_classification_report(path):
    """Returns {class_name: {"precision":, "recall":, "f1":, "support":}}, plus the report's
    own overall accuracy, macro-F1, and eval-weighted (support-weighted) accuracy — parsed
    straight from sklearn's fixed-width text format, not re-derived."""
    text = Path(path).read_text()
    per_class = {}
    overall = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name in ("accuracy",):
            # sklearn's "accuracy" row omits precision/recall columns; support is the last
            # number, the value before it (in the recall column position for this regex) is
            # actually the accuracy score for that malformed-relative-to-other-rows line —
            # handled separately below since the column layout differs.
            continue
        vals = dict(
            precision=float(m.group("precision")), recall=float(m.group("recall")),
            f1=float(m.group("f1")), support=int(m.group("support")),
        )
        if name in ("macro avg", "weighted avg"):
            overall[name] = vals
        elif name in POPULATION_COUNTS:
            per_class[name] = vals
        else:
            raise ValueError(f"{path}: unrecognized class name {name!r} — is POPULATION_COUNTS "
                              f"in this script out of date with the label set?")

    acc_match = re.search(r"^\s*accuracy\s+([\d.]+)\s+(\d+)\s*$", text, re.MULTILINE)
    if not acc_match:
        raise ValueError(f"{path}: couldn't find the 'accuracy' line — is this a "
                          f"classification_report_eval.txt from evaluate.py?")
    overall["accuracy"] = float(acc_match.group(1))
    overall["eval_support"] = int(acc_match.group(2))

    missing = set(POPULATION_COUNTS) - set(per_class)
    if missing:
        raise ValueError(f"{path}: report is missing expected classes {missing} — "
                          f"stale POPULATION_COUNTS or a different label set than expected?")
    return per_class, overall


def population_weighted_accuracy(per_class):
    pop_total = sum(POPULATION_COUNTS.values())
    return sum(per_class[c]["recall"] * POPULATION_COUNTS[c] for c in POPULATION_COUNTS) / pop_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="one or more classification_report_eval.txt paths")
    args = ap.parse_args()

    pop_total = sum(POPULATION_COUNTS.values())
    print(f"population reference: {pop_total} bouts across {len(POPULATION_COUNTS)} classes "
          f"(Immobile excluded)\n")

    rows = []
    for path in args.reports:
        per_class, overall = parse_classification_report(path)
        pop_acc = population_weighted_accuracy(per_class)
        label = Path(path).parent.name or Path(path).stem
        rows.append((label, overall["macro avg"]["f1"], overall["accuracy"], pop_acc))

    name_w = max(len(r[0]) for r in rows)
    header = f"{'checkpoint':<{name_w}}  {'macro-F1':>8}  {'eval-wt acc':>11}  {'pop-wt acc':>10}  {'gap':>6}"
    print(header)
    print("-" * len(header))
    for label, macro_f1, eval_acc, pop_acc in sorted(rows, key=lambda r: -r[3]):
        print(f"{label:<{name_w}}  {macro_f1:8.3f}  {eval_acc*100:10.1f}%  {pop_acc*100:9.1f}%  "
              f"{(eval_acc - pop_acc)*100:5.1f}pp")
    print("\nranked by population-weighted accuracy (best real-world estimate first).")


if __name__ == "__main__":
    main()
