"""Result tables in the form of the paper's: success rates on tasks from
the training distribution (T train) and on the out-of-distribution test
tasks (T test), and the total learning time tau in seconds.

    python -m unipred.eval.summarize --domains satellites blocks \\
        --runs oracle unipred=UniPred ivntr=IVNTR
    python -m unipred.eval.summarize --average \\
        --domains satellites blocks packing \\
        --runs unipred unipred+no_llm_loop="w/o unified bilevel learning"

A run is a directory outputs/<domain>/<run>/ written by scripts/run.sh
(<run> is the config's name, followed by +<overlay> for each overlay), with
one seed<k>/ directory per seed. T test and tau come from
seed<k>/results/*.pkl, T train from seed<k>/results_in_domain/*.pkl. A run
can be given a label with run=label.

Values are mean +- sample standard deviation over seeds; a zero standard
deviation is omitted, and tau is "/" for approaches that do not learn.
With --average, each seed's values are first averaged over the domains
(the seeds that every domain has), giving one column group.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dill as pkl
import numpy as np

_METRICS = ("T train", "T test", "tau")


@dataclass(frozen=True)
class SeedResult:
    """What one seed of a run achieved (success rates in percent)."""
    train_success: Optional[float]
    test_success: Optional[float]
    learning_time: Optional[float]

    def values(self) -> Tuple[Optional[float], ...]:
        """The values in the order of _METRICS."""
        return (self.train_success, self.test_success, self.learning_time)


def _load(results_dir: str) -> Optional[Dict]:
    paths = sorted(glob.glob(os.path.join(results_dir, "*.pkl")))
    if not paths:
        return None
    with open(paths[0], "rb") as f:
        return pkl.load(f)["results"]


def _success(results: Optional[Dict]) -> Optional[float]:
    if results is None or not results["num_total"]:
        return None
    return 100.0 * results["num_solved"] / results["num_total"]


def load_run(outputs: str, domain: str, run: str) -> Dict[int, SeedResult]:
    """The results of every seed of a run, by seed."""
    seeds = {}
    for seed_dir in glob.glob(os.path.join(outputs, domain, run, "seed*")):
        name = os.path.basename(seed_dir)
        if not name[len("seed"):].isdigit():
            continue
        test = _load(os.path.join(seed_dir, "results"))
        train = _load(os.path.join(seed_dir, "results_in_domain"))
        if test is None and train is None:
            continue
        learning_time = None if test is None else float(
            test.get("learning_time", 0.0))
        seeds[int(name[len("seed"):])] = SeedResult(_success(train),
                                                    _success(test),
                                                    learning_time)
    return seeds


def format_values(values: Sequence[Optional[float]], is_time: bool) -> str:
    """Mean +- sample standard deviation of the values that exist."""
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    if is_time and all(v == 0.0 for v in present):
        return "/"
    mean = float(np.mean(present))
    std = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
    text = f"{mean:.1f}"
    if round(std, 1) != 0.0:
        text += f"±{std:.1f}"
    return text


def _cells(seeds: Dict[int, SeedResult]) -> List[str]:
    columns = list(zip(*(r.values() for r in seeds.values()))) or [
        (), (), ()
    ]
    return [
        format_values(column, metric == "tau")
        for metric, column in zip(_METRICS, columns)
    ]


def _averaged(per_domain: List[Dict[int, SeedResult]]) -> Dict[int,
                                                                SeedResult]:
    """Per seed, the mean over the domains (seeds that all domains have)."""
    if not per_domain:
        return {}
    common = set.intersection(*(set(d) for d in per_domain))
    averaged = {}
    for seed in sorted(common):
        values = []
        for i in range(len(_METRICS)):
            column = [d[seed].values()[i] for d in per_domain]
            values.append(None if any(v is None for v in column) else float(
                np.mean(column)))
        averaged[seed] = SeedResult(*values)
    return averaged


def table(outputs: str, domains: Sequence[str],
          runs: Sequence[Tuple[str, str]], average: bool) -> str:
    """A Markdown table with a row per run."""
    groups = ["Average of " + ", ".join(domains)] if average else list(domains)
    header = ["Method"] + [f"{g}: {m}" for g in groups for m in _METRICS]
    rows = []
    for run, label in runs:
        per_domain = [load_run(outputs, d, run) for d in domains]
        if average:
            cells = _cells(_averaged(per_domain))
        else:
            cells = [c for seeds in per_domain for c in _cells(seeds)]
        rows.append([label] + cells)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _parse_runs(items: Sequence[str]) -> List[Tuple[str, str]]:
    runs = []
    for item in items:
        run, _, label = item.partition("=")
        runs.append((run, label or run))
    return runs


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the table described by the command line (see module doc)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run directory names, optionally run=label")
    parser.add_argument("--average", action="store_true",
                        help="average each seed over the domains")
    parser.add_argument("--title", default="")
    args = parser.parse_args(argv)
    text = table(args.outputs, args.domains, _parse_runs(args.runs),
                 args.average)
    if args.title:
        text = f"{args.title}\n\n{text}"
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
