"""Tests for the result tables."""

import dill as pkl

from unipred.eval.summarize import format_values, table


def _write(path, num_solved, learning_time=0.0):
    path.mkdir(parents=True)
    with open(path / "run.pkl", "wb") as f:
        pkl.dump(
            {
                "results": {
                    "num_solved": num_solved,
                    "num_total": 10,
                    "learning_time": learning_time
                }
            }, f)


def test_format_values():
    """Mean and sample standard deviation; zero deviations are omitted;
    tau is '/' for approaches that do not learn."""
    assert format_values([90.0, 100.0], False) == "95.0±7.1"
    assert format_values([100.0, 100.0], False) == "100.0"
    assert format_values([0.0, 0.0], True) == "/"
    assert format_values([None], False) == "-"


def test_table(tmp_path):
    """Rows per run, columns per domain, or averaged over domains per
    seed."""
    for seed, (train, test, tau) in enumerate([(10, 8, 100.0),
                                               (10, 6, 300.0)]):
        run = tmp_path / "satellites" / "unipred+no_semantic" / f"seed{seed}"
        _write(run / "results", test, tau)
        _write(run / "results_in_domain", train)
        run = tmp_path / "blocks" / "unipred+no_semantic" / f"seed{seed}"
        _write(run / "results", 4, 50.0)
        _write(run / "results_in_domain", 10)
    _write(tmp_path / "satellites" / "oracle" / "seed0" / "results", 10)
    runs = [("unipred+no_semantic", "w/o Semantic"), ("oracle", "Oracle")]
    lines = table(str(tmp_path), ["satellites", "blocks"], runs,
                  average=False).splitlines()
    assert lines[2] == ("| w/o Semantic | 100.0 | 70.0±14.1 | 200.0±141.4 "
                        "| 100.0 | 40.0 | 50.0 |")
    assert lines[3] == "| Oracle | - | 100.0 | / | - | - | - |"
    lines = table(str(tmp_path), ["satellites", "blocks"], runs,
                  average=True).splitlines()
    assert lines[2] == "| w/o Semantic | 100.0 | 55.0±7.1 | 125.0±70.7 |"
    assert lines[3] == "| Oracle | - | - | - |"
