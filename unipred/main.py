"""Run a UniPred experiment.

    python -m unipred.main --config configs/satellites/unipred.yaml --seed 0

Takes the same flags as predicators/main.py, plus ``--config FILE`` (see
unipred/config.py). Flags given explicitly override the configs.
"""

import os
import sys


def main() -> None:
    """Expand the configs, register UniPred and run predicators' main."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        # predicators needs a fixed hash seed; it is read at start-up.
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable,
                 [sys.executable, "-m", "unipred.main"] + sys.argv[1:])
    import unipred  # pylint: disable=import-outside-toplevel,unused-import
    from predicators import main as predicators_main  # pylint: disable=import-outside-toplevel
    from unipred.config import expand_config_args  # pylint: disable=import-outside-toplevel
    args = expand_config_args(sys.argv[1:])
    if "--log_file" in args:
        log_dir = os.path.dirname(args[args.index("--log_file") + 1])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    sys.argv = [sys.argv[0]] + args
    predicators_main.main()


if __name__ == "__main__":
    main()
