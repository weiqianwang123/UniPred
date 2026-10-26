"""Run a UniPred experiment.

    python -m unipred.main --config configs/satellites/unipred.yaml --seed 0

Takes the same flags as predicators/main.py, plus ``--config FILE`` (see
unipred/config.py). Flags given explicitly override the configs. torch uses
--torch_num_threads threads (see unipred/settings.py).
"""

import argparse
import os
import sys


def main() -> None:
    """Expand the configs, register UniPred and run predicators' main."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        # predicators needs a fixed hash seed; it is read at start-up.
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable,
                 [sys.executable, "-m", "unipred.main"] + sys.argv[1:])
    # pylint: disable=import-outside-toplevel
    import torch

    import unipred  # pylint: disable=unused-import
    from predicators import main as predicators_main
    from unipred.config import expand_config_args
    from unipred.settings import UniPredSettings
    args = expand_config_args(sys.argv[1:])
    threads = argparse.ArgumentParser(add_help=False)
    threads.add_argument("--torch_num_threads",
                         type=int,
                         default=UniPredSettings.torch_num_threads)
    torch.set_num_threads(threads.parse_known_args(args)[0].torch_num_threads)
    if "--log_file" in args:
        log_dir = os.path.dirname(args[args.index("--log_file") + 1])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    sys.argv = [sys.argv[0]] + args
    predicators_main.main()


if __name__ == "__main__":
    main()
