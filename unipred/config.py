"""Experiment configs: YAML files of command line flags.

A config file maps flag names to values, for example::

    include: base.yaml
    approach: unipred
    llm_model: gpt-4o
    unipred_derived_selection: true

``include`` (a path or a list of paths, relative to the including file)
loads other configs first; keys of the including file override theirs.
On the command line, ``--config FILE`` may be given several times (later
files override earlier ones), and explicit flags override every config.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import yaml

from predicators.args import create_arg_parser


def load_config(path: str) -> Dict[str, Any]:
    """Load a config file and the files it includes."""
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Config {path} must be a mapping of flags.")
    includes = content.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    config: Dict[str, Any] = {}
    for include in includes:
        config.update(
            load_config(os.path.join(os.path.dirname(path), include)))
    config.update(content)
    return config


def _store_true_flags() -> set:
    parser = create_arg_parser(env_required=False,
                               approach_required=False,
                               seed_required=False)
    return {
        action.dest
        for action in parser._actions  # pylint: disable=protected-access
        if isinstance(action, argparse._StoreTrueAction)  # pylint: disable=protected-access
    }


def _value_to_str(name: str, value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_value_to_str(name, v) for v in value) + "]"
    raise ValueError(f"Unsupported value for flag {name}: {value!r}")


def expand_config_args(argv: List[str]) -> List[str]:
    """Replace each ``--config FILE`` in the arguments by the flags of the
    file and return the merged arguments (without the program name)."""
    store_true = _store_true_flags()
    flags: Dict[str, Any] = {}
    explicit: Dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            raise ValueError(f"Expected a flag, got {token!r}.")
        name = token[2:]
        if name in store_true:
            explicit[name] = True
            i += 1
            continue
        if i + 1 >= len(argv):
            raise ValueError(f"Flag {token} has no value.")
        if name == "config":
            flags.update(load_config(argv[i + 1]))
        else:
            explicit[name] = argv[i + 1]
        i += 2
    flags.update(explicit)
    merged: List[str] = []
    for name, value in flags.items():
        if name in store_true:
            if value is True:
                merged.append(f"--{name}")
            elif value is not False:
                raise ValueError(f"Flag {name} takes true or false.")
            continue
        merged.extend([f"--{name}", _value_to_str(name, value)])
    return merged
