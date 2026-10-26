"""Shared test fixtures."""

import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# As unipred.main does (--torch_num_threads): the models are small.
torch.set_num_threads(1)


@pytest.fixture(scope="session", name="processed")
def _processed(tmp_path_factory):
    """The synthetic real-robot fixture, processed with the fake perception
    backends; returns the output directory."""
    out = tmp_path_factory.mktemp("processed")
    subprocess.run([
        sys.executable, "examples/real_robot/process_demos.py", "--raw",
        "examples/real_robot/fixture/raw", "--out",
        str(out), "--domain", "configs/table_clean_real/domain.yaml",
        "--detector", "fake", "--encoder", "fake"
    ],
                   cwd=REPO,
                   check=True,
                   capture_output=True)
    return str(out)
