"""Setup script."""
from setuptools import find_packages, setup

setup(
    name="predicators",
    version="0.1.0",
    packages=find_packages(include=["predicators", "predicators.*"]),
    install_requires=[
        "numpy>=1.23,<2",
        "gym==0.23.1",
        "torch",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "imageio==2.22.2",
        "imageio-ffmpeg",
        "opencv-python",
        "pillow",
        "tabulate==0.9.0",
        "dill==0.3.8",
        "cloudpickle",
        "pathos",
        "pyperplan",
        "pyyaml==6.0.2",
        "natsort",
        "tqdm",
        "z3-solver==4.13.3",
    ],
    include_package_data=True,
    extras_require={
        "wandb": ["wandb"],
        "develop": [
            "pytest==7.1.3",
            "pytest-cov==2.12.1",
            "pytest-pylint==0.18.0",
            "mypy==1.8.0",
            "pylint==2.14.5",
            "types-PyYAML",
            "yapf==0.32.0",
            "docformatter==1.4",
            "isort==5.10.1",
        ]
    })
