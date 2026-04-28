#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys


MODULES = [
    ("numpy", "numpy"),
    ("huggingface_hub", "huggingface-hub"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("coremltools", "coremltools"),
    ("tokenizers", "tokenizers"),
    ("mlx", "mlx"),
    ("mlx_embeddings", "mlx-embeddings"),
]

PROFILE_REQUIREMENTS = {
    "convert": {
        "numpy",
        "huggingface-hub",
        "torch",
        "transformers",
        "coremltools",
        "tokenizers",
    },
    "mlx": {
        "numpy",
        "huggingface-hub",
        "mlx",
        "mlx-embeddings",
    },
    "all": {distribution_name for _, distribution_name in MODULES},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_REQUIREMENTS),
        default="all",
        help="Dependency profile to require.",
    )
    parser.add_argument(
        "--no-fail-missing-optional",
        action="store_true",
        help="Report missing packages without returning a failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {},
    }
    missing = []
    required = PROFILE_REQUIREMENTS[args.profile]
    for module_name, distribution_name in MODULES:
        if importlib.util.find_spec(module_name) is None:
            rows["packages"][distribution_name] = None
            if distribution_name in required:
                missing.append(distribution_name)
            continue
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
        rows["packages"][distribution_name] = version

    print(json.dumps(rows, indent=2, sort_keys=True))
    if missing and not args.no_fail_missing_optional:
        print()
        print("Missing optional packages:", ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
