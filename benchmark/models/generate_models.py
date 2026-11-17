#!/usr/bin/env python3
"""Writes the benchmark ONNX models.

Every generators/<file>.py defines the models it builds: a MODELS table
{name: builder}, or a single make() for the model named like the file. With
no --outdir the files are written next to this script, which is where the
torch_models/ scripts look for them.

    python3 generate_models.py --list                 model names, one per line
    python3 generate_models.py [--outdir DIR] [NAME...]
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR_DIR = os.path.join(HERE, "generators")


def load_models():
    """{model name: builder} collected from every generators/<file>.py."""
    if GENERATOR_DIR not in sys.path:
        sys.path.insert(0, GENERATOR_DIR)
    models = {}
    for f in sorted(os.listdir(GENERATOR_DIR)):
        if f.endswith(".py") and not f.startswith("_"):
            mod = importlib.import_module(f[:-3])
            models.update(getattr(mod, "MODELS", None) or {f[:-3]: mod.make})
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the model names and exit")
    parser.add_argument("--outdir", default=HERE, help="output directory (default: %(default)s)")
    parser.add_argument("models", nargs="*", help="models to generate (default: all)")
    args = parser.parse_args()

    models = load_models()
    if args.list:
        print("\n".join(sorted(models)))
        return 0

    unknown = [n for n in args.models if n not in models]
    if unknown:
        print(f"error: unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    import onnx

    os.makedirs(args.outdir, exist_ok=True)
    names = args.models or sorted(models)
    for name in names:
        onnx.save(models[name](), os.path.join(args.outdir, name + ".onnx"))
    print(f"generated {len(names)} models in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
