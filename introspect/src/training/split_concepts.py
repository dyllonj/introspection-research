"""Utility to generate a deterministic train/holdout concept split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from ..eval_common import load_words, select_target_words
from ..vectors import DEFAULT_WORDS_PATH


def split_concepts(
    words_file: Path,
    *,
    n_holdout: int,
    seed: int,
) -> dict[str, list[str]]:
    words = load_words(words_file)
    targets = list(words.iter_targets())
    rng = random.Random(seed)
    holdout = rng.sample(targets, min(n_holdout, len(targets)))
    train = [w for w in targets if w not in set(holdout)]
    return {"train": train, "holdout": holdout}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words-file", type=Path, default=Path(DEFAULT_WORDS_PATH))
    parser.add_argument("--n-holdout", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional path to save JSON split")
    args = parser.parse_args(argv)

    split = split_concepts(args.words_file, n_holdout=args.n_holdout, seed=args.seed)
    data = json.dumps(split, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data)


if __name__ == "__main__":
    main()
