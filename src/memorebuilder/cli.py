"""Command-line interface for memorebuilder.

Usage examples
--------------
Encode a character image and print its polynomial equation::

    memorebuilder encode hero.png

Encode and save the identity to a JSON file::

    memorebuilder encode hero.png --label "Hero" --output hero_identity.json

Compare two images (or a saved identity JSON with an image)::

    memorebuilder compare hero.png hero2.png
    memorebuilder compare hero_identity.json hero2.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .encoder import EquationEncoder
from .extractor import FeatureExtractor
from .identity import CharacterIdentity


def _cmd_encode(args: argparse.Namespace) -> int:
    """Handle the ``encode`` sub-command."""
    extractor = FeatureExtractor()
    encoder = EquationEncoder(degree=args.degree)

    try:
        features = extractor.extract(args.image)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error reading image: {exc}", file=sys.stderr)
        return 1

    coefficients = encoder.encode(features)
    equation = encoder.to_equation_string(coefficients)

    print(f"Character equation:\n  {equation}\n")
    print(f"Polynomial degree : {args.degree}")
    print(f"Coefficients ({len(coefficients)}):")
    for i, c in enumerate(coefficients):
        print(f"  a_{len(coefficients) - 1 - i} = {c:.8g}")

    if args.output:
        identity = CharacterIdentity(
            coefficients=coefficients,
            label=args.label or Path(args.image).stem,
            metadata={"source_image": args.image, "degree": args.degree},
        )
        identity.save(args.output)
        print(f"\nIdentity saved to: {args.output}")

    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Handle the ``compare`` sub-command."""

    def _load(path: str) -> CharacterIdentity:
        p = Path(path)
        if p.suffix.lower() == ".json":
            return CharacterIdentity.load(p)
        # Treat as image
        return CharacterIdentity.from_image(path, label=p.stem, degree=args.degree)

    try:
        id_a = _load(args.source)
        id_b = _load(args.target)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        print(f"Error loading input: {exc}", file=sys.stderr)
        return 1

    score = id_a.similarity(id_b)
    matched = id_a.matches(id_b, threshold=args.threshold)

    print(f"Source : {id_a.label or args.source}")
    print(f"Target : {id_b.label or args.target}")
    print(f"Similarity score : {score:.4f}")
    print(f"Identity match   : {'YES ✓' if matched else 'NO ✗'} (threshold={args.threshold})")

    return 0 if matched else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memorebuilder",
        description=(
            "Extract a character's visual identity from an image and encode it "
            "as a polynomial equation for cross-generation identity preservation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- encode -------------------------------------------------------
    enc = sub.add_parser("encode", help="Encode a character image as a polynomial identity")
    enc.add_argument("image", help="Path to the character image")
    enc.add_argument("--label", default="", help="Human-readable name for the character")
    enc.add_argument(
        "--degree",
        type=int,
        default=12,
        help="Polynomial degree (default: 12)",
    )
    enc.add_argument(
        "--output",
        metavar="FILE",
        default="",
        help="Save identity to a JSON file",
    )

    # -- compare ------------------------------------------------------
    cmp = sub.add_parser(
        "compare",
        help="Compare two character identities (images or saved .json files)",
    )
    cmp.add_argument("source", help="First image or identity JSON")
    cmp.add_argument("target", help="Second image or identity JSON")
    cmp.add_argument(
        "--degree",
        type=int,
        default=12,
        help="Polynomial degree when encoding images (default: 12)",
    )
    cmp.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for a positive match (default: 0.9)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {"encode": _cmd_encode, "compare": _cmd_compare}
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":  # pragma: no cover
    main()
