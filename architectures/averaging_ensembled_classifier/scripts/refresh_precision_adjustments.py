#!/usr/bin/env python3
"""Refresh the PRECISION_ADJUSTMENTS table from Oracle D derivation (ADR-033 Phase 2).

Runs ``derive_all_adjustments()`` across the standard problem × precision
matrix and emits a Python dict literal that replaces the hand-tuned
``PRECISION_ADJUSTMENTS`` in ``tests/convergence/precision_adjustments.py``.

Usage::

    cd architectures/averaging_ensembled_classifier
    python -m scripts.refresh_precision_adjustments          # print to stdout
    python -m scripts.refresh_precision_adjustments --apply  # rewrite file in-place

CI staleness gate (Phase 3)::

    python -m scripts.refresh_precision_adjustments --check  # exit 1 if stale
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.convergence.precision_adjustments import PrecisionAdjustment

ADJUSTMENTS_FILE = (
    Path(__file__).resolve().parent.parent
    / "tests" / "convergence" / "precision_adjustments.py"
)


def _format_table(table: dict[str, PrecisionAdjustment]) -> str:
    """Format the adjustment table as a Python dict literal."""
    lines = ["PRECISION_ADJUSTMENTS: dict[str, PrecisionAdjustment] = {"]
    for key, adj in sorted(table.items()):
        lines.append(
            f'    "{key}": PrecisionAdjustment('
            f"accuracy_delta={adj.accuracy_delta:.6f}, "
            f"loss_delta={adj.loss_delta:.6f}, "
            f"epoch_multiplier={adj.epoch_multiplier:.4f}),"
        )
    lines.append("}")
    return "\n".join(lines)


def _parse_committed_table(source: str) -> dict[str, tuple[float, float, float]]:
    """Extract the committed PRECISION_ADJUSTMENTS values from source text."""
    pattern = re.compile(
        r'"(\w+)":\s*PrecisionAdjustment\('
        r'accuracy_delta=([\d.e+-]+),\s*'
        r'loss_delta=([\d.e+-]+),\s*'
        r'epoch_multiplier=([\d.e+-]+)\)',
    )
    result: dict[str, tuple[float, float, float]] = {}
    for m in pattern.finditer(source):
        result[m.group(1)] = (
            float(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh PRECISION_ADJUSTMENTS from Oracle D derivation.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite precision_adjustments.py in-place.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if committed table diverges from fresh derivation.",
    )
    args = parser.parse_args()

    # Import here so the script's --help works without dependencies
    from tests.convergence.precision_adjustments import derive_all_adjustments

    print("Deriving precision adjustments from Oracle D...", file=sys.stderr)
    table = derive_all_adjustments()

    new_block = _format_table(table)

    if args.check:
        source = ADJUSTMENTS_FILE.read_text()
        committed = _parse_committed_table(source)

        stale = False
        for key, adj in table.items():
            if key not in committed:
                print(f"STALE: key {key!r} missing from committed table", file=sys.stderr)
                stale = True
                continue
            c_acc, c_loss, c_ep = committed[key]
            if (
                abs(adj.accuracy_delta - c_acc) > 0.005
                or abs(adj.loss_delta - c_loss) > 0.01
                or abs(adj.epoch_multiplier - c_ep) > 0.1
            ):
                print(
                    f"STALE: {key!r} committed=({c_acc:.6f}, {c_loss:.6f}, {c_ep:.4f}) "
                    f"derived=({adj.accuracy_delta:.6f}, {adj.loss_delta:.6f}, {adj.epoch_multiplier:.4f})",
                    file=sys.stderr,
                )
                stale = True

        if stale:
            print(
                "Committed PRECISION_ADJUSTMENTS table is stale. "
                "Run: python -m scripts.refresh_precision_adjustments --apply",
                file=sys.stderr,
            )
            return 1
        print("PRECISION_ADJUSTMENTS table is up to date.", file=sys.stderr)
        return 0

    if args.apply:
        source = ADJUSTMENTS_FILE.read_text()

        # Replace the PRECISION_ADJUSTMENTS block
        pattern = re.compile(
            r"^PRECISION_ADJUSTMENTS:.*?^\}",
            re.MULTILINE | re.DOTALL,
        )
        new_source, count = pattern.subn(new_block, source)
        if count != 1:
            print(
                f"ERROR: expected 1 PRECISION_ADJUSTMENTS block, found {count}",
                file=sys.stderr,
            )
            return 1

        ADJUSTMENTS_FILE.write_text(new_source)
        print(f"Updated {ADJUSTMENTS_FILE}", file=sys.stderr)
        return 0

    # Default: print to stdout
    print(new_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
