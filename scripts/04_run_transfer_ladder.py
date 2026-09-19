"""Full-window transfer-ladder entry point; deliberately unavailable in pilot mode."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-window-approved", action="store_true")
    args = parser.parse_args()
    if not args.full_window_approved:
        raise SystemExit("M3/transfer ladder is outside the approved ten-day pilot scope")
    raise SystemExit("Full-window data for ETH/SOL/DOGE is not present; no transfer run was attempted")


if __name__ == "__main__":
    main()
