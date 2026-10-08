"""Command-line interface: `kna`."""

from __future__ import annotations

import argparse

from kazner_agents import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kna", description="Kazakh NER annotation agents")
    parser.add_argument("--version", action="version", version=f"kna {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
