#!/usr/bin/env python
"""
Top-level Dojo command-line interface.

This lets you run commands like:
    python dojo.py train
    python dojo.py eval
or create an alias:
    alias dojo="python dojo.py"
"""

import sys

from dojo_cli.dojo_train import cli


def main() -> None:
    """CLI entry point for Dojo."""
    # Delegate to Click CLI
    cli()


if __name__ == "__main__":
    sys.exit(main())
