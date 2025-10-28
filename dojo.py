#!/usr/bin/env python3
"""
Top-level Dojo command-line interface.

This lets you run commands like:
    python dojo.py train
    python dojo.py eval
or create an alias:
    alias dojo="python dojo.py"
"""

from dojo_cli.dojo_train import cli

if __name__ == "__main__":
    cli()
