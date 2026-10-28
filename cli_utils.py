"""Dependency-free input validation shared by command line tools."""
import argparse
from pathlib import Path

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")

def read_names(path):
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Class names must be nonempty and unique")
    return names
