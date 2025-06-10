#!/usr/bin/env python3
"""Basic test to verify Python runs in the repository."""

import sys
import os

def test_python_execution():
    """Test that Python can execute basic operations."""
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    assert sys.version_info >= (3, 6), "Python 3.6+ required"
    assert os.path.exists("mla"), "mla directory should exist"
    assert 2 + 2 == 4, "Basic arithmetic should work"
    print("✓ Python execution test passed")

if __name__ == "__main__":
    test_python_execution()