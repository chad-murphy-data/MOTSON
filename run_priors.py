#!/usr/bin/env python3
"""Wrapper to run 5-year prior estimation with API key."""
import os
import subprocess
import sys

os.environ['FOOTBALL_DATA_API_KEY'] = 'dafdc5ed7f844360912c50ba4d87644a'

result = subprocess.run(
    [sys.executable, 'scripts/estimate_5year_priors.py', '--seasons', '5', '--n-warmup', '500', '--n-samples', '1000'],
    env=os.environ,
    cwd=os.path.dirname(os.path.abspath(__file__))
)
sys.exit(result.returncode)
