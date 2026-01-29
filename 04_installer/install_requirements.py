#!/usr/bin/env python3
"""Install project dependencies from requirements.txt."""

import subprocess
import sys
from pathlib import Path


def main():
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)

    print(f"Installing requirements from {requirements_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
