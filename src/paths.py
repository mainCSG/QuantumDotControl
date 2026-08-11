"""
This modules defines the path used in the project.
"""
from pathlib import Path
import os

# project structure
"""


QuantumDotControl/
├── configs/
├── src/
    └── main.py
    ...
├── Protocols/  # each protocol/ run has its own folder with a timestamp
│   └── current_protocol/
        └── Data/
            └── <data_subfolder>/
"""

# fix root path to be the root of the project
ROOT = Path(__file__).resolve().parent.parent

# fix config folder path
CONFIG_FOLDER = ROOT / "configs"
# fix protocols folder path
PROTOCOLS = ROOT / "Protocols"