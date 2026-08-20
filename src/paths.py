"""
This modules defines the path used in the project.
"""
from pathlib import Path
import os
import datetime

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
# device configuration path
DEVICE_CONFIG = CONFIG_FOLDER / 'device configs'
# station configuration path 
STATION_CONFIG = CONFIG_FOLDER / 'station configs'
# fix protocols folder path
PROTOCOLS = ROOT / "Protocols"
# Fix Current Run Path
PROTOCOL_FOLDER = PROTOCOLS / f"Protocol_Run_{datetime.datetime.now().strftime('%m-%d-%Y')}"
# Data folder path
DATA = PROTOCOL_FOLDER / "Data"

