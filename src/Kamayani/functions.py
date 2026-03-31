from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
import time
import os
from datetime import datetime
from typing import List, Tuple
from typing import Sequence
from typing import Any
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ground_device_sim(sim, slot: int, step: float = 0.1, delay: float = 0.5):
    V0 = sim.get_voltage(slot)
    if abs(V0) < 1e-12:
        return
    
    direction = -np.sign(V0)
    setpoints = np.arange(V0, 0, direction * step)

    for V in setpoints:
        sim.set_voltage(slot, float(V))
        time.sleep(delay)

    sim.set_voltage(slot, 0.0)
    time.sleep(delay)

def set_voltage_sim(sim, slot: int, target: float, step: float = 0.1, delay: float = 0.5):
    current = sim.get_voltage(slot)
    if abs(current - target) < 1e-12:
        return
    
    direction = -np.sign(current - target)
    setpoints = np.arange(current, target, direction * step)

    for V in setpoints:
        sim.set_voltage(slot, float(V))
        time.sleep(delay)

    sim.set_voltage(slot, target)
    time.sleep(delay)

    print(f'The voltage of the channel {slot} of sim900 is set to {target: .6f} V.')