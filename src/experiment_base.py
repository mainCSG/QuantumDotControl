'''
File: experiment_base.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Defines SweepLayer objects and the Sweep class to handle all the running of all sweeps used in the Autotuner. 

'''

# Imports

from __future__ import annotations

import csv
import os
from datetime import datetime

import time
import threading
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from qcodes.station import Station
from instrument_handler import TunerFuture
from experiment_handler import ExperimentThread, experiment_job
from tunerlog import TunerLog

logger = TunerLog('Exp. Base')

@dataclass
class SweepParam:
    parameter: str
    start: float
    end: float

@dataclass
class SweepLayer:
    targets: list[SweepParam]
    num_points: int
    measurement_time: float

    def __post_init__(self):
        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")

class Sweep:

    def __init__(self, layers, measure):
        
        self.layers = layers
        self.measure = measure
        self.results = []

        self.all_params = [
            p.parameter
            for layer in self.layers
            for p in layer.targets
        ]

        self.filename = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        self.csv_path = os.path.join(desktop, self.filename)

        self._csv_file = None
        self._csv_writer = None

    def _open_csv(self):
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        header = self.all_params + ["data"]
        self._csv_writer.writerow(header)

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()

    def set_voltage_configuration(self, instr_handler, abort_event):

        try:
            self.set_voltage_layer(
                0,
                instr_handler,
                abort_event,
                current_setpoints={}
            )

        finally:
            print("Voltage Configuration Set!")

    def set_voltage_layer(self, idx, instr_handler, abort_event, current_setpoints):

        if idx == len(self.layers):
            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")
            return

        layer = self.layers[idx]

        values_per_param = [
            np.linspace(p.start, p.end, layer.num_points)
            for p in layer.targets
        ]

        for i in range(layer.num_points):

            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")

            step_values = {}

            for p, values in zip(layer.targets, values_per_param):
                val = float(values[i])

                instr, param = p.parameter.split('.', 1)

                logger.info(f"[SWEEP] {p.parameter} -> {val}")

                instr_handler.set_parameter(
                    instr,
                    {param: val},
                    wait=True
                )

                step_values[p.parameter] = val

            # Wait

            t0 = time.monotonic()
            while time.monotonic() - t0 < layer.measurement_time:
                if abort_event.is_set():
                    raise RuntimeError("Sweep aborted")
                time.sleep(0.001)

            new_setpoints = current_setpoints.copy()
            new_setpoints.update(step_values)

            # Recurse

            self.set_voltage_layer(
                idx + 1,
                instr_handler,
                abort_event,
                new_setpoints
            )

    def run(self, instr_handler, abort_event):

        try:
            self._open_csv()

            self._run_layer(
                0,
                instr_handler,
                abort_event,
                current_setpoints={}
            )

        finally:
            self._close_csv()

    def _run_layer(self, idx, instr_handler, abort_event, current_setpoints):

        if idx == len(self.layers):
            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")

            data = self.measure(instr_handler, current_setpoints.copy())  # ✅ fixed
            
            self.results.append({
                "setpoints": current_setpoints.copy(),
                "data": data
            })

            # CSV write

            row = [current_setpoints.get(p, None) for p in self.all_params] + [data]
            self._csv_writer.writerow(row)
            self._csv_file.flush()

            return

        layer = self.layers[idx]

        values_per_param = [
            np.linspace(p.start, p.end, layer.num_points)
            for p in layer.targets
        ]

        for i in range(layer.num_points):

            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")

            step_values = {}

            for p, values in zip(layer.targets, values_per_param):
                val = float(values[i])

                instr, param = p.parameter.split('.', 1)

                logger.info(f"[SWEEP] {p.parameter} -> {val}")  # ✅ fixed

                instr_handler.set_parameter(
                    instr,
                    {param: val},
                    wait=True
                )

                step_values[p.parameter] = val

            # Wait

            t0 = time.monotonic()
            while time.monotonic() - t0 < layer.measurement_time:
                if abort_event.is_set():
                    raise RuntimeError("Sweep aborted")
                time.sleep(0.001)

            logger.info("[SWEEP] Measuring...")

            new_setpoints = current_setpoints.copy()
            new_setpoints.update(step_values)

            # Recurse

            self._run_layer(
                idx + 1,
                instr_handler,
                abort_event,
                new_setpoints
            )