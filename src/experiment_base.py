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
import numpy as np
from dataclasses import dataclass
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

        keys = [
            'agilent_left.volt',
            'agilent_right.volt'
        ]
        ap = list(self.all_params)
        
        self._header = ap + keys
        
        self._csv_file = open(self.csv_path, "w", newline="")
        
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._header)

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()

    def set_voltage_configuration(self, instr_handler, abort_event, current_setpoints = {}):

        try:
            self.set_voltage_layer(
                0,
                instr_handler,
                abort_event,
                current_setpoints=current_setpoints
            )

        finally:
            print("Voltage Configuration Set!")

    def set_voltage_layer(self, idx, instr_handler, abort_event, current_setpoints):

        """
        A method that sets a particular voltage configuration without measurement. The intended use of this method
        is to set voltage configurations in between experiments, as well as allow for smooth resetting of voltages
        once a layer has been completely swept. THIS METHOD DOES NOT RECURSE.

        Parameters
        ----------
        name: idx
            The layer index for the sweep. In set_voltage_configuration, this is always set to 0 initially.

        instr_handler:  
            The instrument_handler instance that instantiates when the gui is run. 
        
        abort_event:
            The abort event that can be dynamically updated to abort any experiment job if needed.

        current_setpoints:
            The current values set on the instrument. Defaults to empty.

        """

        if idx != 0:
            raise ValueError("Setting a voltage layer should only have one layer!")

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

    def run(self, instr_handler, abort_event, current_setpoints = {}):        

        try:
            self._open_csv()

            self._run_layer(
                0,
                instr_handler,
                abort_event,
                current_setpoints=current_setpoints
            )

        finally:
            self._close_csv()

    def _run_layer(self, idx, instr_handler, abort_event, current_setpoints):

        if idx == len(self.layers):
            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")
            
            data, keys = self.measure(instr_handler, current_setpoints.copy())

            self.results.append({
                "setpoints": current_setpoints.copy(),
                "data": data
            })

            row = [current_setpoints.get(p, None) for p in self.all_params]

            if isinstance(data, dict):
                for k in keys:
                    val = data.get(k, None)
                    row.append(float(val) if val is not None else "")
            else:
                try:
                    row.append(float(data))
                except (TypeError, ValueError):
                    row.append("")
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

                logger.info(f"[SWEEP] {p.parameter} -> {val}")

                instr_handler.set_parameter(
                    instr,
                    {param: val},
                    wait=True
                )

                step_values[p.parameter] = val

                # Wait

                t0 = time.perf_counter()

                while time.perf_counter() - t0 < layer.measurement_time:
                    if abort_event.is_set():
                        raise RuntimeError("Sweep aborted")
                    time.sleep(0.001)

            new_setpoints = current_setpoints.copy()
            new_setpoints.update(step_values)

            # Recurse

            self._run_layer(
                idx + 1,
                instr_handler,
                abort_event,
                new_setpoints
            )

            if idx < len(self.layers) - 1 and i < layer.num_points - 1:

                reset_layer = self._build_reset_layers(
                    idx,
                    p.end,
                    p.start,
                    num_points=50
                )

                print(f"reset_layers: {reset_layer}")

                # Save original layers
                original_layers = self.layers

                print(f"original_layers: {original_layers}")

                try:
                    # Swap in reset layers
                    self.layers = reset_layer

                    # Call your existing function
                    self.set_voltage_layer(
                        0,
                        instr_handler,
                        abort_event,
                        new_setpoints
                    )

                finally:
                    # Restore original layers
                    self.layers = original_layers

    def _build_reset_layers(self, idx, start_setpoints, end_setpoints, num_points=100):
        
        """
        Build a temporary list of layers that sweep from end_setpoints back to start_setpoints
        using the same parameter structure as self.layers[idx:].
        """

        reset_layer = []

        new_targets = []

        for layer in self.layers[idx + 1:]:

            for p in layer.targets:
                param = p.parameter

                v_start = start_setpoints
                v_end = end_setpoints

                if v_start is None or v_end is None:
                    continue

                # Create a shallow copy-like object with reversed sweep
                new_p = type(p)(
                    parameter=p.parameter,
                    start=v_start,
                    end=v_end
                )

                new_targets.append(new_p)

        # Recreate layer
        
        new_layer = type(layer)(
            targets=new_targets,
            num_points=num_points,
            measurement_time=layer.measurement_time
        )

        reset_layer.append(new_layer)

        return reset_layer