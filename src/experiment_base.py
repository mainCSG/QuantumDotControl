'''
File: experiment_base.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Defines SweepLayer, AbstractSweep, and do_sweep_job for use with ExperimentThread.

All hardware I/O is routed through the instrument_handler using set_parameter and
get_parameter — the sweep never touches QCoDeS instruments directly.

A SweepLayer describes one axis: which instrument parameter to drive and over which
setpoints. An AbstractSweep composes layers outermost → innermost, and calls a
user-supplied measurement callback at every innermost point.

do_sweep_job matches the ExperimentThread calling convention (f(*args, abort_event))
and is registered like:

    thread.add_job(do_sweep_job, (sweep, instr_handler), priority=1)

Measurement callback signature
--------------------------------
    def my_measure(instr_handler: instrument_handler,
                   setpoints: tuple) -> Any:
        return instr_handler.get_parameter("vna", "S21")

The return value is stored in AbstractSweep.results as:
    {'setpoints': (v0, v1, ...), 'data': <return value of measure>}
'''

# Imports

from __future__ import annotations

import time
import threading
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from instrument_handler import instrument_handler
from tunerlog import TunerLog

logger = TunerLog("Expt. Base")

@dataclass
class SweepLayer:
    param: str
    start: float
    end: float
    num_points: int
    measurement_time: float

    def __post_init__(self):
        if '.' not in self.param:
            raise ValueError("param must be 'instrument.parameter'")
        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")

    @property
    def instrument(self):
        return self.param.split('.', 1)[0]

    @property
    def parameter(self):
        return self.param.split('.', 1)[1]

class AbstractSweep:

    def __init__(self, layers, measure):
        self.layers = layers
        self.measure = measure
        self.results = []

    def run(self, instr_handler, abort_event):
        self._run_layer(0, instr_handler, abort_event, [])

    def _run_layer(self, idx, instr_handler, abort_event, current_setpoints):

        if idx == len(self.layers):
            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")

            data = self.measure(instr_handler, tuple(current_setpoints))
            self.results.append({
                "setpoints": tuple(current_setpoints),
                "data": data
            })
            return

        layer = self.layers[idx]

        values = np.linspace(layer.start, layer.end, layer.num_points)

        for val in values:

            if abort_event.is_set():
                raise RuntimeError("Sweep aborted")

            logger.info(f"[SWEEP] Step: setting values {val}")

            # Set parameter
            instr_handler.set_parameter(
                layer.instrument,
                {layer.parameter: float(val)},
                wait=True
            )

            # Measurement wait (interruptible)
            t0 = time.monotonic()
            while time.monotonic() - t0 < layer.measurement_time:
                if abort_event.is_set():
                    raise RuntimeError("Sweep aborted")
                time.sleep(0.01)

            logger.info("[SWEEP] Measuring...")



            # recurse
            self._run_layer(
                idx + 1,
                instr_handler,
                abort_event,
                current_setpoints + [float(val)]
            )

    def do_sweep_job(sweep, instr_handler, abort_event):
        sweep.run(instr_handler, abort_event)
        return sweep.results
