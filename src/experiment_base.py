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
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from instrument_handler import instrument_handler
from tunerlog import TunerLog

logger = TunerLog("Expt. Base")

@dataclass
class SweepLayers:
    
    '''
    One axis of a multi-dimensional sweep.

    Parameters resolved through instrument_handler
    -----------------------------------------------
    instrument : str
        The instrument name as registered in the station / instrument_handler
        (e.g. "dac", "vna").
    parameter : str
        The QCoDeS parameter name on that instrument (e.g. "ch1_v", "frequency").
        Supports dotted sub-parameters in the same format as instrument_handler
        (e.g. "ch1.voltage").
    setpoints : Sequence
        Ordered values to step through on this axis.
    delay : float
        Seconds to wait after set_parameter completes before continuing.
        Uses interruptible sleep so aborts remain responsive.
    before_sweep : Callable | None
        Called once before this axis begins.
        Signature: (layer: SweepLayer) -> None
    after_sweep : Callable | None
        Called once after this axis finishes (runs even on abort via finally).
        Signature: (layer: SweepLayer) -> None
    after_step : Callable | None
        Called after each step on this layer, before descending into inner layers.
        Signature: (layer: SweepLayer, value: Any) -> None
    name : str
        Human-readable label for logging. Falls back to "instrument.parameter".
    '''

    instrument: str
    parameter: str
    layers: list[list]
    before_sweep: Callable[[SweepLayers], None] | None = None
    after_sweep:  Callable[[SweepLayers], None] | None = None
    after_step:   Callable[[SweepLayers, Any], None] | None = None
    name: str = ''

    def __post_init__(self) -> None:
        if not self.instrument:
            raise ValueError("SweepLayer.instrument must not be empty.")
        if not self.parameter:
            raise ValueError("SweepLayer.parameter must not be empty.")
        if len(self.setpoints) == 0:
            raise ValueError("SweepLayer.setpoints must not be empty.")

    @property
    def label(self) -> str:
        return self.name or f"{self.instrument}.{self.parameter}"

class Sweep:

    def __init__(self):

        pass
