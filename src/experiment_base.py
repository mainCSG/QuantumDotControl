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
from paths import DATA

logger = TunerLog('Exp. Base')

@dataclass
class SweepParam:

    '''
    Description
    -----------
    A dataclass that defines a single parameter to sweep over.
    '''

    parameter: str
    start: float
    end: float

@dataclass
class SweepLayer:

    '''
    Description
    -----------
    A dataclass that defines a single layer of a sweep. A layer is defined as a set of parameters to sweep over,
    the number of points to sweep over, and the time to wait after setting the parameters before measuring.
    '''

    targets: list[SweepParam]
    num_points: int
    measurement_time: float

    def __post_init__(self):
        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")

@dataclass
class VirtualSweepParam:
    """
    Defines one virtual parameter and its mapping to physical parameters.

    The physical parameters are calculated as:

        physical = baseline + offset + sum(coefficient * virtual_value)

    Example
    -------
    VirtualSweepParam(
        parameter="virtual.detuning",
        start=-0.1,
        end=0.1,
        coefficients={
            "dac.left_gate": 1.0,
            "dac.right_gate": -1.0,
        },
    )
    """

    parameter: str
    start: float
    end: float
    coefficients: dict[str, float]

    def __post_init__(self):
        if not self.parameter:
            raise ValueError("Virtual parameter name cannot be empty")

        if not self.coefficients:
            raise ValueError(
                f"Virtual parameter {self.parameter!r} must control at least "
                "one physical parameter"
            )

        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError(
                f"Virtual parameter {self.parameter!r} must have finite "
                "start and end values"
            )

        for physical_parameter, coefficient in self.coefficients.items():
            if "." not in physical_parameter:
                raise ValueError(
                    f"Physical parameter {physical_parameter!r} must use "
                    "'instrument.parameter' notation"
                )

            if not np.isfinite(coefficient):
                raise ValueError(
                    f"Coefficient for {physical_parameter!r} must be finite"
                )

@dataclass
class VirtualSweepLayer:
    """
    Defines one layer of a virtual-gate sweep.

    All targets in a layer are swept together. Separate layers produce a
    nested sweep.
    """

    targets: list[VirtualSweepParam]
    num_points: int
    measurement_time: float

    def __post_init__(self):
        if not self.targets:
            raise ValueError(
                "A VirtualSweepLayer must contain at least one target"
            )

        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")

        if self.measurement_time < 0:
            raise ValueError("measurement_time must be >= 0")

        names = [target.parameter for target in self.targets]

        if len(names) != len(set(names)):
            raise ValueError(
                "A virtual parameter cannot appear more than once in a layer"
            )

class VirtualSweep:
    """
    Runs nested sweeps in virtual-gate coordinates.

    Parameters
    ----------
    layers : list[VirtualSweepLayer]
        The virtual sweep layers.
    measure : callable
        Called as:

            data, keys = measure(instr_handler, physical_setpoints)

        This matches the measurement interface used by Sweep.
    physical_offsets : dict[str, float], optional
        Constant physical-gate offsets. The complete transformation is:

            physical[g] = baseline[g] + physical_offsets[g]
                          + sum(C[g, v] * virtual[v])

        The baseline is supplied through current_setpoints in run().
    """

    def __init__(self, layers, measure, physical_offsets=None):
        self.layers = layers
        self.measure = measure
        self.physical_offsets = dict(physical_offsets or {})

        self.results = []

        self.all_virtual_params = [
            target.parameter
            for layer in self.layers
            for target in layer.targets
        ]

        if len(self.all_virtual_params) != len(
            set(self.all_virtual_params)
        ):
            raise ValueError(
                "Each virtual parameter may occur in only one sweep layer"
            )

        self._virtual_params = {
            target.parameter: target
            for layer in self.layers
            for target in layer.targets
        }

        # Preserve declaration order while removing duplicates.
        self.all_physical_params = []
        seen_physical_params = set()

        for layer in self.layers:
            for target in layer.targets:
                for physical_parameter in target.coefficients:
                    if physical_parameter not in seen_physical_params:
                        self.all_physical_params.append(
                            physical_parameter
                        )
                        seen_physical_params.add(physical_parameter)

        for physical_parameter in self.physical_offsets:
            if "." not in physical_parameter:
                raise ValueError(
                    f"Physical offset {physical_parameter!r} must use "
                    "'instrument.parameter' notation"
                )

            if not np.isfinite(
                self.physical_offsets[physical_parameter]
            ):
                raise ValueError(
                    f"Physical offset for {physical_parameter!r} "
                    "must be finite"
                )

            if physical_parameter not in seen_physical_params:
                self.all_physical_params.append(physical_parameter)
                seen_physical_params.add(physical_parameter)

        # Compatibility with code that expects Sweep.all_params.
        self.all_params = self.all_virtual_params

        self._csv_file = None
        self._csv_writer = None
        self._header = None
        self._measurement_keys = None

    def _open_csv(self, filename):
        """
        Open the CSV file.

        The header is written after the first measurement because the
        measurement keys are provided by the measurement function.
        """

        self.filename = filename
        self.csv_path = os.path.join(DATA, self.filename)

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        self._header = None
        self._measurement_keys = None

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()

        self._csv_file = None
        self._csv_writer = None

    @staticmethod
    def _wait(duration, abort_event):
        """
        Wait for duration seconds while remaining responsive to aborts.
        """

        start_time = time.monotonic()

        while time.monotonic() - start_time < duration:
            if abort_event.is_set():
                raise RuntimeError("Virtual sweep aborted")

            time.sleep(0.001)

    def _initial_virtual_setpoints(
        self,
        current_virtual_setpoints=None,
    ):
        """
        Initialize every virtual coordinate to its sweep start value.
        """

        virtual_setpoints = {
            parameter: target.start
            for parameter, target in self._virtual_params.items()
        }

        if current_virtual_setpoints:
            unknown_parameters = (
                set(current_virtual_setpoints)
                - set(self.all_virtual_params)
            )

            if unknown_parameters:
                raise ValueError(
                    "Unknown virtual parameters: "
                    f"{sorted(unknown_parameters)}"
                )

            virtual_setpoints.update(current_virtual_setpoints)

        return virtual_setpoints

    def virtual_to_physical(
        self,
        virtual_setpoints,
        current_setpoints=None,
    ):
        """
        Convert virtual coordinates into physical instrument setpoints.

        Parameters
        ----------
        virtual_setpoints : dict[str, float]
            Current virtual-gate values.
        current_setpoints : dict[str, float], optional
            Baseline physical values.

        Returns
        -------
        dict[str, float]
            Values for every physical parameter controlled by this sweep.
        """

        baseline = current_setpoints or {}

        physical_setpoints = {
            physical_parameter: (
                float(baseline.get(physical_parameter, 0.0))
                + float(
                    self.physical_offsets.get(
                        physical_parameter,
                        0.0,
                    )
                )
            )
            for physical_parameter in self.all_physical_params
        }

        for virtual_parameter, virtual_value in (
            virtual_setpoints.items()
        ):
            target = self._virtual_params.get(virtual_parameter)

            if target is None:
                raise ValueError(
                    f"Unknown virtual parameter: {virtual_parameter!r}"
                )

            for physical_parameter, coefficient in (
                target.coefficients.items()
            ):
                physical_setpoints[physical_parameter] += (
                    float(coefficient) * float(virtual_value)
                )

        return physical_setpoints

    def _apply_virtual_setpoints(
        self,
        instr_handler,
        abort_event,
        virtual_setpoints,
        current_setpoints,
    ):
        """
        Transform virtual coordinates and apply the resulting physical
        parameters.
        """

        if abort_event.is_set():
            raise RuntimeError("Virtual sweep aborted")

        physical_setpoints = self.virtual_to_physical(
            virtual_setpoints,
            current_setpoints,
        )

        # Group parameters by instrument. For example:
        #
        # {
        #     "dac": {
        #         "left_gate": 0.1,
        #         "right_gate": -0.1,
        #     }
        # }
        grouped_parameters = {}

        for full_parameter, value in physical_setpoints.items():
            instrument, parameter = full_parameter.split(".", 1)

            grouped_parameters.setdefault(instrument, {})
            grouped_parameters[instrument][parameter] = value

        for instrument, parameters in grouped_parameters.items():
            if abort_event.is_set():
                raise RuntimeError("Virtual sweep aborted")

            for parameter, value in parameters.items():
                logger.info(
                    f"[VIRTUAL SWEEP] "
                    f"{instrument}.{parameter} -> {value}"
                )

            instr_handler.set_parameter(
                instrument,
                parameters,
                wait=True,
            )

        return physical_setpoints

    @staticmethod
    def _csv_value(value):
        if value is None:
            return ""

        try:
            return float(value)
        except (TypeError, ValueError):
            return ""

    def _write_result(
        self,
        virtual_setpoints,
        physical_setpoints,
        data,
        keys,
    ):
        """
        Write one measurement to the CSV file.

        Measurement columns are determined from the first measurement.
        """

        if self._header is None:
            if isinstance(data, dict):
                if keys is None:
                    self._measurement_keys = list(data.keys())
                else:
                    self._measurement_keys = list(keys)
            else:
                supplied_keys = list(keys or [])
                self._measurement_keys = (
                    supplied_keys
                    if supplied_keys
                    else ["measurement"]
                )

            self._header = (
                list(self.all_virtual_params)
                + list(self.all_physical_params)
                + [
                    str(key)
                    for key in self._measurement_keys
                ]
            )

            self._csv_writer.writerow(self._header)

        row = [
            self._csv_value(virtual_setpoints.get(parameter))
            for parameter in self.all_virtual_params
        ]

        row.extend(
            self._csv_value(physical_setpoints.get(parameter))
            for parameter in self.all_physical_params
        )

        if isinstance(data, dict):
            row.extend(
                self._csv_value(data.get(key))
                for key in self._measurement_keys
            )
        else:
            row.append(self._csv_value(data))

            # Keep the row length consistent if more than one scalar key
            # was supplied.
            row.extend(
                ""
                for _ in range(
                    max(0, len(self._measurement_keys) - 1)
                )
            )

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def run(
        self,
        instr_handler,
        abort_event,
        filename,
        current_setpoints=None,
        current_virtual_setpoints=None,
    ):
        """
        Run the complete virtual sweep.

        Parameters
        ----------
        instr_handler : instrument handler
            The instrument handler used to set physical parameters.
        abort_event : Event
            Event used to abort the sweep.
        filename : str
            CSV filename.
        current_setpoints : dict[str, float], optional
            Baseline physical setpoints. Controlled physical gates are
            calculated relative to these values.
        current_virtual_setpoints : dict[str, float], optional
            Initial virtual values. Values not provided here default to
            their VirtualSweepParam.start values.

        Returns
        -------
        list[dict]
            The measurements collected during the sweep.
        """

        physical_baseline = dict(current_setpoints or {})

        virtual_setpoints = self._initial_virtual_setpoints(
            current_virtual_setpoints
        )

        self.results = []

        try:
            self._open_csv(filename)

            self._run_layer(
                idx=0,
                instr_handler=instr_handler,
                abort_event=abort_event,
                virtual_setpoints=virtual_setpoints,
                physical_baseline=physical_baseline,
            )

        finally:
            self._close_csv()
            logger.info("Virtual sweep data recorded!")

        return self.results

    def _run_layer(
        self,
        idx,
        instr_handler,
        abort_event,
        virtual_setpoints,
        physical_baseline,
    ):
        """
        Recursively execute one virtual sweep layer.
        """

        if abort_event.is_set():
            raise RuntimeError("Virtual sweep aborted")

        # Base case: every layer has been configured, so measure.
        if idx == len(self.layers):
            physical_setpoints = self.virtual_to_physical(
                virtual_setpoints,
                physical_baseline,
            )

            # Preserve unrelated physical setpoints in the dictionary
            # received by the measurement function.
            measurement_setpoints = physical_baseline.copy()
            measurement_setpoints.update(physical_setpoints)

            data, keys = self.measure(
                instr_handler,
                measurement_setpoints.copy(),
            )

            self.results.append({
                "virtual_setpoints": virtual_setpoints.copy(),
                "physical_setpoints": measurement_setpoints.copy(),
                "data": data,
            })

            self._write_result(
                virtual_setpoints=virtual_setpoints,
                physical_setpoints=physical_setpoints,
                data=data,
                keys=keys,
            )

            return

        layer = self.layers[idx]

        values_per_param = [
            np.linspace(
                target.start,
                target.end,
                layer.num_points,
            )
            for target in layer.targets
        ]

        for point_index in range(layer.num_points):
            if abort_event.is_set():
                raise RuntimeError("Virtual sweep aborted")

            next_virtual_setpoints = virtual_setpoints.copy()

            # Targets in the same layer advance together.
            for target, values in zip(
                layer.targets,
                values_per_param,
            ):
                value = float(values[point_index])

                logger.info(
                    f"[VIRTUAL SWEEP] "
                    f"{target.parameter} -> {value}"
                )

                next_virtual_setpoints[target.parameter] = value

            self._apply_virtual_setpoints(
                instr_handler=instr_handler,
                abort_event=abort_event,
                virtual_setpoints=next_virtual_setpoints,
                current_setpoints=physical_baseline,
            )

            self._wait(
                layer.measurement_time,
                abort_event,
            )

            self._run_layer(
                idx=idx + 1,
                instr_handler=instr_handler,
                abort_event=abort_event,
                virtual_setpoints=next_virtual_setpoints,
                physical_baseline=physical_baseline,
            )

class Sweep:

    def __init__(self, layers, measure):

        '''
        Description
        -----------
        A class that defines a sweep. A sweep is defined as a set of layers to sweep over, and a measurement function

        Parameters
        ----------
        layers : list[SweepLayer]
            A list of SweepLayer objects that define the layers of the sweep.
        measure : callable
            A function that takes in the instrument handler and the current setpoints, and returns the measurement
        '''
        
        self.layers = layers
        self.measure = measure
        self.results = []

        self.all_params = [
            p.parameter
            for layer in self.layers
            for p in layer.targets
        ]

        self._csv_file = None
        self._csv_writer = None

    def _open_csv(self, filename):

        '''
        Description
        -----------
        A method that opens a csv file to write the results of the sweep to. The csv file is created in the directory defined in the __init__ method.

        Parameters
        ----------
        filename : str
            The name of the csv file to create. If the file already exists, it will be overwritten.
        '''

        keys = [
            'agilent_left.volt',
            'agilent_right.volt'
        ]

        ap = list(self.all_params)
        
        self._header = ap + keys

        self.filename = filename

        self.csv_path = os.path.join(DATA, self.filename)

        self._csv_file = open(self.csv_path, "w", newline="")

        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._header)

    def _close_csv(self):

        '''
        Description
        -----------
        A method that closes the csv file.
        '''

        if self._csv_file is not None:
            self._csv_file.close()

    def set_voltage_configuration(self, instr_handler, abort_event, current_setpoints = {}):

        '''
        Description
        -----------
        A method that sets the voltage configuration of the sweep without measuring.
        This is useful for setting the voltages in between experiments, as well as for resetting the voltages after a sweep has been completed.

        Parameters
        ----------
        instr_handler : instance of the instrument handler
            The instrument_handler instance that instantiates when the gui is run.
        abort_event : Event Object
            The abort event that can be dynamically updated to abort any experiment job if needed.
        current_setpoints : dict, optional
            The current values set on the instrument. Defaults to empty.
        '''

        try:
            self.set_voltage_layer(
                0,
                instr_handler,
                abort_event,
                current_setpoints=current_setpoints
            )

        finally:
            print()

    def set_voltage_layer(self, idx, instr_handler, abort_event, current_setpoints):

        """
        Description
        -----------
        A method that sets a particular voltage configuration without measurement. The intended use of this method
        is to set voltage configurations in between experiments, as well as allow for smooth resetting of voltages
        once a layer has been completely swept. THIS METHOD DOES NOT RECURSE.

        Parameters
        ----------
        name : idx
            The layer index for the sweep. In set_voltage_configuration, this is always set to 0 initially.
        instr_handler : instance of the instrument handler
            The instrument_handler instance that instantiates when the gui is run.
        abort_event : Event Object
            The abort event that can be dynamically updated to abort any experiment job if needed.
        current_setpoints : dict, optional
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

    def run(self, instr_handler, abort_event, filename, current_setpoints = {}):    

        '''
        Description
        -----------
        A method that runs the sweep and records the results to a csv file.

        Parameters
        ----------
        instr_handler : instance of the instrument handler
            The instrument_handler instance that instantiates when the gui is run.
        abort_event : Event Object
            The abort event that can be dynamically updated to abort any experiment job if needed.
        filename : str
            The name of the csv file to create. If the file already exists, it will be overwritten.
        current_setpoints : dict, optional
            The current values set on the instrument. Defaults to empty.
        '''    

        try:

            self._open_csv(filename = filename)

            self._run_layer(
                0,
                instr_handler,
                abort_event,
                current_setpoints=current_setpoints
            )

        finally:
            self._close_csv()

            logger.info("Data Recorded!")

    def _run_layer(self, idx, instr_handler, abort_event, current_setpoints):

        """
        Description
        -----------
        A method that sets a particular voltage configuration without measurement. The intended use of this method
        is to set voltage configurations in between experiments, as well as allow for smooth resetting of voltages
        once a layer has been completely swept. THIS METHOD DOES NOT RECURSE.

        Parameters
        ----------
        name : idx
            The layer index for the sweep. In set_voltage_configuration, this is always set to 0 initially.
        instr_handler : instance of the instrument handler
            The instrument_handler instance that instantiates when the gui is run.
        abort_event : Event Object
            The abort event that can be dynamically updated to abort any experiment job if needed.
        current_setpoints : dict, optional
            The current values set on the instrument. Defaults to empty.

        Return
        ------
        None
        """ 

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

                reset_layer = self.layers[idx - 1]

                logger.info(f"{reset_layer}")

                reset_targets = reset_layer.targets[0]

                reset_start = reset_targets.end

                reset_end = reset_targets.start

                logger.info(f"start: {reset_start}, end: {reset_end}")

                reset_layer = self._build_reset_layers(
                    idx,
                    reset_start,
                    reset_end,
                    num_points=50
                )

                # Save original layers
                original_layers = self.layers

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
        Description
        -----------
        Build a temporary list of layers that sweep from end_setpoints back to start_setpoints
        using the same parameter structure as self.layers[idx:].

        Parameters
        ----------
        idx : int
            The index of the layer to reset.
        start_setpoints : dict
            The starting setpoints for the reset sweep.
        end_setpoints : dict
            The ending setpoints for the reset sweep.
        num_points : int, optional
            The number of points to sweep. Defaults to 100.

        Returns
        -------
        reset_layer : list[SweepLayer]
            A list of layers for the reset sweep.
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

        logger.info(
            f"RESET: {new_p.parameter} "
            f"{new_p.start} -> {new_p.end}"
        )

        return reset_layer