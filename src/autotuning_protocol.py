# Import modules

import yaml, datetime, sys, time, os, shutil, json,re
from pathlib import Path

import pandas as pd

import numpy as np

import scipy as sp
from scipy.ndimage import convolve

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from typing import List, Dict

import qcodes as qc
from qcodes.dataset import AbstractSweep, Measurement
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase
import numpy.typing as npt

from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import diamond, rectangle  # noqa

from main import gui
from experiment_base import SweepParam, SweepLayer, Sweep

import sys

from nicegui import ui
from tunerlog import TunerLog

logger = TunerLog('Autotuner')

class Protocol:

    def __init__(self, device_config):
        
        '''
        Initializes the protocol. Reads the device configuration file provided and creates a path from gate name to dac.

        
        '''

        # First, we load in the config file

        logger.info("Loading Device Config file...")

        self._load_config_file(device_config)

        # Now, we create a dictionary to house a map between gate names and dacs

        self.gates_to_dacs = {}

        for i in self.device_gates:

            self.gates_to_dacs[i] = self.device_gates[i]['channel']

    def _load_config_file(self, device_config):
        
        # Read the tuner config information

        self.config = yaml.safe_load(Path(device_config).read_text())

        # Read the config information

        self.charge_carrier = self.config['device']['characteristics']['charge_carrier']
        self.operation_mode = self.config['device']['characteristics']['operation_mode']

        # Set the voltage sign for the gates, based on the charge carrier and mode of the device

        if (self.charge_carrier, self.operation_mode) == ('e', 'acc'):
            self.voltage_sign = +1
        
        if (self.charge_carrier, self.operation_mode) == ('e', 'dep'):
            self.voltage_sign = -1
        
        if (self.charge_carrier, self.operation_mode) == ('h', 'acc'):
            self.voltage_sign = -1
        
        if (self.charge_carrier, self.operation_mode) == ('h', 'dep'):
            self.voltage_sign = +1

        # Get the device gates

        self.device_gates = self.config['device']['gates']
        
        # Re-label all the gates as ohmics, barriers, leads, plungers, accumulation gates and screening gates 

        self.ohmics = []
        self.barriers = []
        self.leads = []
        self.plungers = []
        self.accumulation = []
        self.screening = []
        
        for gate, details in self.device_gates.items():
            
            if details['type'] == 'ohmic':
                self.ohmics.append(gate)
            
            if details['type'] == 'barrier':
                self.barriers.append(gate)
            
            if details['type'] == 'lead':
                self.leads.append(gate)
            
            if details['type'] == 'plunger':
                self.plungers.append(gate)
            
            if details['type'] == 'accumulation':
                self.accumulation.append(gate)

            if details['type'] == 'screening':
                self.screening.append(gate)

        
        self.all_gates = list(self.device_gates.keys())

        # Contraints

        self.abs_max_current_triple_dot = self.config['device']['constraints']['abs_max_current_triple_dot']
        self.abs_max_current_SET_dot = self.config['device']['constraints']['abs_max_current_SET_dot']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.config['device']['constraints']['abs_max_gate_differential']

        # Equipment Setup

        self.voltage_divider_triple_dot = self.config['setup']['voltage_dividers']['voltage_divider_triple_dot']
        self.voltage_divider_SET_dot = self.config['setup']['voltage_dividers']['voltage_divider_SET_dot']
        self.triple_dot_preamp_bias = self.config['setup']['triple_dot_preamp']['preamp_bias']
        self.triple_dot_preamp_sensitivity = self.config['setup']['triple_dot_preamp']['preamp_sensitivity']
        self.SET_dot_preamp_bias = self.config['setup']['SET_dot_preamp']['preamp_bias']
        self.SET_dot_preamp_sensitivity = self.config['setup']['SET_dot_preamp']['preamp_sensitivity']


class Bootstrapping(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

        self.noise_floor = None

        self.ground_device(instr_handler = gui.instrument_handler)

        self.measure_noise_floor()

    def ground_device(self, instr_handler):
        
        # First, we grab all the connected dacs and current values

        dacs_and_vals = {}

        for i in self.gates_to_dacs:
            
            p = self.device_gates[i]['channel']

            instr, param = p.parameter.split('.', 1)

            dacs_and_vals[i] = instr_handler.get_parameter(
                                        instr,
                                        param,
                                        wait=True
                                        )
            
        print(dacs_and_vals)

        # Now, we create the sweep parameters

        targets = []

        for i in dacs_and_vals:

            param = SweepParam(
                parameter = i,
                start = dacs_and_vals[i],
                end = 0.0
            )

            targets.append(param)

        print(targets)

        sweep_layer = SweepLayer(
            targets = targets,
            num_points = 200,
            measurement_time = 0.1
        )

        logger.info("Grounding Device...")

        future = gui.experiment_handler.set_voltage_configuration(sweep = sweep_layer,
                                                                  instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Device Grounded!")

    def measure_noise_floor(self):

        # We collect the readout buffer for 1 minute and average the values to measure the noise floor.

        self.noise_floor = gui.instrument_handler.read_buffer(
            ['agilent_left.volt', 'agilent_right.volt'],
            t_avg = 0,
            t_stop = 60
        )

    def turn_on(self, ohmic_bias, screening_voltage, gate_voltage, num_points):

        # First we grab all the dacs and values to set to our ohmics

        ohmic_targets = []

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Ohmic":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                ohmic_voltage = ohmic_bias / self.voltage_divider_SET_dot

                sparam = SweepParam(
                    parameter = param,
                    start = 0.0,
                    end = ohmic_voltage
                )

                ohmic_targets.append(sparam)            
        
        print(ohmic_targets)

        sweep_layer = SweepLayer(
            targets = ohmic_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        logger.info("Setting Ohmic Bias...")

        future = gui.experiment_handler.set_voltage_configuration(sweep = sweep_layer,
                                                                  instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Ohmic Bias Set!")

        # Next, we set all constant initial voltages before Turn-On. For the Intel device, this is the screening gates

        screening_targets = []

        screening_types = ["Dot Screening", "Sensor Screening", "Central Screening"]

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] in screening_types:

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = 0.0,
                    end = screening_voltage
                )

                screening_targets.append(sparam)            
        
        print(screening_targets)

        sweep_layer = SweepLayer(
            targets = screening_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        logger.info("Setting Screening Gate Voltages...")

        future = gui.experiment_handler.set_voltage_configuration(sweep = sweep_layer,
                                                                  instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Screening Gate Voltages Set!")

        # Now, we create the sweep parameters for all the gates

        gate_targets = []

        excluded_types = ["Dot Ohmic", "Sensor Ohmic", "Dot Screening", "Sensor Screening", "Central Screening"]

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] not in excluded_types:

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = 0,
                    end = gate_voltage
                )

                gate_targets.append(param)

        print(gate_targets)

        sweep_layer = SweepLayer(
            targets = gate_targets,
            num_points = num_points,
            measurement_time = 0.3
        )

        logger.info("Device Turn-On Starting...")

        future = gui.experiment_handler.do_sweep(sweep = sweep_layer,
                                                 instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Device Turn-On Sweep Complete! Confirming Turn-On...")

        # Now, we determine if there was a measured current above the noise floor. If so, we fit our data to the ReLU function





    def pinch_off(self, gate_voltage, final_voltage, num_points):

        # We first construct a loop to pinch-off each individual gate

        excluded_types = ["Dot Ohmic", "Sensor Ohmic", "Dot Screening", "Sensor Screening", "Central Screening"]

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] not in excluded_types:

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = gate_voltage,
                    end = final_voltage
                )

                sweep_layer = SweepLayer(
                    targets = sparam,
                    num_points = num_points,
                    measurement_time = 0.3
                )

                logger.info(f"{self.device_gates[i]['label']} Starting Pinch-Off...")

                future = gui.experiment_handler.do_sweep(sweep = sweep_layer,
                                                        instrument_handler = gui.instrument_handler)
                
                print(future)

                logger.info(f"{self.device_gates[i]['label']} Pinch-Off Complete! Confirming Pinch-Off...")

                """
                Now, we determine if there was a measured current comparable to the noise floor. 
                If so, we fit our data to the sigmoid function

                """



    def SET_current_check(self, instr_handler, minimum_current, maximum_current):

        # First, we need to read the current and check if it is above or below the current values specified.

        self.current_level = gui.instrument_handler.read_buffer(
            ['agilent_left.volt', 'agilent_right.volt'],
            t_avg = 0,
            t_stop = 60
        )

        # Here, we get the current accumulation voltages

        included_types = ['Dot Accumulation', 'Sensor Accumulation']

        self.accumulation_voltages = {}

        for i in self.gates_to_dacs:

            if i in included_types:

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                self.accumulation_voltages[i] = instr_handler.get_parameter(
                                        instr,
                                        param,
                                        wait=True
                                        )

        if self.current_level > maximum_current:

            # We reduce the voltages on the accumulation gates by 1 mV

            self.accumulation_voltages -= 1e-3

            for i in self.accumulation_voltages:

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                instr_handler.set_parameter(
                                        instr,
                                        {param: self.accumulation_voltages[i]},
                                        wait=True
                                        )

            self.SET_current_check(minimum_current, maximum_current)
        
        elif self.current_level < minimum_current:

            # We increase the voltages on the accumulation gates by 1 mV

            self.accumulation_voltages += 1e-3

            for i in self.accumulation_voltages:

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                instr_handler.set_parameter(
                                        instr,
                                        {param: self.accumulation_voltages[i]},
                                        wait=True
                                        )

            self.SET_current_check(minimum_current, maximum_current)

        else:
            pass

    def barrier_barrier_sweep(self, lower_voltages, upper_voltages, num_points):

        # First, we gather the lower voltages to which we set our barriers

        barrier_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Dot Barrier" or self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                barrier_dacs_and_vals[i] = gui.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        print(barrier_dacs_and_vals)

        barrier_targets = []

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Dot Barrier" or self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = barrier_dacs_and_vals[i],
                    end = upper_voltages[i]
                )

                barrier_targets.append(sparam)            
        
        print(barrier_targets)

        sweep_layer = SweepLayer(
            targets = barrier_targets,
            num_points = num_points,
            measurement_time = 0.05
        )

        logger.info("Setting Initial Barrier Voltages...")

        future = gui.experiment_handler.set_voltage_configuration(sweep = sweep_layer,
                                                                  instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Initial Barrier Voltages Set!")

        # Now, we create the sweep parameters for all the gates

        gate_targets_dots = []

        gate_targets_sensors = []

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Dot Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = upper_voltages[i],
                    end = lower_voltages[i]
                )

                gate_targets_dots.append(param)

            elif self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = upper_voltages[i],
                    end = lower_voltages[i]
                )

                gate_targets_sensors.append(param)

        print(gate_targets_dots)
        print(gate_targets_sensors)

        for first, second in zip(gate_targets_dots, gate_targets_dots[1:]):

            sweep_layer = SweepLayer(
                targets = [first, second],
                num_points = num_points,
                measurement_time = 0.05
            )

            logger.info("Dot Barrier-Barrier Scan Starting...")

            future = gui.experiment_handler.do_sweep(sweep = sweep_layer,
                                                 instrument_handler = gui.instrument_handler)
        
            print(future)

            logger.info("Dot Barrier Scan Complete! Finding Set Points...")

        # Now, we ensure that the charge sensor has an appropriate current level before tuning the barriers

        self.SET_current_check(1e-9, 5e-9)

        for first, second in zip(gate_targets_sensors, gate_targets_sensors[1:]):

            sweep_layer = SweepLayer(
                targets = [first, second],
                num_points = num_points,
                measurement_time = 0.05
            )

            logger.info("Sensor Barrier-Barrier Scan Starting...")

            future = gui.experiment_handler.do_sweep(sweep = sweep_layer,
                                                 instrument_handler = gui.instrument_handler)
        
            print(future)

            logger.info("Sensor Barrier-Barrier Scan Complete! Finding Working Points...")

    def coulomb_blockade_sweep(self, barrier_voltages, lower_voltage, upper_voltage, num_points):

        # First, we set the barriers to the lower voltages

        barrier_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Dot Barrier" or self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                barrier_dacs_and_vals[i] = gui.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        print(barrier_dacs_and_vals)

        barrier_targets = []

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Dot Barrier" or self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = barrier_dacs_and_vals[i],
                    end = barrier_voltages[i]
                )

                barrier_targets.append(sparam)            
        
        print(barrier_targets)

        sweep_layer = SweepLayer(
            targets = barrier_targets,
            num_points = num_points,
            measurement_time = 0.05
        )

        logger.info("Setting Initial Barrier Voltages...")

        future = gui.experiment_handler.set_voltage_configuration(sweep = sweep_layer,
                                                                  instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Initial Barrier Voltages Set!")

        # Now, we define the sensor plunger sweeps

        sensor_plunger_targets = []

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                instr, param = p.parameter.split('.', 1)

                sparam = SweepParam(
                    parameter = param,
                    start = lower_voltage,
                    end = upper_voltage
                )

                sensor_plunger_targets.append(sparam)            
        
        print(sensor_plunger_targets)

        sweep_layer = SweepLayer(
            targets = sensor_plunger_targets,
            num_points = num_points,
            measurement_time = 0.05
        )

        logger.info("Charge Sensor Plunger Sweep Starting...")

        future = gui.experiment_handler.do_sweep(sweep = sweep_layer,
                                                                  instrument_handler = gui.instrument_handler)
        
        print(future)

        logger.info("Ohmic Bias Set!")
        
        pass

    def coulomb_diamonds():
        pass

class GlobalChargeTuning(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def tune_lead_dot_tunneling():
        pass

    def plunger_plunger_sweep():
        pass

class VirtualGating(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def lever_arm_matrix():
        pass

class ChargeStateTuning(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def determine_charge_states():
        pass

class FineTuning(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def rabi_oscilations():
        pass

