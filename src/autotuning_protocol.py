# Standard Library Imports
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Third-party Imports
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import yaml
from nicegui import ui
from scipy.ndimage import convolve
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import diamond, rectangle  # noqa
from skimage.transform import probabilistic_hough_line

import qcodes as qc
from qcodes.dataset import AbstractSweep, Measurement
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase

# Local Imports
from data_analysis import (
    extract_lever_arms,
    extract_max_conductance_pair,
    extract_max_conductance_points,
    extract_pinch_off_curve_ranges,
    extract_turn_on_voltage,
    extract_working_point,
    hough_transform,
)
from experiment_base import Sweep, SweepLayer, SweepParam
from tunerlog import TunerLog

logger = TunerLog('Autotuning Protocol')

class Protocol:

    def __init__(self, device_config: str, instr_handler = None, exp_handler = None): 
        
        '''
        Initializes the Autotuning Protocol. The Protocol class is effectively a dataclass that contains information
        from the configuration file provided by the user.

        Paramters
        ---------
        device_config : str
            The filepath to the device config, as a literal string. The expected file type is a .yaml file
        
        instr_handler : instrument_handler instance
            The instance from the gui of the instrument_handler

        exp_handler : experiment_handler instance
            The instance from the gui of the experiment_handler
        '''

        self.instrument_handler = instr_handler
        self.experiment_handler = exp_handler

        # First, we load in the config file

        logger.info("Loading Device Config file...")

        self._load_config_file(device_config)

        # self.directory = r"C:\Users\BaughLaflamme\Desktop\3d1s_W151_1 Measurements\3D1S_w151_1 - Autotuning Tests"
        self.directory = rf"../Protocol_Run_{datetime.now().strftime('%m-%d-%Y')}/Data"

        # Now, we create a dictionary to house a map between gate names and dacs

        self.gates_to_dacs = {}

        for i in self.device_gates:

            self.gates_to_dacs[i] = self.device_gates[i]['channel']

    def _load_config_file(self, device_config):
        
        '''
        Loads the configuration file specified in the __init__ call. Defines attributes used in all Autotuning Protocol stages.

        Paramters
        ---------
        device_config : str
            The filepath to the device config, as a literal string. The expected file type is a .yaml file
        '''

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
        self.abs_max_current_SET = self.config['device']['constraints']['abs_max_current_SET']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.config['device']['constraints']['abs_max_gate_differential']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.initial_ohmic_bias = self.config['device']['constraints']['initial_ohmic_bias']
        self.screening_initial_voltages = self.config['device']['constraints']['screening_initial_voltages']

        # Equipment Setup

        self.voltage_divider_triple_dot = self.config['setup']['voltage_dividers']['voltage_divider_triple_dot']
        self.voltage_divider_SET = self.config['setup']['voltage_dividers']['voltage_divider_SET']
        self.triple_dot_preamp_bias = self.config['setup']['triple_dot_preamp']['preamp_bias']
        self.triple_dot_preamp_sensitivity = self.config['setup']['triple_dot_preamp']['preamp_sensitivity']
        self.SET_preamp_bias = self.config['setup']['SET_preamp']['preamp_bias']
        self.SET_preamp_sensitivity = self.config['setup']['SET_preamp']['preamp_sensitivity']

    def parameter_snapshot(self, name):

        # Collect voltage parameter from all gates and capture the data in the logger. 

        logger.info("Creating Snapshot...")

        snapshot_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            p = self.device_gates[i]['channel']

            instr, param = p.split('.', 1)

            snapshot_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                        instr,
                                        param,
                                        wait=True
                                        )
        
        logger.info(f"Snapshot of {name} created!")
        logger.info(f"{snapshot_dacs_and_vals}")

class Bootstrapping(Protocol):

    def __init__(self, device_config, instr_handler, exp_handler):
        
        '''
        Initializes the Bootstrapping stage of the Autotuning Protocol. This definition first instantiates a
        Protocol object, then runs the Bootstrapping experiments

        Paramters
        ---------
        device_config : str
            The filepath to the device config, as a literal string. The expected file type is a .yaml file
        
        instr_handler : instrument_handler instance
            The instance from the gui of the instrument_handler

        exp_handler : experiment_handler instance
            The instance from the gui of the experiment_handler
        '''

        super().__init__(device_config = device_config, 
                         instr_handler = instr_handler, 
                         exp_handler = exp_handler
                        ) 

    def autotune(self, instr_handler, exp_handler, num_points_bootstrapping: list[int]):

        # First, we reset the noise floor

        self.noise_floor = None

        # Here, we ensure the device is grounded

        self.ground_device(instr_handler = instr_handler, exp_handler = exp_handler)

        # Now, we measure the noise floor with all gates at 0 V.

        self.noise_floor = self.measure_noise_floor()

        logger.info(f"The noise floor is: {self.noise_floor}")

        names = self.instrument_handler.read_buffer(
                ['agilent_left.volt', 'agilent_right.volt'],
            ).keys()

        # Here, we grab the means

        self.means = []

        for i in names:

            mean_name = i + "_mean"
            mean = self.noise_floor[mean_name]

            self.means.append(mean)

        logger.info(f"{self.means}")

        # Now, we attempt to turn on the device

        turn_on_voltages = self.turn_on(ohmic_bias = self.initial_ohmic_bias,
                                        screening_voltage = self.screening_initial_voltages,
                                        gate_voltage = self.abs_max_gate_voltage,
                                        num_points = num_points_bootstrapping[0])
        
        logger.info(f"The Turn On voltages are: {turn_on_voltages}")

        # Now, we attempt to pinch-off

        pinch_off_voltages, saturation_voltages = self.pinch_off(gate_voltage = 1.5,
                                                                 final_voltages = turn_on_voltages,
                                                                 num_points = num_points_bootstrapping[1])

        logger.info(f"The barrier pinch-off voltages are: {pinch_off_voltages}")
        logger.info(f"The saturation voltages are: {saturation_voltages}")

        # Now, we perform scans of adjacent barrier gates on both the dot and sensor sides

        working_point, dot_barrier_set_points = self.barrier_barrier_sweep(lower_voltages = pinch_off_voltages,
                                                   upper_voltages = saturation_voltages,
                                                   num_points = num_points_bootstrapping[2])
        
        # Here, we define the lower voltages (starting) for the charge sensor plungers to be the midpoint between its surrounding barrier gates 

        self.sensor_barrier_voltages = list(working_point)

        self.sensor_plunger_lower_voltages = [sum(working_point) / len(working_point)]

        # Similarly, for the dot plungers, we define their lower (starting) voltages to be the midpoint between the surrounding barrier gates

        self.dot_plunger_lower_voltages = []

        for i in dot_barrier_set_points:
            
            voltage = sum(i) / len(i)

            self.dot_plunger_lower_voltages.append(voltage)

        # Here, we perform a sweep of the charge sensor plunger gates to find a sensitive coulomb blockade peak

        self.sensing_points, (self.best_point, self.best_conductance) = self.coulomb_blockade_sweep(sensor_barrier_voltages = self.sensor_barrier_voltages, 
                                                                                                    lower_voltages = self.sensor_plunger_lower_voltages, 
                                                                                                    upper_voltages = [1.5], 
                                                                                                    num_points = num_points_bootstrapping[3])

        logger.info(f" The best plunger voltages for sensing are: {self.sensing_points}")

        # Here, we also perform a coulomb diamond scan of each charge sensor

        self.coulomb_diamonds(lower_sd_voltages = [-0.0002], 
                              upper_sd_voltages = [0.0002], 
                              lower_plunger_voltages = self.plunger_starting_voltages,
                              upper_plunger_voltages = [1.5], 
                              num_points = num_points_bootstrapping[4])

        pass

    def autotune_bootstrapping(self, instr_handler, exp_handler, ohmic_bias, screening_voltage, gate_voltage, num_points):

        # First, we reset the noise floor

        self.noise_floor = None

        # Here, we ensure the device is grounded

        self.ground_device(instr_handler = instr_handler, exp_handler = exp_handler)

        # Now, we measure the noise floor with all gates at 0 V.

        self.noise_floor = self.measure_noise_floor()

        logger.info(f"The noise floor is: {self.noise_floor}")

        names = self.instrument_handler.read_buffer(
                ['agilent_left.volt', 'agilent_right.volt'],
            ).keys()

        # Here, we grab the means

        self.means = []

        for i in names:

            mean_name = i + "_mean"
            mean = self.noise_floor[mean_name]

            self.means.append(mean)

        logger.info(f"{self.means}")

        # Now, we attempt to turn on the device

        turn_on_voltages = self.turn_on(ohmic_bias = ohmic_bias,
                                        screening_voltage = screening_voltage,
                                        gate_voltage = gate_voltage,
                                        num_points = num_points)
        
        logger.info(f"The Turn On voltages are: {turn_on_voltages}")

        # Now, we attempt to pinch-off

        pinch_off_voltages, saturation_voltages = self.pinch_off(gate_voltage = gate_voltage,
                                                                 final_voltages = 0.0,
                                                                 num_points = num_points)

        logger.info(f"The barrier pinch-off voltages are: {pinch_off_voltages}")
        logger.info(f"The saturation voltages are: {saturation_voltages}")

        # Now, we perform scans of adjacent barrier gates on both the dot and sensor sides

        working_point, dot_barrier_set_points = self.barrier_barrier_sweep(lower_voltages = pinch_off_voltages,
                                                   upper_voltages = saturation_voltages,
                                                   num_points = num_points)
        
        # Here, we define the lower voltages (starting) for the charge sensor plungers to be the midpoint between its surrounding barrier gates 

        self.sensor_barrier_voltages = list(working_point)

        self.sensor_plunger_lower_voltages = [sum(working_point) / len(working_point)]

        # Similarly, for the dot plungers, we define their lower (starting) voltages to be the midpoint between the surrounding barrier gates

        self.dot_plunger_lower_voltages = []

        for i in dot_barrier_set_points:
            
            voltage = sum(i) / len(i)

            self.dot_plunger_lower_voltages.append(voltage)

        # Here, we perform a sweep of the charge sensor plunger gates to find a sensitive coulomb blockade peak

        self.sensing_points, (self.best_point, self.best_conductance) = self.coulomb_blockade_sweep(sensor_barrier_voltages = self.sensor_barrier_voltages, 
                                                                                                    lower_voltages = self.sensor_plunger_lower_voltages, 
                                                                                                    upper_voltages = [gate_voltage], 
                                                                                                    num_points = num_points
                                                                                                   )

        logger.info(f" The best plunger voltages for sensing are: {self.sensing_points}")

        # Here, we also perform a coulomb diamond scan of each charge sensor

        self.coulomb_diamonds(lower_sd_voltages = [-0.0002], 
                              upper_sd_voltages = [0.0002], 
                              lower_plunger_voltages = self.plunger_starting_voltages,
                              upper_plunger_voltages = [1.5], 
                              num_points = 100)

        pass

    def ground_device(self, instr_handler, exp_handler):
        
        '''
        Grounds the device by smoothly setting all gate electrodes from their current voltages to 0 V. 

        Paramters
        ---------
        instr_handler : instrument_handler instance
            The instance from the gui of the instrument_handler

        exp_handler : experiment_handler instance
            The instance from the gui of the experiment_handler
        '''

        # First, we grab all the connected dacs and current values

        dacs_and_vals = {}

        for i in self.gates_to_dacs:
            
            p = self.device_gates[i]['channel']

            instr, param = p.split('.', 1)

            dacs_and_vals[i] = instr_handler.get_parameter(
                                        instr,
                                        param,
                                        wait=True
                                        )

        # Now, we create the sweep parameters

        targets = []

        for gate, dac_and_val in dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                param = SweepParam(
                    parameter = "spi_rack." + dac,
                    start = starting_val,
                    end = 0.0
                )

                targets.append(param)

        sweep_layer = SweepLayer(
            targets = targets,
            num_points = 100,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        # Now, we perform the sweep

        logger.info("Grounding Device...")

        future = exp_handler.set_voltage_configuration(sweep = sweep,
                                                       instrument_handler = instr_handler)

        logger.info("Device Grounded!")

    def measure_noise_floor(self, measurement_time = 30):

        '''
        Measures the noise floor of both device conductive channels for a static voltage configuration and
        collects statistics of the current outputs.

        Paramters
        ---------
        measurement_time : float
            The amount of time over which data is taken from the readout buffer. This determines
            how many points are being used when gathering statistics, default is 30 s.
        
        Returns
        -------
        results : dict
            A dictionary containing statistics of the data acquired during the measurement time,
            including the mean, median, std, mad, and robust std
        '''

        # We collect the readout buffer for 1 minute and average the values to measure the noise floor.

        logger.info("Starting noise floor measurement...")

        stats = {
            'agilent_left.volt': {
                'n': 0,
                'mean': 0.0,
                'M2': 0.0,
                'samples': []
            },
            'agilent_right.volt': {
                'n': 0,
                'mean': 0.0,
                'M2': 0.0,
                'samples': []
            }
        }

        for i in range(measurement_time):

            vals = self.instrument_handler.read_buffer(
                ['agilent_left.volt', 'agilent_right.volt'],
            )

            # Extract both values for each DMM

            for name, x in vals.items():

                s = stats[name]

                s['samples'].append(x)

                s['n'] += 1
                n = s['n']

                delta = x - s['mean']
                s['mean'] += delta / n
                delta2 = x - s['mean']

                s['M2'] += delta2 * delta

        results = {}

        logger.info("Analyzing!")

        for name, s in stats.items():

            mean = s['mean']

            if s['n'] > 1:
                
                variance = s['M2'] / (s['n'] - 1)
                std = np.sqrt(variance)
            else:
                std = 0.0

            samples = np.array(s['samples'])
            median = np.median(samples)
            mad = np.median(np.abs(samples - median))

            robust_std = 1.4826 * mad

            results[f"{name}_mean"] = mean
            results[f"{name}_std"] = std
            results[f"{name}_median"] = median
            results[f"{name}_mad"] = mad
            results[f"{name}_robust_std"] = robust_std

        #print("Noise Floor Found!")

        return results

    def turn_on(self, ohmic_bias, screening_voltage, gate_voltage, num_points):

        '''
        A sweep defined from 0 V to the maximum gate voltage for all accumulation, barrier, and plunger gates. The
        assumption the protocol makes is that dots are formed underneath the plunger gates

        Paramters
        ---------
        ohmic_bias : float
            The bias set to the conducting channels of the dot and sensor sides before turn-on
        
        screening_voltage : float
            The voltage set to the screening gates for the dot and sensor side, as well as the central screening gate
        
        gate_voltage : float
            The maximum (endpoint) gate voltage for which to sweep all accumulation, barrier, and plunger gates 
        
        num_points : int
            The number of points from 0 to the maximum voltage, inclusive, for the sweep

        Returns
        -------
        turn_on_voltages : list[float] | None :
            A list of voltages at which the current channels in the device turn on. If one or more channels
            was determined to not turn on, returns None 

        '''

        # First we grab all the dacs and values to set to our ohmics
 
        ohmic_targets = []

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Dot Ohmic" or self.device_gates[i]['type'] == "Sensor Ohmic":

                p = self.device_gates[i]['channel']

                ohmic_voltage = ohmic_bias / self.voltage_divider_SET

                sparam = SweepParam(
                    parameter = p,
                    start = 0.0,
                    end = ohmic_voltage
                )

                ohmic_targets.append(sparam)            
        
        sweep_layer = SweepLayer(
            targets = ohmic_targets,
            num_points = num_points,
            measurement_time = 0.05
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        # Now, we set the ohmic biases

        logger.info("Setting Ohmic Bias...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                  instrument_handler = self.instrument_handler)

        logger.info("Ohmic Bias Set!")

        # Next, we set all constant initial voltages before Turn-On. For the Intel device, this is the screening gates

        screening_targets = []

        screening_types = ["Dot Screening", "Sensor Screening", "Central Screening"]

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] in screening_types:

                p = self.device_gates[i]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = 0.0,
                    end = screening_voltage
                )

                screening_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = screening_targets,
            num_points = num_points,
            measurement_time = 0.05
        )
        
        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        # Now, we set these initial voltages

        logger.info("Setting Screening Gate Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                  instrument_handler = self.instrument_handler)

        logger.info("Screening Gate Voltages Set!")

        # Now, we create the sweep parameters for all the gates

        gate_targets = []

        excluded_types = ["Dot Ohmic", "Sensor Ohmic", "Dot Screening", "Sensor Screening", "Central Screening"]

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] not in excluded_types:

                p = self.device_gates[i]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = 0,
                    end = gate_voltage
                )

                gate_targets.append(sparam)

        sweep_layer = SweepLayer(
            targets = gate_targets,
            num_points = num_points,
            measurement_time = 0.2
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        # Now, we attempt to turn on the device

        logger.info("Device Turn-On Starting...")

        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = "Turn_On_" + time + ".csv"

        future = self.experiment_handler.do_sweep(sweep = sweep,
                                                 instrument_handler = self.instrument_handler,
                                                 filename = filename)

        logger.info("Device Turn-On Sweep Complete! Confirming Turn-On...")

        # Now, to confirm that the device has turned on, we measure the current level and compare to the noise floor

        turn_on_measurement = self.measure_noise_floor()

        names = self.instrument_handler.read_buffer(
                ['agilent_left.volt', 'agilent_right.volt'],
            ).keys()
        
        means = []

        for i in names:

            mean_name = i + "_mean"
            mean = turn_on_measurement[mean_name]

            means.append(mean)


        """ 
        Now, we compare the means of each measurement. 
        If the mean measured after turn-on is larger than 10 times the noise floor mean, 
        then we say that the device has turned on.
        
        """

        turn_on_check = []

        for i, item in enumerate(self.means):

            if abs(10 * item) < means[i]:
                turn_on_check.append(True)
            else:
                turn_on_check.append(False)

        if all(item is True for item in turn_on_check):

            logger.info("Turn-Ons Confirmed Succussfully! Determining Turn-On Voltages...")

            # Here, we get the data from the CSV

            filepath = os.path.join(self.directory, filename)

            df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

            # Now, we get the data for DMMs and for the set voltages, and convert the data from voltage to current, then to nA. We also convert the mean

            turn_on_sweep = df.iloc[:,0].to_numpy()

            triple_dot_data = df.iloc[:,-2].to_numpy() * self.triple_dot_preamp_sensitivity * 1e9

            SET_data = df.iloc[:,-1].to_numpy() * self.SET_preamp_sensitivity * 1e9

            current_means = []

            for mean in self.means:

                mean *= self.SET_preamp_sensitivity * 1e9

                current_means.append(mean)

            current_data = [triple_dot_data, SET_data] 

            # Now, we extract the turn on voltages and return them

            turn_on_voltages = []

            turn_on_filenames = ['Triple_Dot_Turn_On' + time + '.png', 'SET_Turn_On' + time + '.png']

            for i, mean in enumerate(self.means):

                turnon_voltage = extract_turn_on_voltage(x_data = turn_on_sweep,
                                                         y_data = current_data[i],
                                                         noisefloor = current_means[i],
                                                         filepath = self.directory,
                                                         filename = turn_on_filenames[i],
                                                         plot_results = True)

                turn_on_voltages.append(turnon_voltage)

            return turn_on_voltages

        else:

            # If one or more of the channels fail, we reject the device and ground all gates

            logger.info("""
                        Turn-On confirmation failed for one or more channels! 
                        Please confirm that all gates needed for Turn-On are being swept,
                        otherwise the device does not turn-on for one or more channels within the given maximum voltage.
                        """)
            
            self.ground_device()

            return None

    def pinch_off(self, gate_voltage, num_points):

        '''
        Creates sweeps defined from the maximum gate voltage to 0 to determine the pinch-off windows for all finger gates

        Paramters
        ---------
        gate_voltage : float
            The maximum (startpoint) gate voltage for which to sweep all accumulation, barrier, and plunger gates 
        
        num_points : int
            The number of points from the maximum voltage to 0, inclusive, for the sweep

        Returns
        -------
        barrier_pinch_off_voltages, barrier_saturation_voltages : list[float], list[float]
            lists of voltages at which the current channels pinch-off and saturate respectively, for each gate.
            Failed pinch-offs and Saturation voltages return as None in the list

        '''

        excluded_types = ["Dot Ohmic", "Sensor Ohmic", "Dot Screening", "Sensor Screening", "Central Screening", "Dot Barrier", "Sensor Barrier"]

        pinch_off_voltages= []

        saturation_voltages = []

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] not in excluded_types:

                if self.device_gates[i]['type'].startswith('Dot'):

                    result = self.pinch_off_individual(gate_voltage = gate_voltage, 
                                                       final_voltage = 0.0,
                                                       gate_name = self.device_gates[i]['label'],
                                                       gate_type = self.device_gates[i]['type'], 
                                                       channel = self.device_gates[i]['channel'],
                                                       num_points = num_points
                                                      )

                    pinch_off_voltages.append(result[0])
                    saturation_voltages.append(result[1])

                elif self.device_gates[i]['type'].startswith('Sensor'):

                    result = self.pinch_off_individual(gate_voltage = gate_voltage, 
                                                       final_voltage = 0.0, 
                                                       gate_name = self.device_gates[i]['label'],
                                                       gate_type = self.device_gates[i]['type'],
                                                       channel = self.device_gates[i]['channel'],
                                                       num_points = num_points
                                                      )

                    pinch_off_voltages.append(result[0])
                    saturation_voltages.append(result[1])

                else:

                    logger.info("""
                                Please check the types of your gates in the config file. The allowed types are: 
                                Dot Ohmic, Sensor Ohmic, Dot Screening, Sensor Screening, Central Screening, 
                                Dot Barrier, Sensor Barrier, Dot Accumulation, Sensor Accumulation, Dot Plunger, Sensor Plunger
                                """)
                    
                    return None
                    
        logger.info(f"{pinch_off_voltages}")
        logger.info(f"{saturation_voltages}")

        # Here, we have a check to see if any of the Pinch-Offs  above failed, if so, we indicate that the Pinch-Off stage has failed

        # For this specific device, the accumulation gates do not Pinch-Off, so we will be exluding them. For now, we will exclude this step.

        # Now, we set the voltages on these gates to the the saturation voltages

        accumulation_saturation_voltages = [1.25, 1.25, 1.25, 1.25]

        for i, voltage in enumerate(accumulation_saturation_voltages):

            saturation_voltages[i] = voltage

        pinch_off_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] not in excluded_types:

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                pinch_off_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        pinch_off_targets = []

        endpoint_iter = iter(saturation_voltages)

        for gate, dac_and_val in pinch_off_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                logger.info(f"{p}")

                end_val = next(endpoint_iter)

                logger.info(f"{end_val}")

                if end_val == None:
                    continue

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = end_val
                )

                logger.info(f"{sparam}")

                pinch_off_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = pinch_off_targets,
            num_points = num_points,
            measurement_time = 0.05
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Saturation Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                  instrument_handler = self.instrument_handler)

        logger.info("Saturation Voltages Set!")

        self.SET_current_check(minimum_current = 3, maximum_current = 5)

        # Now that the non-barrier finger gates are set to lower voltages, we perform pinch-offs of the barrier gates for both sides

        included_types = ["Dot Barrier", "Sensor Barrier"]

        barrier_pinch_off_voltages = []
        barrier_saturation_voltages = []

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] in included_types:

                if self.device_gates[i]['type'].startswith('Dot'):

                    result = self.pinch_off_individual(gate_voltage = gate_voltage, 
                                                       final_voltage = 0.0,
                                                       gate_name = self.device_gates[i]['label'],
                                                       gate_type = self.device_gates[i]['type'], 
                                                       channel = self.device_gates[i]['channel'],
                                                       num_points = num_points
                                                      )

                    barrier_pinch_off_voltages.append(result[0])
                    barrier_saturation_voltages.append(result[1])

                elif self.device_gates[i]['type'].startswith('Sensor'):

                    result = self.pinch_off_individual(gate_voltage = gate_voltage, 
                                                       final_voltage = 0.0, 
                                                       gate_name = self.device_gates[i]['label'],
                                                       gate_type = self.device_gates[i]['type'],
                                                       channel = self.device_gates[i]['channel'],
                                                       num_points = num_points
                                                      )

                    barrier_pinch_off_voltages.append(result[0])
                    barrier_saturation_voltages.append(result[1])        

        logger.info("Pinch-Offs Complete!")

        return barrier_pinch_off_voltages, barrier_saturation_voltages

    def pinch_off_individual(self, gate_voltage, final_voltage, gate_name, gate_type, channel, num_points):

        '''
        Runs a sweep defined from the maximum gate voltage to 0 to determine the pinch-off windows for
        a particular finger gate.

        Paramters
        ---------
        gate_voltage : float
            The maximum (startpoint) gate voltage for which the sweep starts 
        
        final_voltage : float
            The minimum (endpoint) gate voltage for which the sweep ends

        gate_name : str
            The name of the gate being swept
        
        gate_type : str
            The gate type, with possible inputs: Accumulation, Barrier, Plunger
            The type is passed into the analysis function to determine the specific saturation point

        channel : str
            The dac channel corresponding to the gate
            Ex: For the Spi Rack, argument will look like spi_rack.module1.dac1.voltage    

        num_points : int
            The number of points from the maximum voltage to 0, inclusive, for the sweep

        Returns
        -------
        pinch_off_window : tuple(float, float) | tuple(None, None)
            A tuple of the form (pinch-off voltage, saturation voltage), or (None, None) if either is not found.
        
        '''

        sparam = SweepParam(parameter = channel, 
                            start = gate_voltage, 
                            end = final_voltage
        )

        sweep_layer = SweepLayer(targets = [sparam], 
                                    num_points = num_points, 
                                    measurement_time = 0.2
        )
        
        measure = lambda ih, sp: (
            ih.read_buffer([
                'agilent_left.volt',
                'agilent_right.volt'
            ]),
            ['agilent_left.volt', 'agilent_right.volt']
        
        )

        sweep = Sweep([sweep_layer], measure)

        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{gate_name}_Pinch_Off_{time_str}.csv"
        filename2 = f"{gate_name}_Pinch_Off_{time_str}.png"

        logger.info(f"{gate_name} sweeping to {final_voltage} V...")

        self.experiment_handler.do_sweep(sweep = sweep,
                                            instrument_handler = self.instrument_handler,
                                            filename = filename)
        
        logger.info(f"{gate_name} Pinch-Off Complete! Confirming Pinch-Off...")

        pinch_off_measurement = self.measure_noise_floor()
        
        names = list(self.instrument_handler.read_buffer(
            ['agilent_left.volt', 'agilent_right.volt']
        ).keys())

        """ 
        Now, we compare the means of each measurement. 
        If the mean measured after Pinch_Off is comparable the noise floor mean, 
        then we say that the device has Pinched Off.
        
        """

        noise_floor_idx = None

        if gate_type.startswith('Dot'):

            mean = pinch_off_measurement[names[0] + "_mean"]
            noise_floor_idx = 0
        
        else:
            
            mean = pinch_off_measurement[names[1] + "_mean"]
            noise_floor_idx = 1

        # The pinched condition translates to the difference in the means being less than 300 pA.

        pinched = abs(mean - abs(self.means[noise_floor_idx])) < 3e-2

        sparam_return = SweepParam(parameter = channel, 
                                    start = final_voltage, 
                                    end = gate_voltage)
        
        sweep_layer_return = SweepLayer(targets = [sparam_return], 
                                        num_points = num_points, 
                                        measurement_time = 0.1)
        
        measure_return = lambda ih, sp: (
            ih.read_buffer([
                'agilent_left.volt',
                'agilent_right.volt'
            ]),
            ['agilent_left.volt', 'agilent_right.volt']
        
        )
        
        self.experiment_handler.set_voltage_configuration(sweep = Sweep([sweep_layer_return], measure_return),
                                                            instrument_handler = self.instrument_handler)
        
        logger.info(f"{gate_name} returned to {gate_voltage} V.")

        if pinched:

            logger.info(f"{gate_name} Pinch-Off confirmed! Finding Pinch-Off Window...")

            filepath = os.path.join(self.directory, filename)
            df = pd.read_csv(filepath, delimiter=",", header=None, skiprows=1)

            pinch_off_sweep = df.iloc[:, 0]
            data = [df.iloc[:, -2] * self.triple_dot_preamp_sensitivity * 1e9, df.iloc[:, -1] * self.SET_preamp_sensitivity * 1e9]

            gate_type_no_side = gate_type.split()[1]

            logger.info(f"Gate Type: {gate_type_no_side}")

            pinch_off_window = extract_pinch_off_curve_ranges(x_data = pinch_off_sweep,
                                                              y_data = data[noise_floor_idx],
                                                              noisefloor = self.means[noise_floor_idx],
                                                              gate_type = gate_type_no_side,
                                                              filepath = self.directory,
                                                              filename = filename2
                                                             )

            logger.info(f"{pinch_off_window}")

            return pinch_off_window

        else:

            logger.info(f"{gate_name} did not pinch off at {final_voltage} V. Pinch-Off Failed. Returning None...")

            return (None, None)

    def SET_current_check(self, minimum_current, maximum_current):

        '''
        Iteratively adjusts the voltages of the accumulation gates and measures the current through the SET
        to ensure that it falls within the specified bounds

        Paramters
        ---------
        minimum_current : float
            The minimum current bound for the SET current 
        
        maximum_current : float
            The maximum current bound for the SET current
        
        '''

        # First, we need to read the current and check if it is above or below the current values specified.

        current_level = self.measure_noise_floor()

        names = list(self.instrument_handler.read_buffer(
                ['agilent_left.volt', 'agilent_right.volt'],
            ).keys())
        
        current_means = []

        for i in names:

            mean_name = i + "_mean"
            mean = current_level[mean_name] * self.SET_preamp_sensitivity * 1e9

            current_means.append(mean)

        # Here, we get the current accumulation voltages

        included_types = ['Dot Accumulation', 'Sensor Accumulation']

        accumulation_voltages = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] in included_types:

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                accumulation_voltages[i] = self.instrument_handler.get_parameter(
                                        instr,
                                        param,
                                        wait=True
                                        )

        SET_current_level = current_means[1]

        logger.info(f"Current Level {SET_current_level}. Checking Current Level...")

        while SET_current_level > maximum_current or SET_current_level < minimum_current:

            if SET_current_level > maximum_current:

                logger.info("Current too high! Reducing Accumulation Gate Voltages by 1 mV...")

                # We reduce the voltages on the accumulation gates by 1 mV

                for key, val in accumulation_voltages.items():

                    p = self.device_gates[key]['channel']

                    instr, param = p.split('.', 1)

                    current_voltage = val[param]
                    new_voltage = current_voltage - 1e-3

                    logger.info(f"Setting {param} from {current_voltage} to {new_voltage}")

                    accumulation_voltages[key][param] = new_voltage

                    self.instrument_handler.set_parameter(
                        instr,
                        {param: new_voltage},
                        wait=True
                    )

                # Update the SET current level

                new_current_level = self.measure_noise_floor()

                new_SET_current_name = names[1] + "_mean"

                new_SET_current_level = (
                    new_current_level[new_SET_current_name]
                    * self.SET_preamp_sensitivity
                    * 1e9
                )

                SET_current_level = new_SET_current_level

                logger.info(f"New Current Level: {SET_current_level}. Checking Current Level")

                time.sleep(10)
        
            elif SET_current_level < minimum_current:

                logger.info("Current too low! Increasing Accumulation Gate Voltages by 1 mV...")

                # We increase the voltages on the accumulation gates by 1 mV

                for key, val in accumulation_voltages.items():

                    p = self.device_gates[key]['channel']

                    instr, param = p.split('.', 1)

                    current_voltage = val[param]
                    new_voltage = current_voltage + 1e-3

                    logger.info(f"Setting {param} from {current_voltage} to {new_voltage}")

                    accumulation_voltages[key][param] = new_voltage

                    self.instrument_handler.set_parameter(
                        instr,
                        {param: new_voltage},
                        wait=True
                    )

                # Update the SET current level

                new_current_level = self.measure_noise_floor()

                new_SET_current_name = names[1] + "_mean"

                new_SET_current_level = (
                    new_current_level[new_SET_current_name]
                    * self.SET_preamp_sensitivity
                    * 1e9
                )

                SET_current_level = new_SET_current_level

                logger.info(f"New Current Level: {SET_current_level}. Checking Current Level")

                time.sleep(10)
        
        logger.info(f"Final SET Current Level: {SET_current_level}")

    def barrier_barrier_sweep(self, lower_voltages, upper_voltages, num_points):

        '''
        Runs 2D sweeps of adjacent barrier gates on the dot side and the sensor side using the pinch-offs and saturation voltages
        as bounds. Determines optimal working points for both the dot and sensor side using ridge detection

        Paramters
        ---------
        lower_voltages : float
            The maximum (startpoint) gate voltages for which the sweep starts 
        
        final_voltages : float
            The minimum (endpoint) gate voltages for which the sweep ends 

        num_points : int
            The number of points, in both voltages, from lower to upper bounds, inclusive, for the sweeps

        Returns
        -------
        best_sensing_point, best_working_points : tuple(float, float), list[tuple(float, float)]
            the best found voltages for all sensor barrier pairs for the charge sensor, and the dot barrier pairs
        
        '''

        # First, we gather the upper voltages to which we set our barriers

        barrier_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Dot Barrier" or self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                barrier_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        barrier_targets = []

        endpoint_iter = iter(upper_voltages)

        for gate, dac_and_val in barrier_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                end_val = next(endpoint_iter)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = end_val
                )

                barrier_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = barrier_targets,
            num_points = num_points,
            measurement_time = 0.05
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Barrier Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Barrier Voltages Set!")

        self.SET_current_check(minimum_current = 3, maximum_current = 5)

        # Now, we create the sweep parameters for all the gates

        gate_targets_dots = []

        gate_targets_sensors = []

        gate_return_dots = []

        dot_barrier_names = []

        sensor_barrier_names = []

        # Here, we build the actual barrier-barrier scans

        barrier_idx = 0

        for item in self.gates_to_dacs:

            if self.device_gates[item]['type'] == "Dot Barrier":

                p = self.device_gates[item]['channel']

                name = self.device_gates[item]["label"]

                dot_barrier_names.append(name)

                sparam = SweepParam(
                    parameter = p,
                    start = upper_voltages[barrier_idx],
                    end = lower_voltages[barrier_idx]
                )

                gate_targets_dots.append(sparam)

                barrier_idx += 1

            elif self.device_gates[item]['type'] == "Sensor Barrier":

                p = self.device_gates[item]['channel']

                name = self.device_gates[item]["label"]

                sensor_barrier_names.append(name)

                sparam = SweepParam(
                    parameter = p,
                    start = upper_voltages[barrier_idx],
                    end = lower_voltages[barrier_idx]
                )

                gate_targets_sensors.append(sparam)

                barrier_idx += 1

        # Here, we define the return sweeps

        barrier_idx = 0

        for item in self.gates_to_dacs:

            if self.device_gates[item]['type'] == "Dot Barrier":

                p = self.device_gates[item]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = lower_voltages[barrier_idx],
                    end = upper_voltages[barrier_idx]
                )

                gate_return_dots.append(sparam)

                barrier_idx += 1

        best_working_points = []

        for i, (first, second, ret_first, ret_second) in enumerate(zip(gate_targets_dots, 
                                                                       gate_targets_dots[1:], 
                                                                       gate_return_dots, 
                                                                       gate_return_dots[1:]
                                                                       )
                                                                  ):

            sweep_layer_1 = SweepLayer(
                targets = [first],
                num_points = num_points,
                measurement_time = 0.2
            )

            sweep_layer_2 = SweepLayer(
                targets = [second],
                num_points = num_points,
                measurement_time = 0.2
            )

            measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
            )

            sweep = Sweep([sweep_layer_1, sweep_layer_2], measure)

            logger.info("Dot Barrier-Barrier Scan Starting...")

            gate_name_1 = dot_barrier_names[i]
            gate_name_2 = dot_barrier_names[i + 1]

            gates = [gate_name_1, gate_name_2]

            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{gate_name_1}_{gate_name_2}_Scan_{time_str}.csv"

            future = self.experiment_handler.do_sweep(sweep = sweep,
                                                      instrument_handler = self.instrument_handler,
                                                      filename = filename)

            # Here, we find the set points for the dot barrier-barrier scans

            logger.info("Dot Barrier Scan Complete! Finding Set Points...")

            filepath = os.path.join(self.directory, filename)

            df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

            lb_data = df.iloc[:, 0]
            rb_data = df.iloc[:, 1]

            current_data = df.iloc[:, -2]

            filename = filename.removesuffix('.csv') + ".png"

            best_point, set_points, perp_traces = extract_working_point(lb_data = lb_data,
                                                                        rb_data = rb_data,
                                                                        current_data = current_data,
                                                                        gates = gates,
                                                                        DotTuning = "Triple Dot",
                                                                        barrier_pinch_offs = [lower_voltages[i], lower_voltages[i + 1]],                          
                                                                        filepath = self.directory,
                                                                        filename = filename
                                                                       )

            best_working_points.append(best_point)

            logger.info(f"Best point: {best_point}")

            # Now, we reset the barriers back to their starting voltages

            return_layer = SweepLayer(
                targets = [ret_first, ret_second],
                num_points = num_points,
                measurement_time = 0.05
            )

            return_sweep = Sweep([return_layer], measure)

            logger.info("Returning Barriers to starting values...")

            future = self.experiment_handler.set_voltage_configuration(sweep = return_sweep,
                                                                       instrument_handler = self.instrument_handler)

        # Now, we unpack the tuples found into voltages to set to the Dot Barriers

        logger.info(f"Best Points: {best_working_points}")

        dot_barrier_voltages = []

        dot_barrier_voltages = [best_working_points[0][0]]
        
        for left, right in zip(best_working_points[:-1], best_working_points[1:]):
            
            dot_barrier_voltages.append((left[1] + right[0]) / 2)
        
        dot_barrier_voltages.append(best_working_points[-1][1])

        logger.info(f"Barrier Voltages: {dot_barrier_voltages}")

        # Now, we set the barriers on the dot side to the set points found. 

        barrier_targets_dots = []

        barrier_idx = 0

        dot_side = None

        for item in self.gates_to_dacs:

            if self.device_gates[item]['type'] == "Dot Barrier":
                
                dot_side = True

                p = self.device_gates[item]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = upper_voltages[barrier_idx],
                    end = dot_barrier_voltages[barrier_idx]
                )

                barrier_targets_dots.append(sparam)

                barrier_idx += 1

        if dot_side:

            sweep_layer = SweepLayer(
                    targets = barrier_targets_dots,
                    num_points = num_points,
                    measurement_time = 0.05
            )
            
            measure = lambda ih, sp: (
                    ih.read_buffer([
                        'agilent_left.volt',
                        'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
            )

            sweep = Sweep([sweep_layer], measure)

            logger.info("Setting Dot Barriers to Set Points...")

            future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                       instrument_handler = self.instrument_handler)

            logger.info("Barrier Set Points Set!")

        # Now, we ensure that the charge sensor has an appropriate current level before tuning the barriers

        self.SET_current_check(minimum_current = 2, maximum_current = 4)

        for i, (first, second) in enumerate(zip(gate_targets_sensors, 
                                                gate_targets_sensors[1:]
                                               )
                                           ):

            sweep_layer_1 = SweepLayer(
                targets = [first],
                num_points = num_points,
                measurement_time = 0.2
            )

            sweep_layer_2 = SweepLayer(
                targets = [second],
                num_points = num_points,
                measurement_time = 0.2
            )

            measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
            )

            sweep = Sweep([sweep_layer_1, sweep_layer_2], measure)

            logger.info("Sensor Barrier-Barrier Scan Starting...")

            gate_name_1 = sensor_barrier_names[i]
            gate_name_2 = sensor_barrier_names[i + 1]

            gates = [gate_name_1, gate_name_2]

            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{gate_name_1}_{gate_name_2}_Scan_{time_str}.csv"

            future = self.experiment_handler.do_sweep(sweep = sweep,
                                                      instrument_handler = self.instrument_handler,
                                                      filename = filename)

            # Here, we determine the working points for the charge sensor

            logger.info("Sensor Barrier-Barrier Scan Complete! Finding Working Points...")

            filepath = os.path.join(self.directory, filename)

            df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

            lb_data = df.iloc[:, 1]
            rb_data = df.iloc[:, 0]

            current_data = df.iloc[:, -1]

            filename = filename.removesuffix('.csv') + ".png"

            best_sensing_point, working_points, perp_traces = extract_working_point(lb_data = lb_data,
                                                                            rb_data = rb_data,
                                                                            current_data = current_data,
                                                                            gates = gates,
                                                                            DotTuning = "SET",
                                                                            barrier_pinch_offs = [lower_voltages[-i -2], lower_voltages[-i - 1]],                          
                                                                            filepath = self.directory,
                                                                            filename = filename
                                                                           )

            logger.info(f"Best: {best_sensing_point}")

            return best_sensing_point, best_working_points

    def coulomb_blockade_sweep(self, sensor_barrier_voltages, lower_voltages, upper_voltages, num_points):

        '''
        Runs 1D traces of the charge sensor plungers and determines the most sensitive points in this space

        Paramters
        ---------
        sensor_barrier_voltages : float
            The voltages the sensor barriers are set to before sweeping (i.e. the working point) 
        
        lower_voltages : float
            The minimum (endpoint) gate voltages for which the sweep ends 

        upper_voltages : float
            The maximum (endpoint) gate voltages for which the sweep ends 

        num_points : int
            The number of points, inclusive, for the sweeps

        Returns
        -------
        conductance_points, (best_point, best_conductance) : list[float], tuple(float, float)
            All highest sensitivity points for all coulomb blockade peaks, with the best given as (plunger voltage, conductance)
        
        '''

        # First, we set the sensor_barriers to the their respective voltages

        sensor_barrier_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Barrier":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                sensor_barrier_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        barrier_targets = []

        endpoint_itr = iter(sensor_barrier_voltages)

        for gate, dac_and_val in sensor_barrier_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                barrier_val = next(endpoint_itr)

                sparam = SweepParam(
                    parameter = "spi_rack." + dac,
                    start = starting_val,
                    end = barrier_val
                )

                barrier_targets.append(sparam)

        sweep_layer = SweepLayer(
            targets = barrier_targets,
            num_points = 100,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Sensor Barriers to Working Point...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Sensor Barriers Set!")         

        # Now, we set our charge sensor plunger gates to their initial values

        charge_sensor_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                charge_sensor_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )
                
        charge_sensor_targets = []

        endpoint_iter = iter(lower_voltages)

        for gate, dac_and_val in charge_sensor_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                sensor_plunger_val = next(endpoint_iter)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = sensor_plunger_val
                )

                charge_sensor_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = charge_sensor_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Charge Sensor Plunger Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Charge Sensor Plunger Voltages Set!")

        # Now, we define the sensor plunger sweeps

        sensor_plunger_targets = []

        sensor_plunger_idx = 0

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = lower_voltages[sensor_plunger_idx],
                    end = upper_voltages[sensor_plunger_idx]
                )

                sensor_plunger_targets.append(sparam)      

                sensor_plunger_idx += 1      
        
        logger.info(f"{sensor_plunger_targets}")

        sensor_plunger_idx = 0

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Sensor Plunger':
                
                sweep_layer = SweepLayer(
                    targets = [sensor_plunger_targets[sensor_plunger_idx]],
                    num_points = num_points,
                    measurement_time = 0.2
                )

                sensor_plunger_idx += 1

                measure = lambda ih, sp: (
                        ih.read_buffer([
                            'agilent_left.volt',
                            'agilent_right.volt'
                        ]),
                        ['agilent_left.volt', 'agilent_right.volt']
                )

                sweep = Sweep([sweep_layer], measure)

                logger.info("Charge Sensor Plunger Sweep Starting...")

                gate_name = self.device_gates[i]['label']

                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{gate_name}_Sweep_{time_str}.csv"

                future = self.experiment_handler.do_sweep(sweep = sweep,
                                                          instrument_handler = self.instrument_handler,
                                                          filename = filename
                                                         )
        
                logger.info("Charge Sensor Plunger Sweep Complete! Finding Sensing Point...")

                filepath = os.path.join(self.directory, filename)

                df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

                plunger_data = df.iloc[:, 0]

                current_data = df.iloc[:, -1]

                filename = filename.removesuffix('.csv') + ".png"

                conductance_points, (best_point, best_conductance) = extract_max_conductance_points(x_data = plunger_data, 
                                                                  y_data = current_data, 
                                                                  filepath = self.directory, 
                                                                  filename = filename
                                                                 )

        return conductance_points, (best_point, best_conductance)

    def coulomb_diamonds(self, lower_sd_voltages, upper_sd_voltages, lower_plunger_voltages, upper_plunger_voltages, num_points):
        
        '''
        Runs 2D sweeps of relevant sensor plungers and ohmics to produce coulomb diamonds.
        The shapes and sizes of the diamonds are extracted to determine the lever arms of the sensors

        TODO Update the function to automatically detect the S/D bias range for coulomb diamonds

        Paramters
        ---------
        sensor_barrier_voltages : float
            The maximum (startpoint) gate voltages for which the sweep starts 
        
        lower_voltages : float
            The minimum (endpoint) gate voltages for which the sweep ends

        upper_voltages : float
            The maximum (endpoint) gate voltages for which the sweep ends

        num_points : int
            The number of points, inclusive, for the sweeps

        Returns
        -------
        conductance_points, (best_point, best_conductance) : list[float], tuple(float, float)
            All highest sensitivity points for all coulomb blockade peaks, with the best given as (plunger voltage, conductance)
        
        '''

        # First, we get our S/D biases to their lower thresholds

        sd_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Ohmic":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                sd_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        logger.info(f"{sd_dacs_and_vals}")

        # We also get our plunger voltages to set them to their lower voltages

        plunger_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )        

        logger.info(f"{plunger_dacs_and_vals}")

        sd_targets = []

        lower_ohmic_voltages = []
        upper_ohmic_voltages = []

        for i, item in enumerate(lower_sd_voltages):

            lower_ohmic_voltage = item / self.voltage_divider_SET
            upper_ohmic_voltage = upper_sd_voltages[i] / self.voltage_divider_SET

            lower_ohmic_voltages.append(lower_ohmic_voltage)
            upper_ohmic_voltages.append(upper_ohmic_voltage)

        startpoint_itr = iter(lower_ohmic_voltages)

        for gate, dac_and_val in sd_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                sd_val = next(startpoint_itr)

                sparam = SweepParam(
                    parameter = "spi_rack." + dac,
                    start = starting_val,
                    end = sd_val
                )

                sd_targets.append(sparam)

        logger.info(f"{sd_targets}")

        sweep_layer = SweepLayer(
            targets = sd_targets,
            num_points = 100,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Sensor Ohmics to Initial Points...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Sensor Ohmics Set!")

        plunger_targets = []

        startpoint_itr = iter(lower_plunger_voltages)

        for gate, dac_and_val in plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                plunger_val = next(startpoint_itr)

                sparam = SweepParam(
                    parameter = "spi_rack." + dac,
                    start = starting_val,
                    end = plunger_val
                )

                plunger_targets.append(sparam)

        logger.info(f"{plunger_targets}")

        sweep_layer = SweepLayer(
            targets = plunger_targets,
            num_points = 100,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Sensor Plungers to Initial Points...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Sensor Plungers Set!")

        # Now, we make the Coulomb Diamond Sweeps

        sensor_plunger_targets = []

        sensor_ohmic_targets = []

        sensor_plunger_idx = 0

        sensor_ohmic_idx = 0

        sensor_plunger_names = []

        sensor_ohmic_names = []

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                name = self.device_gates[i]["label"]

                sensor_plunger_names.append(name)

                sparam = SweepParam(
                    parameter = p,
                    start = lower_plunger_voltages[sensor_plunger_idx],
                    end = upper_plunger_voltages[sensor_plunger_idx]
                )

                sensor_plunger_targets.append(sparam)      

                sensor_plunger_idx += 1

            elif self.device_gates[i]['type'] == "Sensor Ohmic":

                p = self.device_gates[i]['channel']

                name = self.device_gates[i]["label"]

                sensor_ohmic_names.append(name)

                sparam = SweepParam(
                    parameter = p,
                    start = lower_ohmic_voltages[sensor_ohmic_idx],
                    end = upper_ohmic_voltages[sensor_ohmic_idx]
                )

                sensor_ohmic_targets.append(sparam)      

                sensor_ohmic_idx += 1

        for i, item in enumerate(sensor_plunger_targets):

            sweep_layer_1 = SweepLayer(
                targets = [sensor_ohmic_targets[i]],
                num_points = num_points,
                measurement_time = 0.2
            )

            sweep_layer_2 = SweepLayer(
                targets = [item],
                num_points = num_points,
                measurement_time = 0.2
            )

            measure = lambda ih, sp: (
                    ih.read_buffer([
                        'agilent_left.volt',
                        'agilent_right.volt'
                    ]),
                    ['agilent_left.volt', 'agilent_right.volt']
            )

            sweep = Sweep([sweep_layer_1, sweep_layer_2], measure)

            logger.info("Coulomb Diamond Sweep Starting...")

            logger.info(
                f"Ohmic: {sensor_ohmic_targets[i].parameter}, "
                f"{sensor_ohmic_targets[i].start} -> {sensor_ohmic_targets[i].end}"
            )

            logger.info(
                f"Plunger: {sensor_plunger_targets[i].parameter}, "
                f"{sensor_plunger_targets[i].start} -> {sensor_plunger_targets[i].end}"
            )

            future = self.experiment_handler.do_sweep(sweep = sweep,
                                                      instrument_handler = self.instrument_handler,
                                                      filename = filename
                                                     )
            
            logger.info("Coulomb Diamond Sweep Complete! Finding Diamonds...")


class GlobalChargeTuning(Bootstrapping):

    def __init__(self, device_config, instr_handler, exp_handler):

        super().__init__(self,
                          device_config=device_config,
                          instr_handler = instr_handler,
                          exp_handler = exp_handler
                        )

    def autotune(self, instr_handler, exp_handler, num_points_bootstrapping: list[int], num_points_global_charge_tuning: list[int]):

        super().autotune(instr_handler = instr_handler, 
                         exp_handler = exp_handler, 
                         num_points_list = num_points_bootstrapping
                        )

        """ self.plunger_starting_voltages = [1.1674690763420748]

        self.plunger_ending_voltages = [1.5]

        self.dot_plunger_lower_voltages = [1.2587939698492463 - 0.02, 1.2814070351758793 - 0.02, 1.2587939698492463 - 0.02]

        self.dot_plunger_idle_voltages = [1.2587939698492463, 1.2814070351758793, 1.2587939698492463]

        self.dot_plunger_upper_voltages = [1.2587939698492463 + 0.02, 1.2814070351758793 + 0.02, 1.2587939698492463 + 0.02] """

        self.plunger_crosstalk_vals = self.calibrate_countersweeping(lower_dot_plunger_voltages = self.dot_plunger_lower_voltages, 
                                                                     upper_dot_plunger_voltages = self.dot_plunger_upper_voltages,
                                                                     num_points = num_points_global_charge_tuning[0]
                                                                    )
        
        logger.info(f"{self.plunger_crosstalk_vals}")

        confirmation = self.confirm_charge_transitions(lower_plunger_voltages = self.dot_plunger_lower_voltages, 
                                                       upper_plunger_voltages = [1.5, 1.5, 1.5], 
                                                       plunger_crosstalk_vals = self.plunger_crosstalk_vals, 
                                                       num_points = num_points_global_charge_tuning[1]
                                                      )

        logger.info(f"{confirmation}")

        """ self.dot_plunger_lower_voltages = [1.2587939698492463 - 0.12, 1.2814070351758793 - 0.12, 1.2587939698492463 - 0.12]

        self.plunger_crosstalk_vals = [np.float64(-3.0840343159529215 - 0.5), np.float64(-3.948856914488276 - 0.5), np.float64(-5.174829249744017 - 0.5)] """

        self.plunger_plunger_sweep(lower_plunger_voltages = self.dot_plunger_lower_voltages, 
                                   idle_plunger_voltages = self.dot_plunger_idle_voltages, 
                                   upper_plunger_voltages = [self.abs_max_gate_voltage, self.abs_max_gate_voltage, self.abs_max_gate_voltage], 
                                   plunger_crosstalk_vals = self.plunger_crosstalk_vals, 
                                   num_points = num_points_global_charge_tuning[2]
                                  )

        pass

    def autotune_global_charge_tuning(self, plunger_starting_voltages, plunger_ending_voltages, plunger_lower_voltages, plunger_idle_voltages, plunger_upper_voltages, num_points):

        self.plunger_starting_voltages = plunger_starting_voltages

        self.plunger_ending_voltages = plunger_ending_voltages

        self.dot_plunger_lower_voltages = plunger_lower_voltages

        self.dot_plunger_idle_voltages = plunger_idle_voltages

        self.dot_plunger_upper_voltages = plunger_upper_voltages

        """ self.plunger_starting_voltages = [1.1674690763420748]

        self.plunger_ending_voltages = [1.5]

        self.dot_plunger_lower_voltages = [1.2587939698492463 - 0.02, 1.2814070351758793 - 0.02, 1.2587939698492463 - 0.02]

        self.dot_plunger_idle_voltages = [1.2587939698492463, 1.2814070351758793, 1.2587939698492463]

        self.dot_plunger_upper_voltages = [1.2587939698492463 + 0.02, 1.2814070351758793 + 0.02, 1.2587939698492463 + 0.02] """

        self.plunger_crosstalk_vals = self.calibrate_countersweeping(lower_dot_plunger_voltages = self.dot_plunger_lower_voltages, 
                                                                     upper_dot_plunger_voltages = self.dot_plunger_upper_voltages,
                                                                     num_points = 150
                                                                    )
        
        logger.info(f"{self.plunger_crosstalk_vals}")

        confirmation = self.confirm_charge_transitions(lower_plunger_voltages = self.dot_plunger_lower_voltages, 
                                                       upper_plunger_voltages = [1.5, 1.5, 1.5], 
                                                       plunger_crosstalk_vals = self.plunger_crosstalk_vals, 
                                                       num_points = 400
                                                      )

        logger.info(f"{confirmation}")

        self.dot_plunger_lower_voltages = [1.2587939698492463 - 0.12, 1.2814070351758793 - 0.12, 1.2587939698492463 - 0.12]

        self.plunger_crosstalk_vals = [np.float64(-3.0840343159529215 - 0.5), np.float64(-3.948856914488276 - 0.5), np.float64(-5.174829249744017 - 0.5)]

        self.plunger_plunger_sweep(lower_plunger_voltages = self.dot_plunger_lower_voltages, 
                                   idle_plunger_voltages = self.dot_plunger_idle_voltages, 
                                   upper_plunger_voltages = [1.5, 1.5, 1.5], 
                                   plunger_crosstalk_vals = self.plunger_crosstalk_vals, 
                                   num_points = 200
                                  )

        pass

    def recalibrate_charge_sensors(self, lower_voltages, upper_voltages, num_points):        

        # First, we set our charge sensor plunger gates to their initial values

        charge_sensor_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                charge_sensor_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )
                
        charge_sensor_targets = []

        endpoint_iter = iter(lower_voltages)

        for gate, dac_and_val in charge_sensor_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                sensor_plunger_val = next(endpoint_iter)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = sensor_plunger_val
                )

                charge_sensor_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = charge_sensor_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Charge Sensor Plunger Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Charge Sensor Plunger Voltages Set!")

        # Now, we define the sensor plunger sweeps

        sensor_plunger_targets = []

        sensor_plunger_idx = 0

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = lower_voltages[sensor_plunger_idx],
                    end = upper_voltages[sensor_plunger_idx]
                )

                sensor_plunger_targets.append(sparam)      

                sensor_plunger_idx += 1      
        
        logger.info(f"{sensor_plunger_targets}")

        sensor_plunger_idx = 0

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Sensor Plunger':
                
                sweep_layer = SweepLayer(
                    targets = [sensor_plunger_targets[sensor_plunger_idx]],
                    num_points = num_points,
                    measurement_time = 0.2
                )

                sensor_plunger_idx += 1

                measure = lambda ih, sp: (
                        ih.read_buffer([
                            'agilent_left.volt',
                            'agilent_right.volt'
                        ]),
                        ['agilent_left.volt', 'agilent_right.volt']
                )

                sweep = Sweep([sweep_layer], measure)

                logger.info("Charge Sensor Plunger Sweep Starting...")

                gate_name = self.device_gates[i]['label']

                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{gate_name}_Sweep_{time_str}.csv"

                future = self.experiment_handler.do_sweep(sweep = sweep,
                                                          instrument_handler = self.instrument_handler,
                                                          filename = filename
                                                         )
        
                logger.info("Charge Sensor Plunger Sweep Complete! Finding Sensing Point...")

                filepath = os.path.join(self.directory, filename)

                df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

                plunger_data = df.iloc[:, 0]

                current_data = df.iloc[:, -1]

                filename = filename.removesuffix('.csv') + ".png"

                conductance_points = extract_max_conductance_pair(x_data = plunger_data, 
                                                                  y_data = current_data, 
                                                                  filepath = self.directory, 
                                                                  filename = filename
                                                                 )

        return conductance_points

    def calibrate_countersweeping(self, lower_dot_plunger_voltages, upper_dot_plunger_voltages, num_points): 

        # First, we get the current plunger gate voltages

        dot_plunger_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                dot_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )   

        logger.info(f"{dot_plunger_dacs_and_vals}")

        # Now, we set our dot plungers to their initial values

        dot_plunger_targets = []

        endpoint_iter_dot = iter(lower_dot_plunger_voltages)

        for gate, dac_and_val in dot_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                dot_plunger_val = next(endpoint_iter_dot)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = dot_plunger_val
                )

                dot_plunger_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = dot_plunger_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Dot Plunger Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Dot Plunger Voltages Set!")

        # Now, we recalibrate the charge sensor

        sensing_points = self.recalibrate_charge_sensors(lower_voltages = self.plunger_starting_voltages,
                                                         upper_voltages = self.plunger_ending_voltages,
                                                         num_points = 200
                                                        )
        
        sensor_plunger_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                sensor_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )
        
        logger.info(f"{sensor_plunger_dacs_and_vals}")

        sensor_plunger_targets = []

        sensor_lower_bound = [sensing_points[0] - 0.01]

        sensor_upper_bound = [sensing_points[1] + 0.01]

        endpoint_iter_sensor = iter(sensor_lower_bound)

        for gate, dac_and_val in sensor_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                sensor_plunger_val = next(endpoint_iter_sensor)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = sensor_plunger_val
                )

                sensor_plunger_targets.append(sparam)

        sweep_layer = SweepLayer(
            targets = sensor_plunger_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Charge Sensor to Newly Calibrated Point...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Charge Sensor Calibrated!")

        # Now, we construct 2D scans, in which the Sensor plunger is swept and the dot plungers are stepped and the return sweeps

        dot_plunger_targets = []

        sensor_plunger_targets = []

        dot_plunger_idx = 0

        sensor_plunger_idx = 0

        dot_plunger_names = []

        sensor_plunger_names = []

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                p = self.device_gates[i]['channel']

                gate_name = self.device_gates[i]['label']

                dot_plunger_names.append(gate_name)

                sparam = SweepParam(
                    parameter = p,
                    start = lower_dot_plunger_voltages[dot_plunger_idx],
                    end = upper_dot_plunger_voltages[dot_plunger_idx]
                )

                dot_plunger_targets.append(sparam)

                dot_plunger_idx += 1

            elif self.device_gates[i]['type'] == 'Sensor Plunger':

                p = self.device_gates[i]['channel']

                gate_name = self.device_gates[i]['label']

                sensor_plunger_names.append(gate_name)

                sparam = SweepParam(
                    parameter = p,
                    start = sensor_lower_bound[sensor_plunger_idx],
                    end = sensor_upper_bound[sensor_plunger_idx]
                )

                sensor_plunger_targets.append(sparam)

                sensor_plunger_idx += 1

        # Here we define the return sweeps

        dot_return_targets = []

        sensor_return_targets = []

        endpoint_iter_dot = iter(upper_dot_plunger_voltages)

        endpoint_iter_sensor = iter(sensor_upper_bound)

        startpoint_iter_sensor = iter(sensor_lower_bound)

        for gate, dac_and_val in dot_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                dot_plunger_val = next(endpoint_iter_dot)

                sparam = SweepParam(
                    parameter = p,
                    start = dot_plunger_val,
                    end = starting_val
                )

                dot_return_targets.append(sparam)            

        for gate, dac_and_val in sensor_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                start_sensor_plunger_val = next(endpoint_iter_sensor)

                end_sensor_plunger_val = next(startpoint_iter_sensor)

                sparam = SweepParam(
                    parameter = p,
                    start = start_sensor_plunger_val,
                    end = end_sensor_plunger_val
                )

                sensor_return_targets.append(sparam)

        # Now, we run the sweeps

        crosstalk_vals = []

        for i, sensor in enumerate(sensor_plunger_targets):

            sensor_layer = SweepLayer(targets = [sensor],
                                      num_points = num_points,
                                      measurement_time = 0.2
                                     )

            sensor_name = sensor_plunger_names[i]

            for j, dot in enumerate(dot_plunger_targets):

                dot_layer = SweepLayer(targets = [dot],
                                       num_points = 20,
                                       measurement_time = 0.2
                                      )

                dot_name = dot_plunger_names[j]

                measure = lambda ih, sp: (
                        ih.read_buffer([
                            'agilent_left.volt',
                            'agilent_right.volt'
                        ]),
                        ['agilent_left.volt', 'agilent_right.volt']
                )

                sweep = Sweep([dot_layer, sensor_layer], measure)

                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{sensor_name}_{dot_name}_Scan_{time_str}.csv"

                logger.info(f"Determining coupling between {sensor_name} and {dot_name}...")

                future = self.experiment_handler.do_sweep(sweep = sweep,
                                                          instrument_handler = self.instrument_handler,
                                                          filename = filename
                                                         )
        
                logger.info("Scan Complete! Finding cross-talk coefficient...")

                filepath = os.path.join(self.directory, filename)

                df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

                sensor_data = df.iloc[:, 1]
                dot_data = df.iloc[:, 0]

                current_data = df.iloc[:, -1]

                filename = filename.removesuffix('.csv') + ".png"

                slope, intercept = hough_transform(x_data = sensor_data,
                                        y_data = dot_data,
                                        current_data = current_data,
                                        filepath = self.directory,
                                        filename = filename
                                       )
                
                crosstalk_vals.append(slope)

                logger.info("Returning plungers to the original points...")

                return_layer = SweepLayer(targets = [dot_return_targets[j], sensor_return_targets[i]],
                                          num_points = num_points,
                                          measurement_time = 0.1
                                         )

                measure = lambda ih, sp: (
                        ih.read_buffer([
                            'agilent_left.volt',
                            'agilent_right.volt'
                        ]),
                        ['agilent_left.volt', 'agilent_right.volt']
                )

                sweep = Sweep([return_layer], measure)

                future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                           instrument_handler = self.instrument_handler
                                                                          )

        return crosstalk_vals

    def confirm_charge_transitions(self, lower_plunger_voltages, upper_plunger_voltages, plunger_crosstalk_vals, num_points):

        # First, we get the current dot plunger gate voltages

        dot_plunger_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                dot_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )   

        logger.info(f"{dot_plunger_dacs_and_vals}")

        # Now, we set our dot plungers to their initial values
                
        dot_plunger_targets = []

        endpoint_iter_dot = iter(lower_plunger_voltages)

        for gate, dac_and_val in dot_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                dot_plunger_val = next(endpoint_iter_dot)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = dot_plunger_val
                )

                dot_plunger_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = dot_plunger_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Dot Plunger Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Dot Plunger Voltages Set!")

        # Now, we recalibrate our charge sensor

        sensing_points = self.recalibrate_charge_sensors(lower_voltages = self.plunger_starting_voltages,
                                                         upper_voltages = self.plunger_ending_voltages,
                                                         num_points = 200
                                                        )
        
        logger.info(f"Sensing Points: {sensing_points}")

        sensor_plunger_dacs_and_vals = {}

        sensor_parameters = []

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                sensor_parameters.append(p)

                instr, param = p.split('.', 1)

                sensor_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        sensor_plunger_targets = []

        sensing_point = sensing_points[0]

        logger.info(f"Sensing Point: {sensing_point}")

        sensing_point_list = [sensing_point]

        endpoint_iter_sensor = iter(sensing_point_list)

        for gate, dac_and_val in sensor_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                sensor_plunger_val = next(endpoint_iter_sensor)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = sensor_plunger_val
                )

                sensor_plunger_targets.append(sparam)

        sweep_layer = SweepLayer(
            targets = sensor_plunger_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Charge Sensor to Newly Calibrated Point...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Charge Sensor Calibrated!")

        # Now, we determine the voltage ranges over which our sensor plungers wil be counterswept

        plunger_crosstalk_vals = [np.float64(-3.0840343159529215 - 0.5), np.float64(-3.948856914488276 - 0.5), np.float64(-5.174829249744017 - 0.5)]

        final_sensor_voltages = []

        for i, slope in enumerate(plunger_crosstalk_vals):

            dot_step = (upper_plunger_voltages[i] - lower_plunger_voltages[i]) / num_points

            sensor_step = dot_step / slope

            sensor_endpoint = sensing_point + (sensor_step * num_points)

            logger.info(f"sensor endpoint: {sensor_endpoint} type: {type(sensor_endpoint)}")

            final_sensor_voltages.append(float(sensor_endpoint.item()))

        logger.info(f"Final Sensor Voltages: {final_sensor_voltages}")

        # Now, we sweep our dot plungers and read the SET current to determine if we can sense charge transitions

        confirmations = []

        dot_plunger_idx = 0

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                p = self.device_gates[i]['channel']

                gate_name = self.device_gates[i]['label']

                sparam = SweepParam(
                    parameter = p,
                    start = lower_plunger_voltages[dot_plunger_idx],
                    end = upper_plunger_voltages[dot_plunger_idx]
                )

                logger.info(f"sparam: {sparam}")

                sensorparam = SweepParam(
                    parameter = sensor_parameters[0],
                    start = sensing_point,
                    end = final_sensor_voltages[dot_plunger_idx]
                )

                logger.info(f"sensorparam: {sensorparam}")

                sweep_layer = SweepLayer(
                targets = [sparam, sensorparam],
                num_points = num_points,
                measurement_time = 0.2
                )

                measure = lambda ih, sp: (
                        ih.read_buffer([
                            'agilent_left.volt',
                            'agilent_right.volt'
                        ]),
                        ['agilent_left.volt', 'agilent_right.volt']
                )

                sweep = Sweep([sweep_layer], measure)

                logger.info(f"Confirming Charge Transition detection for {gate_name}...")

                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{gate_name}_Sweep_{time_str}.csv"

                future = self.experiment_handler.do_sweep(sweep = sweep,
                                                          instrument_handler = self.instrument_handler,
                                                          filename = filename
                                                         )

                logger.info(f"{gate_name} Sweep Complete! Confirming Transition Detection...")

                filepath = os.path.join(self.directory, filename)

                df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

                plunger_data = df.iloc[:, 0]

                current_data = df.iloc[:, -1]

                filename = filename.removesuffix('.csv') + ".png"

                conductance_points = extract_max_conductance_points(x_data = plunger_data, 
                                                                    y_data = current_data, 
                                                                    filepath = self.directory, 
                                                                    filename = filename
                                                                   )

                confirmations.append(1)

                # Now, we define our return sweeps and reset our plunger gates

                return_sparam = SweepParam(
                    parameter = p,
                    start = upper_plunger_voltages[dot_plunger_idx],
                    end = lower_plunger_voltages[dot_plunger_idx]
                )

                return_sensorparam = SweepParam(
                    parameter = sensor_parameters[0],
                    start = final_sensor_voltages[dot_plunger_idx],
                    end = sensing_point
                )

                dot_plunger_idx += 1

                return_sweep_layer = SweepLayer(
                targets = [return_sparam, return_sensorparam],
                num_points = num_points,
                measurement_time = 0.2
                )

                measure = lambda ih, sp: (
                        ih.read_buffer([
                            'agilent_left.volt',
                            'agilent_right.volt'
                        ]),
                        ['agilent_left.volt', 'agilent_right.volt']
                )

                sweep = Sweep([return_sweep_layer], measure)

                logger.info("Returning to starting voltages...")

                future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                           instrument_handler = self.instrument_handler)


        return confirmations

    def tune_lead_dot_tunneling(self, lower_barrier_voltages, upper_barrier_voltages, lower_plunger_voltages, upper_plunger_voltages, num_points):
        
        # First, get the lead-barrier voltages, and the dot plunger voltages

        outer_plunger_dacs_and_vals = {}

        lead_barrier_dacs_and_vals = {}

        plunger_idx = 0

        barrier_idx = 0

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                if plunger_idx == 0 or plunger_idx == 2:

                    p = self.device_gates[i]['channel']

                    instr, param = p.split('.', 1)

                    outer_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                                instr,
                                                param,
                                                wait=True
                                                )   
            
                plunger_idx += 1 

            elif self.device_gates[i]['type'] == 'Dot Barrier':

                if barrier_idx == 0 or barrier_idx == 3:

                    p = self.device_gates[i]['channel']

                    instr, param = p.split('.', 1)

                    lead_barrier_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                                instr,
                                                param,
                                                wait=True
                                                )
                
                barrier_idx +=1

        logger.info(f"{outer_plunger_dacs_and_vals}")

        logger.info(f"{lead_barrier_dacs_and_vals}")

        # Now, we set our plungers and barriers to their starting values

        plunger_targets = []

        barrier_targets = []

        endpoint_iter_plunger = iter(lower_plunger_voltages)

        endpoint_iter_barrier = iter(lower_barrier_voltages) 

        for gate, dac_and_val in outer_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                plunger_val = next(endpoint_iter_plunger)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = plunger_val
                )

                plunger_targets.append(sparam)            

        for gate, dac_and_val in lead_barrier_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                barrier_val = next(endpoint_iter_barrier)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = barrier_val
                )

                barrier_targets.append(sparam)  

        sweep_layer = SweepLayer(
            targets = plunger_targets + barrier_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Lead Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Lead Voltages Set!")

        # Now, we create our sweeps

        plunger_targets = []

        barrier_targets = []

        plunger_idx = 0

        barrier_idx = 0 

        plunger_names = []

        barrier_names = []

        startpoint_iter_plunger = iter(lower_plunger_voltages)

        endpoint_iter_plunger = iter(upper_plunger_voltages)

        startpoint_iter_barrier = iter(lower_barrier_voltages)

        endpoint_iter_barrier = iter(upper_barrier_voltages)

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                if plunger_idx == 0 or plunger_idx == 2:

                    p = self.device_gates[i]['channel']

                    gate_name = self.device_gates[i]['label']

                    plunger_names.append(gate_name)

                    start_val = next(startpoint_iter_plunger)

                    end_val = next(endpoint_iter_plunger)

                    sparam = SweepParam(
                        parameter = p,
                        start = start_val,
                        end = end_val
                    )

                    plunger_targets.append(sparam)

                plunger_idx += 1    

            elif self.device_gates[i]['type'] == 'Dot Barrier':

                if barrier_idx == 0 or barrier_idx == 3:

                    p = self.device_gates[i]['channel']

                    gate_name = self.device_gates[i]['label']

                    barrier_names.append(gate_name)

                    start_val = next(startpoint_iter_barrier)

                    end_val = next(endpoint_iter_barrier)

                    sparam = SweepParam(
                        parameter = p,
                        start = start_val,
                        end = end_val
                    )

                    barrier_targets.append(sparam)

                plunger_idx += 1

        barrier_setpoints = []

        for i, item in enumerate(plunger_targets):

            plunger_layer = SweepLayer(
            targets = [item],
            num_points = num_points,
            measurement_time = 0.2
            )

            barrier_layer = SweepLayer(
            targets = [barrier_targets[i]],
            num_points = num_points,
            measurement_time = 0.2
            )

            measure = lambda ih, sp: (
                    ih.read_buffer([
                        'agilent_left.volt',
                        'agilent_right.volt'
                    ]),
                    ['agilent_left.volt', 'agilent_right.volt']
            )

            sweep = Sweep([barrier_layer, plunger_layer], measure)

            logger.info(f"{plunger_names[i]} vs. {barrier_names[i]} scan starting...")

            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{plunger_names[i]}_{barrier_names[i]}_Scan_{time_str}.csv"

            future = self.experiment_handler.do_sweep(sweep = sweep,
                                                        instrument_handler = self.instrument_handler,
                                                        filename = filename
                                                        )
            
            logger.info(f"{plunger_names[i]} vs. {barrier_names[i]} scan complete! Finding appropriate barrier voltage...")

        return barrier_setpoints
    
    def plunger_plunger_sweep(self, lower_plunger_voltages, idle_plunger_voltages, upper_plunger_voltages, plunger_crosstalk_vals, num_points):
        
        # First, we get the current dot plunger gate voltages

        dot_plunger_dacs_and_vals = {}

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                dot_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )   

        logger.info(f"{dot_plunger_dacs_and_vals}")

        # Now, we set our dot plungers to their initial values
                
        dot_plunger_targets = []

        endpoint_iter_dot = iter(idle_plunger_voltages)

        for gate, dac_and_val in dot_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                dot_plunger_val = next(endpoint_iter_dot)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = dot_plunger_val
                )

                dot_plunger_targets.append(sparam)            

        sweep_layer = SweepLayer(
            targets = dot_plunger_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Initial Dot Plunger Voltages...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Initial Dot Plunger Voltages Set!")

        # Now, we recalibrate our charge sensor

        sensing_points = self.recalibrate_charge_sensors(lower_voltages = self.plunger_starting_voltages,
                                                         upper_voltages = self.plunger_ending_voltages,
                                                         num_points = 200
                                                        )
        
        logger.info(f"Sensing Points: {sensing_points}")

        sensor_plunger_dacs_and_vals = {}

        sensor_parameters = []

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                sensor_parameters.append(p)

                instr, param = p.split('.', 1)

                sensor_plunger_dacs_and_vals[i] = self.instrument_handler.get_parameter(
                                            instr,
                                            param,
                                            wait=True
                                            )

        sensor_plunger_targets = []

        sensing_point = sensing_points[0]

        logger.info(f"Sensing Point: {sensing_point}")

        sensing_point_list = [sensing_point]

        endpoint_iter_sensor = iter(sensing_point_list)

        for gate, dac_and_val in sensor_plunger_dacs_and_vals.items():
            for dac, starting_val in dac_and_val.items():

                p = "spi_rack." + dac

                sensor_plunger_val = next(endpoint_iter_sensor)

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = sensor_plunger_val
                )

                sensor_plunger_targets.append(sparam)

        sweep_layer = SweepLayer(
            targets = sensor_plunger_targets,
            num_points = num_points,
            measurement_time = 0.1
        )

        measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
        )

        sweep = Sweep([sweep_layer], measure)

        logger.info("Setting Charge Sensor to Newly Calibrated Point...")

        future = self.experiment_handler.set_voltage_configuration(sweep = sweep,
                                                                   instrument_handler = self.instrument_handler)

        logger.info("Charge Sensor Calibrated!")

        # Now, we construct the 2D plunger scans, compensating for the inner layer with the charge sensor

        plunger_targets_dots = []

        plunger_return_dots = []

        plunger_dot_parameters = []

        plunger_dot_names = []

        plunger_idx = 0

        for item in self.gates_to_dacs:

            if self.device_gates[item]['type'] == "Dot Plunger":

                p = self.device_gates[item]['channel']

                plunger_dot_parameters.append(p)

                name = self.device_gates[item]["label"]

                plunger_dot_names.append(name)

                sparam = SweepParam(
                    parameter = p,
                    start = lower_plunger_voltages[plunger_idx],
                    end = upper_plunger_voltages[plunger_idx]
                )

                return_sparam = SweepParam(
                    parameter = p,
                    start = upper_plunger_voltages[plunger_idx],
                    end = idle_plunger_voltages[plunger_idx]
                )

                plunger_targets_dots.append(sparam)

                plunger_return_dots.append(return_sparam)

                plunger_idx += 1

        logger.info(f"{plunger_targets_dots}")
        logger.info(f"{plunger_return_dots}")
        logger.info(f"{plunger_dot_parameters}")
        logger.info(f"{plunger_dot_names}")

        # Here, we define the sensor compensation sweeps

        final_sensor_voltages = []

        for i, slope in enumerate(plunger_crosstalk_vals):

            dot_step = (upper_plunger_voltages[i] - lower_plunger_voltages[i]) / num_points

            sensor_step = dot_step / slope

            sensor_endpoint = sensing_point + (sensor_step * num_points)

            logger.info(f"sensor endpoint: {sensor_endpoint} type: {type(sensor_endpoint)}")

            final_sensor_voltages.append(float(sensor_endpoint.item()))

        logger.info(f"Final Sensor Voltages: {final_sensor_voltages}")

        plunger_targets_sensors = []

        plunger_return_sensors = []

        plunger_idx = 0

        for item in self.gates_to_dacs:

            if self.device_gates[item]['type'] == "Dot Plunger":

                p = sensor_parameters[0]

                sparam = SweepParam(
                    parameter = p,
                    start = sensing_point,
                    end = final_sensor_voltages[plunger_idx]
                )

                return_sparam = SweepParam(
                    parameter = p,
                    start = final_sensor_voltages[plunger_idx],
                    end = sensing_point
                )

                plunger_targets_sensors.append(sparam)
                
                plunger_return_sensors.append(return_sparam)

                plunger_idx += 1

        logger.info(f"{plunger_targets_sensors}")
        logger.info(f"{plunger_return_sensors}")

        # Now, we perform the plunger-plunger scans

        for i, (first, second, ret_first, ret_second) in enumerate(zip(plunger_targets_dots, 
                                                                       plunger_targets_dots[1:], 
                                                                       plunger_return_dots, 
                                                                       plunger_return_dots[1:]
                                                                       )
                                                                  ):

            # Here, we set the involved dot plungers to the lower plunger value from their idle values

            lower_param_1 = SweepParam(
                    parameter = plunger_dot_parameters[i],
                    start = idle_plunger_voltages[i],
                    end = lower_plunger_voltages[i]
                )

            lower_param_2 = SweepParam(
                    parameter = plunger_dot_parameters[i + 1],
                    start = idle_plunger_voltages[i + 1],
                    end = lower_plunger_voltages[i + 1]
                )

            lower_layer = SweepLayer(
                targets = [lower_param_1, lower_param_2],
                num_points = num_points,
                measurement_time = 0.2
            )

            measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
            )

            lower_sweep = Sweep([lower_layer], measure)

            logger.info("Setting Plungers to their initial sweep values...")

            future = self.experiment_handler.set_voltage_configuration(sweep = lower_sweep,
                                                                       instrument_handler = self.instrument_handler)
            
            logger.info("Initial Sweep Voltages Set!")

            # Now, we perform the actual sweep

            sweep_layer_1 = SweepLayer(
                targets = [first],
                num_points = num_points,
                measurement_time = 0.2
            )

            sweep_layer_2 = SweepLayer(
                targets = [second],
                num_points = num_points,
                measurement_time = 0.2
            )

            measure = lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
            )

            sweep = Sweep([sweep_layer_2, sweep_layer_1], measure)

            logger.info("Dot Plunger-Plunger Scan Starting...")

            gate_name_1 = plunger_dot_names[i]
            gate_name_2 = plunger_dot_names[i + 1]

            gates = [gate_name_1, gate_name_2]

            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{gate_name_1}_{gate_name_2}_Scan_{time_str}.csv"

            future = self.experiment_handler.do_sweep(sweep = sweep,
                                                      instrument_handler = self.instrument_handler,
                                                      filename = filename)

            # Here, we find the set points for the dot barrier-barrier scans

            logger.info("Dot Plunger Scan Complete! Plotting CSD...")

            filepath = os.path.join(self.directory, filename)

            df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

            lb_data = df.iloc[:, 0]
            rb_data = df.iloc[:, 1]

            current_data = df.iloc[:, -2]

            filename = filename.removesuffix('.csv') + ".png"

            # Now, we reset the barriers back to their starting voltages

            return_layer = SweepLayer(
                targets = [ret_first, ret_second],
                num_points = num_points,
                measurement_time = 0.1
            )

            return_sweep = Sweep([return_layer], measure)

            logger.info("Returning Plungers to starting values...")

            future = self.experiment_handler.set_voltage_configuration(sweep = return_sweep,
                                                                       instrument_handler = self.instrument_handler)
            

class VirtualGating(GlobalChargeTuning):

    def __init__(self, device_config, instr_handler, exp_handler):
        
        super().__init__(device_config = device_config, 
                         instr_handler = instr_handler, 
                         exp_handler = exp_handler
                        ) 

    def autotune(self, instr_handler, exp_handler, num_points_bootstrapping: list[int], num_points_global_charge_tuning: list[int], num_points_virtual_gating: list[int]):

        super().autotune(instr_handler = instr_handler, 
                         exp_handler = exp_handler, 
                         num_points_bootstrapping = num_points_bootstrapping,
                         num_points_global_charge_tuning = num_points_global_charge_tuning
                        )

        pass

    def autotune_virtual_gating(self):

        pass

    def lever_arm_matrix():
        
        pass

class ChargeStateTuning(VirtualGating):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def determine_charge_states():
        pass

class QubitTuning(ChargeStateTuning):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def rabi_oscilations():
        pass

