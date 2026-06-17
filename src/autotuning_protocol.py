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

from datetime import datetime
import threading
from experiment_base import SweepParam, SweepLayer, Sweep
from data_analysis import extract_turn_on_voltage, extract_pinch_off_curve_ranges, extract_working_point, extract_max_conductance_points, extract_lever_arms

import sys

from nicegui import ui
from tunerlog import TunerLog

logger = TunerLog('Autotuning Protocol')

class Protocol:

    def __init__(self, 
                 device_config,
                 instrument_handler=None,
                 experiment_handler=None
    ):
        
        '''
        Initializes the protocol. Reads the device configuration file provided and creates a path from gate name to dac.

        
        '''

        self.instrument_handler = instrument_handler
        self.experiment_handler = experiment_handler

        # First, we load in the config file

        logger.info("Loading Device Config file...")

        self._load_config_file(device_config)

        self.directory = r"C:\Users\BaughLaflamme\Desktop\3d1s_W151_1 Measurements\3D1S_w151_1 - Autotuning Tests"

        # Now, we create a dictionary to house a map between gate names and dacs

        self.gates_to_dacs = {}

        for i in self.device_gates:

            self.gates_to_dacs[i] = self.device_gates[i]['channel']

        #print(self.gates_to_dacs)

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
        self.abs_max_current_SET = self.config['device']['constraints']['abs_max_current_SET']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.config['device']['constraints']['abs_max_gate_differential']

        # Equipment Setup

        self.voltage_divider_triple_dot = self.config['setup']['voltage_dividers']['voltage_divider_triple_dot']
        self.voltage_divider_SET = self.config['setup']['voltage_dividers']['voltage_divider_SET']
        self.triple_dot_preamp_bias = self.config['setup']['triple_dot_preamp']['preamp_bias']
        self.triple_dot_preamp_sensitivity = self.config['setup']['triple_dot_preamp']['preamp_sensitivity']
        self.SET_preamp_bias = self.config['setup']['SET_preamp']['preamp_bias']
        self.SET_preamp_sensitivity = self.config['setup']['SET_preamp']['preamp_sensitivity']


class Bootstrapping(Protocol):

    def __init__(self, device_config, instrument_handler, experiment_handler):
        
        super().__init__(device_config = device_config, instrument_handler = instrument_handler, experiment_handler = experiment_handler) 

        self.noise_floor = None

        self.ground_device(instr_handler = instrument_handler, exp_handler = experiment_handler)

        self.noise_floor = self.measure_noise_floor()

        logger.info(f"The noise floor is: {self.noise_floor}")

        names = self.instrument_handler.read_buffer(
                ['agilent_left.volt', 'agilent_right.volt'],
            ).keys()
        
        self.means = []

        for i in names:

            mean_name = i + "_mean"
            mean = self.noise_floor[mean_name]

            self.means.append(mean)

        logger.info(f"{self.means}")

        turn_on_voltages = self.turn_on(ohmic_bias = -2e-4,
                                        screening_voltage = 0.2,
                                        gate_voltage = 1.5,
                                        num_points = 200)
        
        logger.info(f"{turn_on_voltages}")

        pinch_off_voltages, saturation_voltages = self.pinch_off(gate_voltage = 1.5,
                                                                 final_voltages = turn_on_voltages,
                                                                 num_points = 200)


        logger.info("Into Barrier-Barrier!")

        logger.info(f"{pinch_off_voltages}")
        logger.info(f"{saturation_voltages}")

        new_sat_voltages = []

        for i in saturation_voltages:

            i += 0.1
            new_sat_voltages.append(i)

        working_point = self.barrier_barrier_sweep(lower_voltages = pinch_off_voltages,
                                                   upper_voltages = new_sat_voltages,
                                                   num_points = 200)
        
        sensor_barrier_voltages = list(working_point)

        plunger_starting_voltages = [sum(working_point) / len(working_point)]

        sensing_point = self.coulomb_blockade_sweep(sensor_barrier_voltages = sensor_barrier_voltages, 
                                    lower_voltages = plunger_starting_voltages, 
                                    upper_voltages = [1.5], 
                                    num_points = 200)

        """ self.coulomb_diamonds(lower_sd_voltages = [-0.01], 
                              upper_sd_voltages = [0.01], 
                              lower_plunger_voltages = [0.7],
                              upper_plunger_voltages = [1.5], 
                              num_points = 100) """

    def ground_device(self, instr_handler, exp_handler):
        
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

        logger.info("Grounding Device...")

        future = exp_handler.set_voltage_configuration(sweep = sweep,
                                                       instrument_handler = instr_handler)

        logger.info("Device Grounded!")

    def measure_noise_floor(self, measurement_time = 30):

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

        logger.info("Device Turn-On Starting...")

        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = "Turn_On_" + time + ".csv"

        future = self.experiment_handler.do_sweep(sweep = sweep,
                                                 instrument_handler = self.instrument_handler,
                                                 filename = filename)

        logger.info("Device Turn-On Sweep Complete! Confirming Turn-On...")

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

            # Get the data from the CSV

            filepath = os.path.join(self.directory, filename)

            df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

            # Get the data for DMMs and for the set voltages, and convert the data from voltage to current, then to nA. We also convert the mean

            turn_on_sweep = df.iloc[:,0].to_numpy()

            triple_dot_data = df.iloc[:,-2].to_numpy() * self.triple_dot_preamp_sensitivity * 1e9

            SET_data = df.iloc[:,-1].to_numpy() * self.SET_preamp_sensitivity * 1e9

            current_means = []

            for mean in self.means:

                mean *= self.SET_preamp_sensitivity * 1e9

                current_means.append(mean)

            current_data = [triple_dot_data, SET_data] 

            turn_on_voltages = []

            turn_on_filenames = ['Triple_Dot_Turn_On' + time + '.png', 'SET_Turn_On' + time + '.png']

            for i, mean in enumerate(self.means):

                turnon_voltage, fig = extract_turn_on_voltage(x_data = turn_on_sweep,
                                                               y_data = current_data[i],
                                                               noisefloor = current_means[i],
                                                               filepath = self.directory,
                                                               filename = turn_on_filenames[i],
                                                               plot_results = True)

                turn_on_voltages.append(turnon_voltage)

            return turn_on_voltages

        else:

            logger.info("""
                        Turn-On confirmation failed for one or more channels! 
                        Please confirm that all gates needed for Turn-On are being swept,
                        otherwise the device does not turn-on for one or more channels within the given maximum voltage.
                        """)
            
            self.ground_device()

            return None

    def pinch_off(self, gate_voltage, final_voltages, num_points):

        """ 
        We first construct a loop to pinch-off each individual finger gate other than the barrier gates, 
        which we'll do the same for after adjusting the other gates.
        
        """

        excluded_types = ["Dot Ohmic", "Sensor Ohmic", "Dot Screening", "Sensor Screening", "Central Screening", "Dot Barrier", "Sensor Barrier"]

        triple_dot_turn_on, SET_turn_on = final_voltages

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

        saturation_voltages = [1.225, 1.225, 1.225, 1.225, 1.15, 1.15, 1.15, 1.15]

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

        self.SET_current_check(minimum_current = 2, maximum_current = 4)

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

        """
        Sweeps a single gate to determine pinch-off.

        Returns a (pinch_off_voltage, saturation_voltage) tuple on success, or (None, None) on failure.
        """

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

            pinch_off_window, fig = extract_pinch_off_curve_ranges(
                x_data = pinch_off_sweep,
                y_data = data[noise_floor_idx],
                noisefloor = self.means[noise_floor_idx],
                filepath = self.directory,
                filename = filename2
            )

            logger.info(f"{pinch_off_window}")

            return pinch_off_window

        else:

            logger.info(f"{gate_name} did not pinch off at {final_voltage} V. Pinch-Off Failed. Returning None...")

            return (None, None)

    def SET_current_check(self, minimum_current, maximum_current):

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

        logger.info(f"{current_means}")

        # Here, we get the current accumulation voltages

        included_types = ['Dot Accumulation', 'Sensor Accumulation']

        accumulation_voltages = {}

        for i in self.gates_to_dacs:

            logger.info("For Loop!")

            if self.device_gates[i]['type'] in included_types:

                p = self.device_gates[i]['channel']

                instr, param = p.split('.', 1)

                logger.info(f"{p}")

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

                    logger.info("for loop 2!")

                    p = self.device_gates[key]['channel']

                    instr, param = p.split('.', 1)

                    logger.info(f"p: {p}")
                    logger.info(f"instr: {instr}, param: {param}")
                    logger.info(f"val: {val}")

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

                time.sleep(2)
        
            elif SET_current_level < minimum_current:

                logger.info("Current too low! Increasing Accumulation Gate Voltages by 1 mV...")

                # We increase the voltages on the accumulation gates by 1 mV

                for key, val in accumulation_voltages.items():

                    logger.info("for loop 2!")

                    p = self.device_gates[key]['channel']

                    instr, param = p.split('.', 1)

                    logger.info(f"p: {p}")
                    logger.info(f"instr: {instr}, param: {param}")
                    logger.info(f"val: {val}")

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

                time.sleep(2)
        
        logger.info(f"SET Current Level: {SET_current_level}")

    def barrier_barrier_sweep(self, lower_voltages, upper_voltages, num_points):

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

                logger.info(f"{p}")

                end_val = next(endpoint_iter)

                logger.info(f"{end_val}")

                sparam = SweepParam(
                    parameter = p,
                    start = starting_val,
                    end = end_val
                )

                logger.info(f"{sparam}")

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

        self.SET_current_check(minimum_current = 2, maximum_current = 4)

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

        best_points = []

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

            logger.info("file defined...")

            best_point, set_points, perp_traces, fig = extract_working_point(lb_data = lb_data,
                                                                 rb_data = rb_data,
                                                                 current_data = current_data,
                                                                 gates = gates,
                                                                 DotTuning = "Triple Dot",
                                                                 barrier_pinch_offs = [lower_voltages[i], lower_voltages[i + 1]],                          
                                                                 filepath = self.directory,
                                                                 filename = filename
                                                                )

            best_points.append(best_point)

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

        logger.info(f"Best Points: {best_points}")

        dot_barrier_voltages = []

        dot_barrier_voltages = [best_points[0][0]]
        
        for left, right in zip(best_points[:-1], best_points[1:]):
            
            dot_barrier_voltages.append((left[1] + right[0]) / 2)
        
        dot_barrier_voltages.append(best_points[-1][1])

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

        self.SET_current_check(minimum_current = 1, maximum_current = 3)

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

            lb_data = df.iloc[:, 0]
            rb_data = df.iloc[:, 1]

            current_data = df.iloc[:, -1]

            filename = filename.removesuffix('.csv') + ".png"

            best_point, working_points, perp_traces, fig = extract_working_point(lb_data = lb_data,
                                                                 rb_data = rb_data,
                                                                 current_data = current_data,
                                                                 gates = gates,
                                                                 DotTuning = "SET",
                                                                 barrier_pinch_offs = [lower_voltages[-i -2], lower_voltages[-i - 1]],                          
                                                                 filepath = self.directory,
                                                                 filename = filename
                                                                )

            logger.info(f"Best: {best_point}")
            logger.info(f"{working_points}")

            return best_point

    def coulomb_blockade_sweep(self, sensor_barrier_voltages, lower_voltages, upper_voltages, num_points):

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
            num_points = 20,
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
                    targets = sensor_plunger_targets[sensor_plunger_idx],
                    num_points = num_points,
                    measurement_time = 0.05
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

        for i in self.device_gates:

            if self.device_gates[i]['type'] == "Sensor Plunger":

                gate_name = self.device_gates[i]['label']

                time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{gate_name}_{time_str}.csv"

                filepath = os.path.join(self.directory, filename)

                df = pd.read_csv(filepath, delimiter = ",", header = None, skiprows = 1)

                plunger_data = df.iloc[:, 0]

                current_data = df.iloc[:, -1]

                filename = filename.removesuffix('.csv') + ".png"

                conductance_points = extract_max_conductance_points(x_data = plunger_data, y_data = current_data)

        logger.info(f"{conductance_points}")

    def coulomb_diamonds(self, lower_sd_voltages, upper_sd_voltages, lower_plunger_voltages, upper_plunger_voltages, num_points):
        
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

        for i in self.gates_to_dacs:
            
            if self.device_gates[i]['type'] == "Sensor Plunger":

                p = self.device_gates[i]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = upper_plunger_voltages[sensor_plunger_idx],
                    end = lower_plunger_voltages[sensor_plunger_idx]
                )

                sensor_plunger_targets.append(sparam)      

                sensor_plunger_idx += 1

            elif self.device_gates[i]['type'] == "Sensor Ohmic":

                p = self.device_gates[i]['channel']

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
                targets = [sensor_plunger_targets[i]],
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
                                                    instrument_handler = self.instrument_handler)
            
            logger.info("Coulomb Diamond Sweep Complete! Finding Diamonds...")

class GlobalChargeTuning(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def confirm_charge_transitions(self, lower_plunger_voltages, upper_plunger_voltages, num_points):

        # First, we create our plunger sweeps

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

        # Now, we sweep our plungers and read the SET current to determine if we can sense charge transitions

        sensor_plunger_idx = 0

        for i in self.gates_to_dacs:

            if self.device_gates[i]['type'] == 'Dot Plunger':

                p = self.device_gates[i]['channel']

                sparam = SweepParam(
                    parameter = p,
                    start = lower_plunger_voltages[sensor_plunger_idx],
                    end = upper_plunger_voltages[sensor_plunger_idx]
                )

                sweep_layer = SweepLayer(
                targets = [sparam],
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

                logger.info("Setting Sensor Plungers to Initial Points...")



                future = self.experiment_handler.do_sweep(sweep = sweep,
                                                          instrument_handler = self.instrument_handler,
                                                          filename = filename
                                                         )

                logger.info("Sensor Plungers Set!")

        return

    def tune_lead_dot_tunneling(self):
        
        # First, we need to 

        return


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

class QubitTuning(Protocol):

    def __init__(self, device_config):
        super().__init__(device_config = device_config) 

    def rabi_oscilations():
        pass

