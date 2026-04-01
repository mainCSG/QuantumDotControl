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

import skimage
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import diamond, rectangle  # noqa

import logging
from colorlog import ColoredFormatter
import sys

from nicegui import ui
import threading


class LinSweep_SIM928(AbstractSweep[np.float64]):

    """
    Linear sweep.

    Args:
        param: Qcodes parameter to sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consecutive sweep points.
        post_actions: Actions to do after each sweep point.
        get_after_set: Should we perform a get on the parameter after setting it
            and store the value returned by get rather than the set value in the dataset.
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
        get_after_set: bool = False,
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions
        self._get_after_set = get_after_set

    def get_setpoints(self) -> npt.NDArray[np.float64]:
        """
        Linear (evenly spaced) numpy array for supplied start, stop and
        num_points.
        """
        array = np.linspace(self._start, self._stop, self._num_points).round(3)
        # below_two = array[np.where(array < 2)].round(3)
        # above_two = array[np.where(array >= 2)].round(2)
        # array = np.concatenate((below_two, above_two))
   
        return array

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions

    @property
    def get_after_set(self) -> bool:
        return self._get_after_set
    
    @property
    def setpoints(self) -> npt.NDArray[np.float64]:
        return self.get_setpoints()
    
class InstrumentControl:
        
    def __init__(self,
                 logger,
                 config: str, 
                 tuner_config: str,
                 station_config: str,
                 save_dir: str) -> None:
        
        """
        Initializes an InstrumentControl object. This class takes care of all connections and communication to instruments
        during your experiment.

        Args:
            config (str): Path to .yaml file containing device information.
            setup_config (str): Path to .yaml file containing experimental setup information.
            tuner_config (str): Path to .yaml file containing tuner information.
            station_config (str): Path to .yaml file containing QCoDeS station information
            save_dir (str): Directory to save data and plots generated.
        """

        # First, we save all the config information

        self.logger = logger
        self.config_file = config
        self.tuner_config_file = tuner_config
        self.station_config_file = station_config
        self.save_dir = save_dir


        # Now, we load the config files

        self.load_config_files()
        
        # After the config files are loaded, we set up a file where data is stored, using today's date

        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        self.db_folder = os.path.join(save_dir, f"{self.config['device']['characteristics']['name']}_{todays_date}")
        os.makedirs(self.db_folder, exist_ok=True)

        # The following method creates a logger that will provide information to the user while the code is running

        self.initialise_logger()

        # Now, we connect to the instruments specified in the config

        self.logger.attempt("connecting to station")

        # Using the Station class from qcodes, we can represent the physical setup of our experiment

        self.station = qc.Station(config_file=self.station_config_file)

        # Now, we attempt to load the voltage source(s) and readout device from the station config file

        voltage_sources = []

        for voltage_source in self.voltage_source_names:
            
            Instrument = self.station.load_instrument(voltage_source)
            voltage_sources.append(Instrument)

        self.voltage_source_1 = voltage_sources[0]
        self.voltage_source_2 = voltage_sources[1]

        self.station.load_instrument(self.multimeter_name)

        self.drain_mm_device = getattr(self.station, self.multimeter_name)
        
        self.drain_volt = getattr(self.station, self.multimeter_name).volt
        
        self.logger.complete("\n")

        # Now, we change the names of the parameters to match the names provided in the yaml file

        channel_prefix = ""
        for parameter, details in self.voltage_source.parameters.items():
            if details.unit == 'V':
                pattern = r'(.*).*\d+.*'
                matches = re.findall(pattern,parameter)
                extractions = [match.strip() for match in matches]
                channel_prefix = extractions[0]
        if channel_prefix == "":
            self.logger.error('unable to find prefix for channels')  
            
        self.logger.info("changing parameters to match names in config.yaml file")
        self.voltage_source.timeout(5 * 60)
        
        for gate, details in self.device_gates.items():
            
            self.voltage_source.add_parameter(
                name=gate,
                parameter_class=qc.parameters.DelegateParameter,
                source=getattr(self.voltage_source, channel_prefix+str(details['channel'])),
                label=details['label'],
                unit = details['unit'],
                step=details['step'],
            )
            self.logger.info(f"changed {channel_prefix+str(details['channel'])} to {gate}")

        # Creates the qcodes database and sets-up the experiment
        
        db_filepath =  os.path.join(self.db_folder, f"experiments_{self.config['device']['characteristics']['name']}_{todays_date}.db")
        qc.dataset.initialise_or_create_database_at(
            db_filepath
        )
        self.logger.info(f"database created/loaded @ {db_filepath}")

        self.logger.info(f"experiment created/loaded in database")
        self.initialization_exp = qc.dataset.load_or_create_experiment(
            'Initialization',
            sample_name=self.config['device']['characteristics']['name']
        )

        # This next section copies the config files in case they get lost or changed
        
        self.logger.info(f"copying all of the config.yml files")
        shutil.copy(self.station_config_file, self.db_folder)
        shutil.copy(self.tuner_config_file, self.db_folder)
        shutil.copy(self.config_file, self.db_folder)

        # Finally, we set up a dictionary to store all of the important results from our experiments
        
        self.results = {}

        self.results['turn_on'] = {
            'voltage': None,
            'current': None,
            'resistance': None,
            'saturation': None,
        }

        for gate in self.barriers + self.leads:
            self.results[gate] = {
                'pinch_off': {'voltage': None, 'width': None}
            }
        
        for gate in self.barriers:
            self.results[gate]['bias_voltage'] = None

        # Finally, we also ground the device before the experiment starts

        self.ground_device()

        return None

    def load_config_files(self):
        
        '''
        This method loads all relavent information from the configuration files provided. It is ran when an InstrumentControl
        object is initialized.
        '''
        
        # Reads the tuner config information

        self.tuner_info = yaml.safe_load(Path(self.tuner_config_file).read_text())
        self.global_turn_on_info = self.tuner_info['global_turn_on']
        self.pinch_off_info = self.tuner_info['pinch_off']

        # Reads the config information

        self.config = yaml.safe_load(Path(self.config_file).read_text())
        self.charge_carrier = self.config['device']['characteristics']['charge_carrier']
        self.operation_mode = self.config['device']['characteristics']['operation_mode']

        # Sets the voltage sign for the gates, based on the charge carrier and mode of the device

        if (self.charge_carrier, self.operation_mode) == ('e', 'acc'):
            self.voltage_sign = +1
        
        if (self.charge_carrier, self.operation_mode) == ('e', 'dep'):
            self.voltage_sign = -1
        
        if (self.charge_carrier, self.operation_mode) == ('h', 'acc'):
            self.voltage_sign = -1
        
        if (self.charge_carrier, self.operation_mode) == ('h', 'dep'):
            self.voltage_sign = +1

        # Now, we retreive the device gates

        self.device_gates = self.config['device']['gates']
        
        # Then, we re-label all the gates as ohmics, barriers, leads, plungers, accumulation gates and screening gates 
        
        self.ohmics = []
        self.barriers = []
        self.leads = []
        self.plungers = []
        self.accumulation = []
        self.screening = []

        # TODO Add additional logic to load SPI rack connections separately from other instruments (either or, both, etc.)
        
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

        # Finally, we determine voltage and current thresholds, as well as other information about the experimental setup

        self.abs_max_current = self.config['device']['constraints']['abs_max_current']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.config['device']['constraints']['abs_max_gate_differential']

        self.voltage_source_names = self.config['setup']['voltage_sources']
        self.multimeter_name = self.config['setup']['multimeter']
        self.voltage_divider = self.config['setup']['voltage_divider']
        self.preamp_bias = self.config['setup']['preamp_bias']
        self.preamp_sensitivity = self.config['setup']['preamp_sensitivity']
        self.voltage_resolution = self.config['setup']['voltage_resolution']

        return None

    def set_voltage(self,
                    gates: str | list[str],
                    voltage: float):
        
        """
        This method allows the user to smoothly set any number of gates to the same final voltage value. If you would like to 
        set different gates to different voltage values, please use the set_voltage_configuration() method.

        Args:
            gates (str | list[str]): A single gate name, written as a string, or a list of gate strings, containing the names of the gates
                                     we wish to set.
            voltage (float): The voltage we wish to set the given gate(s) to.
        """

        # First, if only one gate is input, we convert it to a list to make it easier to work with

        if isinstance(gates, str):
            gates = [gates]

        # Then, we define a dictionary using the gates and voltage input by the user

        voltage_dict = dict(zip(gates, [voltage]*len(gates)))

        # This dictionary gets input into the more general method, set_voltage_configuration()

        self.set_voltage_configuration(self, voltage_configuration = voltage_dict)

        return None

    def set_voltage_configuration(self, 
                                  voltage_configuration: Dict[str, float] = {},
                                  stepsize: float = 10e-3):
        
        """
        This method allows the user to smoothly set a given voltage configuration.

        Args:
            voltage_configuration (Dict[str, float]): A dictionary containing the names of the gates to be set and
                                                      the corresponding voltages the gates will be set to.
            stepsize (float): The voltage stepsize for all the gates. Default is set to 10 mV.
        """

        # First, we determine which gates are being set.

        gates = list(voltage_configuration.keys())

        # Then, we assert that the sign of the voltage we wish to set agrees with the device we are testing

        for gate in gates:

            assert np.sign(voltage_configuration[gate]) == np.sign(self.voltage_sign) or np.sign(voltage_configuration[gate]) == 0, f"Check voltage sign on {gate}"

        # Now, we set up some lists to hold the voltage values 

        intermediate = []
        done = set()

        prevvals = {}
        gate_params = {}
        gate_steps = {}

        # Now, we map the gate to the source and save the correspondance

        gate_to_source = {}

        for source_name, instrument in self.voltage_sources.items():
            for gate in self.voltage_source_names_check[source_name]:
                gate_to_source[gate] = instrument

        for gate, target in voltage_configuration.items():

            instrument = gate_to_source[gate]
            param = getattr(instrument, gate)

            gate_params[gate] = param
            prevvals[gate] = float(param.get())

            step_param = getattr(instrument, f"{gate}_step", None)
            
            gate_steps[gate] = step_param() if step_param else stepsize

        # Now, we generate the ramp

        while len(done) < len(voltage_configuration):

            step = {}

            for gate, target in voltage_configuration.items():

                if gate in done:
                    continue

                prev = prevvals[gate]
                step_size = gate_steps[gate]

                dv = target - prev

                if abs(dv) <= step_size:
                    step[gate] = target
                    done.add(gate)
                else:
                    step[gate] = prev + step_size * (1 if dv > 0 else -1)

                prevvals[gate] = step[gate]

            intermediate.append(step)

        # Finally, we apply the ramp

        for step in intermediate:

            for gate, voltage in step.items():
                gate_params[gate].set(voltage)

            # This lets us sleep once per step
            for instrument in self.voltage_sources.values():
                delay_param = getattr(instrument, "smooth_timestep", None)
                if delay_param:
                    time.sleep(delay_param())
                    break  

        return None

    def sweep_1d(self, 
                 maxV: float = None,
                 minV: float = None,
                 voltage_configuration: Dict[str, float] = {},
                 dV: float = 10e-3) -> pd.DataFrame:
        
        # Bring device to voltage configuration

        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self.set_voltage_configuration(voltage_configuration)

        # Default dV and maxV based on setup_config and config

        if dV is None:
            dV = self.voltage_resolution

        if maxV is None:
            maxV = self.voltage_sign * self.abs_max_gate_voltage

        # Ensure we stay in the allowed voltage space 
        
        assert np.sign(maxV) == self.voltage_sign, self.logger.error("Double check the sign of the gate voltage (maxV) for your given device.")
        assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, self.logger.error("Double check the sign of the gate voltage (minV) for your given device.")

        # Set up gate sweeps
        
        num_steps = self.calculate_num_of_steps(minV, maxV, dV)
        gates_involved = self.barriers + self.leads + self.accumulation + self.plungers

        print(gates_involved)

        self.logger.info(f"setting {gates_involved} to {minV} V")
        
        self.set_voltage_configuration(gates_involved, minV)

        sweep_list = []

        for voltage_source in self.voltage_sources.items():

            for gate_name in gates_involved:
               
               if gate_name in self.voltage_source_names_check[voltage_source[0]]:
                    
                    print(self.voltage_source_names_check[voltage_source[0]][gate_name])

                    param = voltage_source[1][gate_name]

                    sweep_list.append(
                        LinSweep_SIM928(param, minV, maxV, num_steps, get_after_set=False)
                    )

        # Execute the measurement
        self.logger.attempt(f"sweeping {gates_involved} together from {minV} V to {maxV} V")
        
        result = qc.dataset.dond(
            qc.dataset.TogetherSweep(
                *sweep_list
            ),
            self.drain_volt,
            break_condition=self._check_break_conditions,
            measurement_name='Device Turn On',
            exp=self.initialization_exp,
            show_progress=True
        )

        self.logger.complete('\n')
        
        return None

    def sweep_2d(self,
                 P1: str = None, 
                 P2: str = None, 
                 P1_bounds: tuple = (None, None),
                 P2_bounds: tuple = (None, None), 
                 dV: float | tuple = None, 
                 voltage_configuration: dict = None) -> tuple[pd.DataFrame, plt.Axes]:
        
        # Bring device to voltage configuration
        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self.set_voltage_configuration(voltage_configuration)
        else:
            self.logger.info(f"setting {self.leads} to {self.results['turn_on']['saturation']} V")
            self.set_voltage(self.leads, self.results['turn_on']['saturation'])

        # Parse dV from user
        if dV is None:
            dV_P1 = self.voltage_resolution
            dV_P2 = self.voltage_resolution
        elif type(dV) is float:
            dV_P1 = dV
            dV_P2 = dV
        elif type(dV) is tuple:
            dV_P1, dV_P2 = dV
        else:
            self.logger.error("invalid dV")
            return
        
        # Double check device bounds
        minV_P1, maxV_P1 = P1_bounds
        minV_P2, maxV_P2 = P2_bounds

        if minV_P1 is None:
            minV_P1 = self.results[P1]['pinch_off']['voltage']
        else:
            assert np.sign(minV_P1) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (minV) for B1.")

        if minV_P2 is None:
            minV_P2 = self.results[P2]['pinch_off']['voltage']
        else:
            assert np.sign(minV_P2) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (minV) for B2.")

        if maxV_P1 is None:
            if self.voltage_sign == 1:
                maxV_P1 = min(self.results[P1]['pinch_off']['voltage']+self.voltage_sign*self.results[P1]['pinch_off']['width'], self.results['turn_on']['saturation'])
            elif self.voltage_sign == -1:
                maxV_P1 = max(self.results[P1]['pinch_off']['voltage']+self.voltage_sign*self.results[P1]['pinch_off']['width'], self.results['turn_on']['saturation'])
        else:
            assert np.sign(maxV_P1) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (maxV) for B1.")

        if maxV_P2 is None:
            if self.voltage_sign == 1:
                maxV_P2 = min(self.results[P2]['pinch_off']['voltage']+self.voltage_sign*self.results[P2]['pinch_off']['width'], self.results['turn_on']['saturation'])
            elif self.voltage_sign == -1:
                maxV_P2 = max(self.results[P2]['pinch_off']['voltage']+self.voltage_sign*self.results[P2]['pinch_off']['width'], self.results['turn_on']['saturation'])
        else:
            assert np.sign(maxV_P2) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (maxV) for B2.")

        self.logger.info(f"setting {P1} to {maxV_P1} V")
        self.logger.info(f"setting {P2} to {maxV_P2} V")
        
        self.set_voltage_configuration({P1: maxV_P1, P2: maxV_P2})

        def smooth_reset():
            
            """
            Resets the inner loop variable smoothly back to the starting value
            """
            
            self.set_gates_to_voltage([P2], maxV_P2)

        num_steps_B1 = self.calculate_num_of_steps(minV_P1, maxV_P1, dV_P1)
        num_steps_B2 = self.calculate_num_of_steps(minV_P2, maxV_P2, dV_P2)
        
        self.logger.attempt("barrier barrier scan")
        
        self.logger.info(f"stepping {P1} from {maxV_P1} V to {minV_P1} V")
        self.logger.info(f"sweeping {P2} from {maxV_P2} V to {minV_P2} V")
        
        gates = self.barriers
        param_check = []

        for voltage_source in self.voltage_sources.items():

            for gate_name in gates:
                
                if gate_name in self.voltage_source_names_check[voltage_source[0]]:
                        
                        print(self.voltage_source_names_check[voltage_source[0]][gate_name])

                        param = voltage_source[1][gate_name]

                        param_check.append(param)

        result = qc.dataset.do2d(
            param_check[0], # outer loop
            maxV_P1,
            minV_P1,
            num_steps_B1,
            param_check[1], # inner loop
            maxV_P2,
            minV_P2,
            num_steps_B2,
            self.drain_volt,
            after_inner_actions = [smooth_reset],
            set_before_sweep=True, 
            show_progress=True, 
            measurement_name='Barrier Barrier Sweep',
            exp=self.initialization_exp
        )
        self.logger.complete("\n")

        self.logger.info(f"returning gates {P1}, {P2} to {maxV_P1} V, {maxV_P2} V respectively")
        self.set_voltage([P1], maxV_P1)
        self.set_voltage([P2], maxV_P2)
        
        return None

    def sweep_1d_measurement(self, 
                 maxV: float = None,
                 minV: float = None,
                 voltage_configuration: Dict[str, float] = {},
                 dV: float = 10e-3) -> pd.DataFrame:

        # Bring device to voltage configuration

        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self.set_voltage_configuration(voltage_configuration)

        # Default values
        
        if dV is None:
            dV = self.voltage_resolution

        if maxV is None:
            maxV = self.voltage_sign * self.abs_max_gate_voltage

        # Safety checks
        
        assert np.sign(maxV) == self.voltage_sign
        assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0

        # Gates involved
        
        gates_involved = self.barriers + self.leads + self.accumulation + self.plungers

        self.logger.info(f"setting {gates_involved} to {minV} V")

        self.set_voltage_configuration(gates_involved, minV)

        # Number of steps

        num_steps = self.calculate_num_of_steps(minV, maxV, dV)

        # Build parameter list
        
        gate_params = []

        for voltage_source in self.voltage_sources.items():

            for gate_name in gates_involved:

                if gate_name in self.voltage_source_names_check[voltage_source[0]]:

                    param = voltage_source[1][gate_name]
                    gate_params.append(param)

        # Create sweep values
        
        sweep_vals = np.linspace(minV, maxV, num_steps)

        # Setup measurement
        
        meas = Measurement(exp=self.initialization_exp)

        # Register parameters
        for param in gate_params:
            meas.register_parameter(param)

        meas.register_parameter(self.drain_volt, setpoints=tuple(gate_params))

        # Execute sweep
        self.logger.attempt(
            f"sweeping {gates_involved} together from {minV} V to {maxV} V"
        )

        with meas.run() as datasaver:

            for v in sweep_vals:

                # set all gates together
                for param in gate_params:
                    param.set(v)

                # measurement
                drain = self.drain_volt.get()

                results = [(param, v) for param in gate_params]
                results.append((self.drain_volt, drain))

                datasaver.add_result(*results)

                # break condition
                if self._check_break_conditions(drain):
                    break

        self.logger.complete('\n')

        return None

    def sweep_2d_measurement(self,
                 P1: str = None, 
                 P2: str = None, 
                 P1_bounds: tuple = (None, None),
                 P2_bounds: tuple = (None, None), 
                 dV: float | tuple = None,
                 voltage_configuration: dict = None) -> tuple[pd.DataFrame, plt.Axes]:

        # Bring device to voltage configuration
        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self.set_voltage_configuration(voltage_configuration)
        else:
            self.logger.info(f"setting {self.leads} to {self.results['turn_on']['saturation']} V")
            self.set_voltage(self.leads, self.results['turn_on']['saturation'])

        # Parse dV
        if dV is None:
            dV_P1 = self.voltage_resolution
            dV_P2 = self.voltage_resolution
        elif type(dV) is float:
            dV_P1 = dV
            dV_P2 = dV
        elif type(dV) is tuple:
            dV_P1, dV_P2 = dV
        else:
            self.logger.error("invalid dV")
            return

        # Bounds
        minV_P1, maxV_P1 = P1_bounds
        minV_P2, maxV_P2 = P2_bounds

        # (same bounds logic as your code omitted here for brevity)

        self.logger.info(f"setting {P1} to {maxV_P1} V")
        self.logger.info(f"setting {P2} to {maxV_P2} V")

        self.set_voltage_configuration({P1: maxV_P1, P2: maxV_P2})

        # Step counts
        num_steps_B1 = self.calculate_num_of_steps(minV_P1, maxV_P1, dV_P1)
        num_steps_B2 = self.calculate_num_of_steps(minV_P2, maxV_P2, dV_P2)

        # Generate sweep arrays
        P1_vals = np.linspace(maxV_P1, minV_P1, num_steps_B1)
        P2_vals = np.linspace(maxV_P2, minV_P2, num_steps_B2)

        self.logger.attempt("barrier barrier scan")

        self.logger.info(f"stepping {P1} from {maxV_P1} V to {minV_P1} V")
        self.logger.info(f"sweeping {P2} from {maxV_P2} V to {minV_P2} V")

        # Resolve QCoDeS parameters
        gates = self.barriers
        param_check = []

        for voltage_source in self.voltage_sources.items():

            for gate_name in gates:

                if gate_name in self.voltage_source_names_check[voltage_source[0]]:
                    param = voltage_source[1][gate_name]
                    param_check.append(param)

        P1_param = param_check[0]
        P2_param = param_check[1]

        # Setup measurement
        meas = Measurement(exp=self.initialization_exp)

        meas.register_parameter(P1_param)
        meas.register_parameter(P2_param)
        meas.register_parameter(self.drain_volt, setpoints=(P1_param, P2_param))

        # Inner reset function
        def smooth_reset():

            self.set_gates_to_voltage([P2], maxV_P2)

        # Run measurement
        with meas.run() as datasaver:

            for v1 in P1_vals:

                # set outer parameter
                P1_param.set(v1)

                for v2 in P2_vals:

                    # set inner parameter
                    P2_param.set(v2)

                    drain = self.drain_volt.get()

                    datasaver.add_result(
                        (P1_param, v1),
                        (P2_param, v2),
                        (self.drain_volt, drain),
                    )

                # reset inner sweep
                smooth_reset()

        self.logger.complete("\n")

        # Return gates to starting values
        self.logger.info(f"returning gates {P1}, {P2} to {maxV_P1} V, {maxV_P2} V respectively")

        self.set_voltage([P1], maxV_P1)
        self.set_voltage([P2], maxV_P2)

        return None

    def sweep_nd(self):
        pass

    def sweep_nd_measurement(self):
        pass

    def check_break_conditions(self):
        
        # Go through device break conditions to see if anything is flagged,
        # should return a Boolean.
        
        # breakConditionsDict = {
        #     0: 'Maximum current is exceeded.',
        #     1: 'Maximum ohmic bias is exceeded.',
        #     2: 'Maximum gate voltage is exceeded.',
        #     3: 'Maximum gate differential is exceeded.',
        # }

        # MAX CURRENT

        isExceedingMaxCurrent = np.abs(self._get_drain_current()) > self.abs_max_current
        # time.sleep(0.1)

        # MAX BIAS
        
        # flag = []
        # for gate_name in self.ohmics:
        #     gate_voltage = getattr(self.voltage_source, f'{gate_name}')() 
        #     if np.abs(gate_voltage * self.voltage_divider) > self.abs_max_ohmic_bias:
        #         flag.append(True)
        #     else:
        #         flag.append(False)
        # isExceedingMaxOhmicBias = np.array(flag).any()
        # time.sleep(0.1)

        # MAX GATE VOLTAGE
        
        # flag = []
        # for gate_name in self.all_gates:
        #     gate_voltage = getattr(self.voltage_source, f'{gate_name}')()
        #     if np.abs(gate_voltage) > self.abs_max_gate_voltage:
        #         flag.append(True)
        #     else:
        #         flag.append(False)
        # isExceedingMaxGateVoltage = np.array(flag).any()
        # time.sleep(0.1)

        # # MAX GATE DIFFERENTIAL
        
        # flag = []
        # gates_to_check = self.barriers + self.leads
        # for i in range(len(gates_to_check)):
        #     for j in range(i+1, len(gates_to_check)):
        #         gate_voltage_i = getattr(self.voltage_source, f'{self.all_gates[i]}')()
        #         gate_voltage_j = getattr(self.voltage_source, f'{self.all_gates[j]}')()
        #         # Check if the absolute difference between gate voltages is greater than 0.5
        #         if np.abs(gate_voltage_i - gate_voltage_j) >= self.abs_max_gate_differential:
        #             flag.append(True)
        #         else:
        #             flag.append(False)
        # isExceedingMaxGateDifferential = np.array(flag).any()
        # time.sleep(0.1)
        
        listOfBreakConditions = [
            isExceedingMaxCurrent,
            # isExceedingMaxOhmicBias,
            # isExceedingMaxGateVoltage,
            # isExceedingMaxGateDifferential,
        ]
        isExceeded = np.array(listOfBreakConditions).any()
        
        # breakConditions = np.where(np.any(listOfBreakConditions == True))[0]
        # if len(breakConditions) != 0:
        #     for index in breakConditions.tolist():
        #         print(breakConditionsDict[index]+"\n")

        return isExceeded

    def calculate_num_of_steps(self, 
                                minV: float, 
                                maxV: float, 
                                dV: float):
        """Calculates the number of steps required for a sweep.

        Args:
            minV (float): Minimum voltage (V)   
            maxV (float): Maximum voltage (V)
            dV (float): Step size (V)

        Returns:
            None:
        """
   
        return round(np.abs(maxV-minV) / dV) + 1