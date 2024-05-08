import pandas as pd
import qcodes as qc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import yaml, datetime, sys, time, os, shutil, json
from pathlib import Path

from qcodes.dataset import AbstractSweep
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase
import numpy.typing as npt

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

class DataFitter:
    def __init__(self) -> None:
        pass

    def logarithmic(self, x, a, b, x0, y0):
        return a * np.log(b*(x-x0)) + y0

    def exponential(self, x, a, b, x0, y0):
        return a * np.exp(b * (x-x0)) + y0

    def sigmoid(self, x, a, b, x0, y0):
        return a/(1+np.exp(b * (x-x0))) + y0
    
    def linear(self, x, m, b):
        return m * x + b         

class SingleQuantumDotTuner:
    def __init__(self, 
                 device_config: str, 
                 station_config: str,
                 tuner_config: str,
                 qcodes_config: str,
                 save_dir: str = "") -> None:

        self.DataFitter = DataFitter()

        # Save file names
        self.device_config = device_config
        self.station_config = station_config
        self.tuner_config = tuner_config
        self.qcodes_config = qcodes_config
        self.save_dir = save_dir

        # Read in tuner config information
        self.tuner_info = yaml.safe_load(Path(tuner_config).read_text())
        self.global_turn_on_info = self.tuner_info['global_turn_on']
        self.pinch_off_info = self.tuner_info['pinch_off']

        # Read in station config information
        self.station_info = yaml.safe_load(Path(station_config).read_text())
        self.general_info = self.station_info['general_info']
        self.multimeter_device = self.general_info['multimeter_device']
        self.voltage_device = self.general_info['voltage_device']
        self.sensitivity = self.general_info['sensitivity']
        self.preamp_bias = self.general_info['preamp_bias']
        self.voltage_divider = self.general_info['voltage_divider']
        self.voltage_resolution = self.general_info['voltage_resolution']

        # Read in device config information
        self.device_info = yaml.safe_load(Path(device_config).read_text())
        self.charge_carrier = self.device_info['characteristics']['charge_carrier']
        self.operation_mode = self.device_info['characteristics']['operation_mode']

        if (self.charge_carrier, self.operation_mode) == ('e', 'acc'):
            self.voltage_sign = +1
        if (self.charge_carrier, self.operation_mode) == ('e', 'dep'):
            self.voltage_sign = -1
        if (self.charge_carrier, self.operation_mode) == ('h', 'acc'):
            self.voltage_sign = -1
        if (self.charge_carrier, self.operation_mode) == ('h', 'dep'):
            self.voltage_sign = +1

        self.ohmics = self.device_info['characteristics']['ohmics']
        self.barriers = self.device_info['characteristics']['barriers']
        self.leads = self.device_info['characteristics']['leads']
        self.plungers = self.device_info['characteristics']['plungers']
        self.all_gates = self.barriers + self.leads + self.plungers

        self.abs_max_current = self.device_info['properties']['constraints']['abs_max_current']
        self.abs_max_ohmic_bias = self.device_info['properties']['constraints']['abs_max_ohmic_bias']
        self.abs_max_gate_voltage = self.device_info['properties']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.device_info['properties']['constraints']['abs_max_gate_differential']

        print("Connecting to station ... ")
        self.station = qc.Station(config_file=qcodes_config)
        self.station.load_all_instruments()
        self.voltage_source = getattr(self.station, self.voltage_device)
        self.drain_mm_device = getattr(self.station, self.multimeter_device)
        self.drain_volt = getattr(self.station, self.multimeter_device).volt
        print("done!")

        print("Grounding device ... ", end=' ')
        self._zero_device()
        print("done!")

        # Creates the qcodes database and sets-up the experiment
        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        self.db_folder = os.path.join(save_dir, f"{self.device_info['characteristics']['sample_name']}_{todays_date}")
        os.makedirs(self.db_folder, exist_ok=True)
        db_filepath =  os.path.join(self.db_folder, f"experiments_{self.device_info['characteristics']['sample_name']}_{todays_date}.db")
        print(f"Creating/initializing a database at {db_filepath} ... ", end=' ')
        qc.dataset.initialise_or_create_database_at(
            db_filepath
        )
        print("done!")

        print(f"Creating/initializing the experiment in the database ... ", end=' ')
        self.initialization_exp = qc.dataset.load_or_create_experiment(
            'Initialization',
            sample_name=self.device_info['characteristics']['sample_name']
        )
        print("done!")

        # Copy all of the configs for safekeeping
        print(f"Copying all of the config.yml files to the new directory ... ", end=' ')
        shutil.copy(self.qcodes_config, self.db_folder)
        shutil.copy(self.device_config, self.db_folder)
        shutil.copy(self.tuner_config, self.db_folder)
        shutil.copy(self.station_config, self.db_folder)
        print("done!")

    def bias_ohmic(self, ohmic: str = None, V: float = 0):
        print(f"Setting ohmic {ohmic} to {V} (which is {(V*self.voltage_divider*1e-3)} mV) ... ", end=" ")
        self.voltage_source.set_smooth({ohmic: V})
        print("done!")

    def turn_on_device(self, minV=0, maxV=None, dV=None, delay=0.01):

        if dV == None:
            dV = self.voltage_resolution

        if maxV == None:
            maxV = self.abs_max_gate_voltage

        # Checks if the gate voltages provided are what they should be
        # given the device charge carrier and operation mode.
        assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, "Double check the sign of the gate voltage (minV) for your given device."

        # Set up gate sweeps
        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        gates_involved = self.barriers + self.leads

        print("Zeroing all gates ... ", end=" ")
        self._zero_gates(gates_involved)
        print("done!")

        print(f"Bringing gates to {minV} V ... ", end=" ")
        self._set_gates_to_voltage(gates_involved, minV)
        print("done!")

        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'volt_{gate_name}'), minV, maxV, num_steps, delay, get_after_set=False)
            )

        print(f"Ramping up gates in {gates_involved} from {minV} V to {maxV} V ...", end=" ")
        # Execute the measurement
        result = qc.dataset.dond(
            qc.dataset.TogetherSweep(
                *sweep_list
            ),
            self.drain_volt,
            write_period=0.1,
            break_condition=self._check_break_conditions,
            measurement_name='Device Turn On',
            exp=self.initialization_exp,
            show_progress=True
        )
        print(f"done!")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_volt_{gates_involved[0]}', marker= 'o',s=10)
        df_current.plot.line(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_volt_{gates_involved[0]}', ax=axes, linewidth=1)
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.global_turn_on_info['abs_min_current'], alpha=0.5, c='g', linestyle=':', label=r'$I_{\min}$')
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.abs_max_current, alpha=0.5, c='g', linestyle='--', label=r'$I_{\max}$')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$V_{GATES}$ (V)')
        axes.legend(loc='best')

        axes.set_title('Global Device Turn-On')

        # Mask out any data that is above the minimum turn-on current
        mask = df_current[f'{self.multimeter_device}_current'].abs() > self.global_turn_on_info['abs_min_current'] 
        X = df_current[f'{self.voltage_device}_volt_{gates_involved[0]}']
        Y = df_current[f'{self.multimeter_device}_current']
        X_masked = X[mask]
        Y_masked = Y[mask]

        # Do the fit if possible
        if len(mask) <= 4 or len(X_masked) <= 4:
            print("Insufficient points above turn-on to do any fitting. Decrease dV or increase Vmax.")
        else:
            try:
                guess = self.global_turn_on_info['guess']
                fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFitter, self.global_turn_on_info['fit_function']), X_masked, Y_masked, guess)
                # Extract relevant data from fit params
                a, b, x0, y0 = fit_params
                if self.global_turn_on_info['fit_function'] == 'exponential':
                    V_turn_on =  np.log(-y0/a)/b + x0
                elif self.global_turn_on_info['fit_function'] == 'logarithmic':
                    V_turn_on = np.exp(-y0/a)/b + x0
                V_sat = df_current[f'{self.voltage_device}_volt_{gates_involved[0]}'].iloc[-2] # saturation is the last voltage on the gates

                # Plot / print results to user
                axes.plot(X_masked, getattr(self.DataFitter, self.global_turn_on_info['fit_function'])(X_masked, a, b, x0, y0), 'r-')
                axes.axvline(x=V_turn_on, alpha=0.5, linestyle=':',c='b',label=r'$V_{\min}$')
                axes.axvline(x=V_sat,alpha=0.5, linestyle='--',c='b',label=r'$V_{\max}$')
                axes.legend(loc='best')

                print(f"Device turns on at {np.round(V_turn_on, 2)} V")
                print(f"Device saturates at {np.round(V_sat, 2)} V")

                # Store in device dictionary for later
                self.device_info['properties']['turn_on']['voltage'] = float(round(V_turn_on, 3))
                self.device_info['properties']['turn_on']['saturation_voltage'] = float(round(V_sat, 3))

                I_measured = self._get_drain_current()
                R_measured = round(V_sat / I_measured,3)
                self.device_info['properties']['turn_on']['measured_current'] = float(I_measured)
                self.device_info['properties']['turn_on']['measured_resistance'] = float(R_measured)
                self.deviceTurnsOn = True

            except RuntimeError:
                print(f"Error - fitting to \"{self.global_turn_on_info['fit_function']}\" failed. Manually adjust device info.")
                
                self.deviceTurnsOn = self._query_yes_no("Did the device actually turn-on?")
                
                if self.deviceTurnsOn:
                    V_turn_on = input("What was the turn on voltage (V)?")
                    V_sat = input("What was the saturation voltage (V)?")
                    # Store in device dictionary for later
                    self.device_info['properties']['turn_on']['voltage'] = V_turn_on
                    self.device_info['properties']['turn_on']['saturation_voltage'] = V_sat
        
        self._save_figure(plot_info='device_turn_on')
        self._update_device_config_yaml()

    def pinch_off_device(self, gates: list = [], minV=None, maxV=None, dV=None, delay=0.01):
        
        # Can't pinch off if not turned on.
        assert self.deviceTurnsOn, "Device does not turn on. Why are you pinching anything off?"

        if dV == None:
            dV = self.voltage_resolution

        # Checks if the gate voltages provided are what they should be
        # given the device charge carrier and operation mode.
        if maxV == None:
            maxV = self.device_info['properties']['turn_on']['saturation_voltage']
        else:
            assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        
        if minV == None:
            if self.voltage_sign == 1:
                minV = max(0, round(self.device_info['properties']['turn_on']['saturation_voltage'] - self.abs_max_gate_differential, 3))
            elif self.voltage_sign == -1:
                minV = min(0, round(self.device_info['properties']['turn_on']['saturation_voltage'] - self.abs_max_gate_differential, 3))
        else:
            assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, "Double check the sign of the gate voltage (minV) for your given device."

        # When checking pinch-off have the gates initially where the device saturated
        num_steps = self._calculate_num_of_steps(minV, maxV, dV)

        if len(gates) == 0:
            gates_involved = self.barriers + self.leads
        else:
            gates_involved = gates

        print(f"Settings gates in {gates_involved} to {maxV} V ... ")
        self._set_gates_to_voltage(gates_involved, maxV)
        print("done!")

        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'volt_{gate_name}'), maxV, minV, num_steps, delay, get_after_set=False)
            )

        def adjusted_break_condition():
            return np.abs(self._get_drain_current()) < self.pinch_off_info['abs_min_current']

        for sweep in sweep_list:

            print(f"Pinching off {str(sweep._param).split('_')[-1]} from {maxV} V to {minV} V ... ", end=" ")
            result = qc.dataset.dond(
                sweep,
                self.drain_volt,
                # break_condition=adjusted_break_condition,
                measurement_name='{} Pinch Off'.format(str(sweep._param).split('_')[-1]),
                exp=self.initialization_exp,
                show_progress=True
            )   
            print(f"done!")

            print(f"Returning {str(sweep._param).split('_')[-1]} back to {maxV} V ... ", end=" ")
            self._set_gates_to_voltage([str(sweep._param).split('_')[-1]], maxV)
            print(f"done!")

            # Get last dataset recorded, convert to current units
            dataset = qc.load_last_experiment().last_data_set()
            df = dataset.to_pandas_dataframe().reset_index()
            df_current = df.copy()
            df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
            df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.sensitivity) # sensitivity

            # Plot current v.s. param being swept
            axes = df_current.plot.scatter(y=f'{self.multimeter_device}_current', x=f'{str(sweep._param)}', marker= 'o',s=10)
            df_current.plot.line(y=f'{self.multimeter_device}_current', x=f'{str(sweep._param)}',  ax=axes, linewidth=1)
            axes.axhline(y=0,  alpha=.5, linewidth=0.5, c='k', linestyle='-')
            axes.axvline(x=0, alpha=.5, linewidth=0.5, c='k', linestyle='-')
            axes.set_ylabel(r'$I$ (A)')
            axes.set_xlabel(r'$V_{{{gate}}}$ (V)'.format(gate=str(sweep._param).split('_')[-1]))
            axes.set_title('{} Pinch Off'.format(str(sweep._param).split('_')[-1]))
            axes.set_xlim(0, self.device_info['properties']['turn_on']['saturation_voltage'])
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[0])*self.pinch_off_info['abs_min_current'], alpha=0.5,c='g', linestyle=':', label=r'$I_{\min}$')
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[0])*self.abs_max_current, alpha=0.5,c='g', linestyle='--', label=r'$I_{\max}$')
                
            # Mask out any data that is above the minimum turn-on current
            if str(sweep._param).split('_')[-1] in self.leads:
                mask = df_current[f'{self.multimeter_device}_current'].abs() > self.pinch_off_info['abs_min_current'] 
                X = df_current[f'{str(sweep._param)}']
                Y = df_current[f'{self.multimeter_device}_current']
                X_masked = X[mask]
                Y_masked = Y[mask]
            else:
                X = df_current[f'{str(sweep._param)}']
                Y = df_current[f'{self.multimeter_device}_current']
                X_masked = X
                Y_masked = Y

            # Do the fit if possible
            if len(X_masked) <= 4:
                print("Insufficient points above turn-on to do any fitting. Decrease dV or increase Vmax.")
                print("Trying to fit to linear fit to see if barrier is broken.")

                try:
                    # Guess a line fit with zero slope.
                    guess = (0,np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.abs_max_current)
                    fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFitter, 'linear'), X, Y, guess)
                    m, b = fit_params
                    plt.plot(X, self.DataFitter.linear(X,m,b), 'r-')
                    print("Fits well to line, most likely barrier is shorted somewhere.")

                except RuntimeError:
                    print("Error - fitting to \"linear\" failed.")

            else:
                # Try and fit to sigmoid and get fit params
                try: 
                    if str(sweep._param).split('_')[-1] in self.leads:

                        fit_function = getattr(self.DataFitter, 'logarithmic')
                        guess = self.global_turn_on_info['guess']

                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params
                        
                        V_pinchoff = float(round(np.exp(-y0/a)/b + x0,3))
                        self.device_info['properties'][str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                   
                    else:

                        fit_function = getattr(self.DataFitter, self.pinch_off_info['fit_function'])
                        guess = (-5e-9, -100, self.device_info['properties']['turn_on'], 5e-9)

                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params

                        V_pinchoff = float(round(min(
                            np.abs(x0 - np.sqrt(8) / b),
                            np.abs(x0 + np.sqrt(8) / b)
                        ),3))
                        V_pinchoff_width = float(abs(round(2 * np.sqrt(8) / b,2)))
                        self.device_info['properties'][str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                        self.device_info['properties'][str(sweep._param).split('_')[-1]]['pinch_off']['width'] = V_pinchoff_width #V

                    plt.plot(X_masked, fit_function(X_masked, a, b, x0, y0), 'r-')
                    axes.axvline(x=V_pinchoff, alpha=0.5, linestyle=':', c='b', label=r'$V_{\min}$')
                    if str(sweep._param).split('_')[-1] not in self.leads:
                        axes.axvline(x=V_pinchoff + self.voltage_sign * V_pinchoff_width, alpha=0.5, linestyle='--', c='b', label=r'$V_{\max}$')

                    axes.legend(loc='best')
                
                except RuntimeError:

                    print("Error - curve_fit to sigmoid failed. Manually adjust device info.")
                    print("Trying to fit to linear fit to see if barrier is broken.")

                    try:
                        # Guess a line fit with zero slope.
                        guess = (0,np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.abs_max_current)
                        fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFitter, 'linear'), X, Y, guess)
                        m, b = fit_params
                        plt.plot(X, self.DataFitter.linear(X,m,b), 'r-')
                        print("Fits well to line, most likely barrier is shorted somewhere.")

                    except RuntimeError:
                        print("Error - fitting to \"linear\" failed.")

            self._save_figure(plot_info='{}_pinch'.format(str(sweep._param).split('_')[-1]))
            self._update_device_config_yaml()

    def sweep_barriers(self, 
                              B1: str = None, 
                              B2: str = None, 
                              B1_bounds: tuple = (None, None),
                              B2_bounds: tuple = (None, None), 
                              dV=None, 
                              delay=0.01, 
                              voltage_configuration: dict = {}):

        if voltage_configuration != {}:
            for gate_name, voltage in voltage_configuration.items():
                print(f"Setting {gate_name} to {voltage} V ... ")
                self._set_gates_to_voltage([gate_name], voltage)
                print("done!")

        if dV == None:
            dV = self.voltage_resolution
        
        # If there are only two barriers in the device config
        # then they are the only things that can be swept
        if B1 == None or B2 == None and len(self.barriers) == 2:
            B1, B2 = self.barriers
        
        minV_B1, maxV_B1 = B1_bounds
        minV_B2, maxV_B2 = B2_bounds

        if minV_B1 == None:
            minV_B1 = self.device_info['properties'][B1]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B1) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for B1."

        if minV_B2 == None:
            minV_B2 = self.device_info['properties'][B2]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B2) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for B2."

        if maxV_B1 == None:
            if self.voltage_sign == 1:
                maxV_B1 = min(self.device_info['properties'][B1]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B1]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
            elif self.voltage_sign == -1:
                maxV_B1 = max(self.device_info['properties'][B1]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B1]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
        else:
            assert np.sign(maxV_B1) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for B1."

        if maxV_B2 == None:
            if self.voltage_sign == 1:
                maxV_B2 = min(self.device_info['properties'][B2]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B2]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
            elif self.voltage_sign == -1:
                maxV_B2 = max(self.device_info['properties'][B2]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B2]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
        else:
            assert np.sign(maxV_B2) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for B2."

        num_steps_B1 = self._calculate_num_of_steps(minV_B1, maxV_B1, dV)
        num_steps_B2 = self._calculate_num_of_steps(minV_B2, maxV_B2, dV)

        print(f"Setting lead gates to saturation gate voltage ... ")
        self._set_gates_to_voltage(self.leads, self.device_info['properties']['turn_on']['saturation_voltage'])
        print("done!")

        print(f"Setting {B1} to {maxV_B1} V ...")
        self._set_gates_to_voltage([B1], maxV_B1)
        print(f"done!")

        print(f"Setting {B2} to {maxV_B2} V ...")
        self._set_gates_to_voltage([B2], maxV_B2)
        print(f"done!")

        def smooth_reset():
            self._set_gates_to_voltage([B2], maxV_B2)

        # Running masurement with do2d
        print(f"Stepping {B1} from {maxV_B1} V to {minV_B1} V ...")
        print(f"Sweeping {B2} from {maxV_B2} V to {minV_B2} V ...")
        result = qc.dataset.do2d(
            getattr(self.voltage_source, f'volt_{B1}'), # outer loop
            maxV_B1,
            minV_B1,
            num_steps_B1,
            delay,
            getattr(self.voltage_source, f'volt_{B2}'), # inner loop
            maxV_B2,
            minV_B2,
            num_steps_B2,
            delay,
            self.drain_volt,
            after_inner_actions = [smooth_reset],
            set_before_sweep=True, 
            show_progress=True, 
            # break_condition=self._check_break_conditions,
            measurement_name='Barrier Barrier Sweep',
            exp=self.initialization_exp
        )
        print("done!")

        print(f"Settings gates {B1}, {B2} to {maxV_B1} V, {maxV_B2} V respectively ... ")
        self._set_gates_to_voltage([B1], maxV_B1)
        self._set_gates_to_voltage([B2], maxV_B2)
        print("done!")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.sensitivity) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_device}_current', index=[f'{self.voltage_device}_volt_{B1}'], columns=[f'{self.voltage_device}_volt_{B2}'])
        B1_data, B2_data = df_pivoted.columns, df_pivoted.index
        raw_current_data = df_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        B1_grad = np.gradient(raw_current_data, axis=1)
        B2_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Barrier Barrier Sweep")

        im_ratio = raw_current_data.shape[0]/raw_current_data.shape[1]

        cbar_ax1 = plt.colorbar(ax1.imshow(
            raw_current_data,
            extent=[B1_data[0], B1_data[-1], B2_data[0], B2_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_ratio
        ), ax=ax1,fraction=0.046, pad=0.04)

        cbar_ax1.set_label(r'$I_{SD}$ (nA)')
        ax1.set_title(r'$I_{SD}$')
        ax1.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=B2))
        ax1.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=B1))

        # V grad is actually horizontal
        grad_vector = (1,1) # Takes the gradient along 45 degree axis

        cbar_ax2 = plt.colorbar(ax2.imshow(
            np.sqrt(grad_vector[0] * B1_grad**2 +   grad_vector[1]* B2_grad**2),
            extent=[B1_data[0], B1_data[-1], B2_data[0], B2_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_ratio
        ), ax=ax2,fraction=0.046, pad=0.04)

        cbar_ax2.set_label(r'$\nabla_{\theta=45\circ} I_{SD}$ (nA/V)')
        ax2.set_title(r'$\nabla I_{SD}$')
        ax2.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=B2))
        ax2.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=B1))

        fig.tight_layout()

        self._save_figure(plot_info=f'{B1}_{B2}_sweep')

    def coulomb_blockade(self, P: str = None, P_bounds: tuple = (None, None), voltage_configuration: dict = {}, dV = None, delay=0.01):
        
        if voltage_configuration != {}:
            for gate_name, voltage in voltage_configuration.items():
                print(f"Setting {gate_name} to {voltage} V ... ")
                self._set_gates_to_voltage([gate_name], voltage)
                print("done!")

        if P == None:
            P = self.plungers[0]

        if dV == None:
            dV = self.voltage_resolution

        minV_P, maxV_P = P_bounds

        if minV_P == None:
            minV_P = 0
        else:
            assert np.sign(minV_P) == self.voltage_sign or np.sign(minV_P) == 0, "Double check the sign of the gate voltage (minV) for B2."

        if maxV_P == None:
            maxV_P = self.voltage_sign
        else:
            assert np.sign(maxV_P) == self.voltage_sign or np.sign(maxV_P) == 0, "Double check the sign of the gate voltage (maxV) for B1."

        num_steps_P = self._calculate_num_of_steps(minV_P, maxV_P, dV)

        print(f"Sweeping plunger gate {P} from {minV_P} V to {maxV_P} V ...")
        P_sweep = LinSweep_SIM928(getattr(self.voltage_source, f'volt_{P}'), minV_P, maxV_P, num_steps_P, delay, get_after_set=False)

        print(f"Setting plunger gate {P} to {minV_P} V ...")
        self._set_gates_to_voltage([P], minV_P)
        # Execute the measurement
        result = qc.dataset.dond(
            P_sweep,
            self.drain_volt,
            write_period=0.1,
            # break_condition=self._check_break_conditions,
            measurement_name='Coulomb Blockade',
            exp=self.initialization_exp,
            show_progress=True
        )

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_volt_{P}', marker= 'o',s=10)
        df_current.plot.line(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_volt_{P}', ax=axes, linewidth=1)
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=P))
        axes.legend(loc='best')

        axes.set_title('Coulomb Blockade')

        volt_config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{P}_sweep_{volt_config_str}')


    def coulomb_diamonds(self, 
                         ohmic: str = None, 
                         gate: str = None, 
                         ohmic_bounds: tuple = (None, None),
                         gate_bounds: tuple = (None, None),
                         dV_ohmic: float = None, 
                         dV_gate: float = None,
                         voltage_configuration: dict = {},
                         delay: float =0.01
                         ):

        if voltage_configuration != {}:
            for gate_name, voltage in voltage_configuration.items():
                print(f"Setting {gate_name} to {voltage} V ... ")
                self._set_gates_to_voltage([gate_name], voltage)
                print("done!")

        if dV_ohmic == None:
            dV_ohmic = self.voltage_resolution
        
        if dV_gate == None:
            dV_gate = self.voltage_resolution
        
        minV_ohmic, maxV_ohmic = ohmic_bounds
        minV_gate, maxV_gate = gate_bounds

        if minV_gate == None:
            minV_gate = self.device_info['properties'][gate]['pinch_off']['voltage']
        else:
            assert np.sign(minV_gate) == self.voltage_sign or np.sign(minV_gate) == 0, f"Double check the sign of the gate voltage (minV) for {gate}."

        if maxV_gate == None:
            maxV_gate = self.device_info['properties']['turn_on']['saturation_voltage']
        else:
            assert np.sign(maxV_gate) == self.voltage_sign or np.sign(maxV_gate) == 0, f"Double check the sign of the gate voltage (maxV) for {gate}."

        if minV_ohmic == None:
            minV_ohmic = -1 * float(getattr(self.voltage_source, f"volt_{ohmic}")())

        if maxV_ohmic == None:
            maxV_ohmic = float(getattr(self.voltage_source, f"volt_{ohmic}")())
   
        num_steps_ohmic = self._calculate_num_of_steps(minV_ohmic, maxV_ohmic, dV_ohmic)
        num_steps_gate = self._calculate_num_of_steps(minV_gate, maxV_gate, dV_gate)

        def smooth_reset():
            self._set_gates_to_voltage([ohmic], maxV_ohmic)

        # Running masurement with do2d
        print(f"Stepping {gate} from {maxV_gate} V to {minV_gate}...")
        print(f"Sweeping {ohmic} from {maxV_ohmic} V to {minV_ohmic}...")
        result = qc.dataset.do2d(
            getattr(self.voltage_source, f'volt_{gate}'), # outer loop
            maxV_gate,
            minV_gate,
            num_steps_gate,
            delay,
            getattr(self.voltage_source, f'volt_{ohmic}'), # inner loop
            maxV_ohmic,
            minV_ohmic,
            num_steps_ohmic,
            delay,
            self.drain_volt,
            after_inner_actions = [smooth_reset],
            set_before_sweep=True, 
            show_progress=True, 
            # break_condition=self._check_break_conditions,
            measurement_name='Coulomb Diamonds',
            exp=self.initialization_exp
        )
        print("done!")

        print(f"Settings gates {gate}, {ohmic} to {maxV_gate} V, {maxV_ohmic} V respectively ... ")
        self._set_gates_to_voltage([gate], maxV_gate)
        self._set_gates_to_voltage([ohmic], maxV_ohmic)
        print("done!")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.sensitivity) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_device}_current', index=[f'{self.voltage_device}_volt_{gate}'], columns=[f'{self.voltage_device}_volt_{ohmic}'])
        gate_data, ohmic_data = df_pivoted.columns, df_pivoted.index
        raw_current_data = df_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        gate_grad = np.gradient(raw_current_data, axis=1)
        ohmic_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Coulomb Diamonds")

        im_aspect = np.abs((maxV_ohmic - minV_ohmic) * (maxV_gate - minV_gate))

        cbar_ax1 = plt.colorbar(ax1.imshow(
            raw_current_data,
            extent=[gate_data[0], gate_data[-1], ohmic_data[0], ohmic_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_aspect
        ), ax=ax1,fraction=0.046, pad=0.04)

        cbar_ax1.set_label(r'$I_{SD}$ (nA)')
        ax1.set_title(r'$I_{SD}$')
        ax1.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=ohmic))
        ax1.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))

        # V grad is actually horizontal
        grad_vector = (0,1) # Takes the gradient along 45 degree axis

        cbar_ax2 = plt.colorbar(ax2.imshow(
            np.sqrt(grad_vector[0] * gate_grad**2 +   grad_vector[1]* ohmic_grad**2),
            extent=[gate_data[0], gate_data[-1], ohmic_data[0], ohmic_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_aspect
        ), ax=ax2,fraction=0.046, pad=0.04)

        cbar_ax2.set_label(r'$\nabla_{\theta=45\circ} I_{SD}$ (nA/V)')
        ax2.set_title(r'$\nabla I_{SD}$')
        ax2.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=ohmic))
        ax2.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))

        fig.tight_layout()

        volt_config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{ohmic}_{gate}_sweep_{volt_config_str}')

    def current_trace(self, f_sampling: int, t_capture: int, NPLC=1.0):

        num_of_samples = f_sampling * t_capture 

        meas = qc.dataset.Measurement(exp=self.initialization_exp)
        meas.register_parameter(self.drain_mm_device.timetrace)

        self.drain_mm_device.NPLC(NPLC)
        self.drain_mm_device.timetrace_dt(1/f_sampling)
        self.drain_mm_device.timetrace_npts(num_of_samples)

        print(f'Minimal allowable dt: {self.drain_mm_device.sample.timer_minimum()} s')

        with meas.run() as datasaver:
            datasaver.add_result((self.drain_mm_device.timetrace, self.drain_mm_device.timetrace()),
                                (self.drain_mm_device.time_axis, self.drain_mm_device.time_axis()))

        time_trace_ds = datasaver.dataset
        axs, cbs = qc.dataset.plot_dataset(time_trace_ds)

    def _check_break_conditions(self):
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
        #     gate_voltage = getattr(self.voltage_source, f'volt_{gate_name}')() 
        #     if np.abs(gate_voltage * self.voltage_divider) > self.abs_max_ohmic_bias:
        #         flag.append(True)
        #     else:
        #         flag.append(False)
        # isExceedingMaxOhmicBias = np.array(flag).any()
        # time.sleep(0.1)

        # MAX GATE VOLTAGE
        # flag = []
        # for gate_name in self.all_gates:
        #     gate_voltage = getattr(self.voltage_source, f'volt_{gate_name}')()
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
        #         gate_voltage_i = getattr(self.voltage_source, f'volt_{self.all_gates[i]}')()
        #         gate_voltage_j = getattr(self.voltage_source, f'volt_{self.all_gates[j]}')()
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

    def _get_drain_current(self):
        # Returns the true current measured in amps
        return self.sensitivity * (self.drain_volt() - self.preamp_bias)

    def _set_gates_to_voltage(self, gates: list, voltage: float):
        self.voltage_source.set_smooth(
            dict(zip(gates, [voltage]*len(gates)))
        )

    def _zero_ohmics(self, ohmics: list):
        self.voltage_source.set_smooth(
            dict(zip(ohmics, [0]*len(ohmics)))
        )

    def _zero_gates(self, gates: list):
        self.voltage_source.set_smooth(
            dict(zip(gates, [0]*len(gates)))
        )

    def _zero_device(self):
        self.voltage_source.set_smooth(
            dict(zip(self.ohmics + self.all_gates, [0]*len(self.ohmics + self.all_gates)))
        )

    def _query_yes_no(self, question, default="no"):
        """Ask a yes/no question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
                It must be "yes" (the default), "no" or None (meaning
                an answer is required of the user).

        The "answer" return value is True for "yes" or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == "":
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

    def _calculate_num_of_steps(self, minV, maxV, dV):
        return round(np.abs(maxV-minV) / dV) + 1
    
    def _update_device_config_yaml(self):

        print("Updating device_config.yml file with findings ... ")
        with open(self.device_config, 'w') as outfile:
            yaml.dump(self.device_info, outfile, default_flow_style=True)
        print("done!")

    def _save_figure(self, plot_info):
        completition_time = str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) # the current minute
        plot_name = os.path.join(self.db_folder, f"{plot_info}_{completition_time}.png")
        plt.savefig(fname=plot_name, dpi=300)