import pandas as pd
import qcodes as qc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import datetime

class DataFitter:
    def __init__(self) -> None:
        pass

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
                 tuner_config: str) -> None:

        self.DataFitter = DataFitter()

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

        self.abs_max_current = self.device_info['properties']['abs_max_current']
        self.abs_max_ohmic_bias = self.device_info['properties']['abs_max_ohmic_bias']
        self.abs_max_gate_voltage = self.device_info['properties']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.device_info['properties']['abs_max_gate_differential']

        print("Connecting to station ... ")
        self.station = qc.Station(config_file=station_config)
        self.station.load_all_instruments()
        self.voltage_source = getattr(self.station, self.voltage_device)
        self.drain = getattr(self.station, self.multimeter_device).volt
        print("Done!")

        print("Grounding device ... ")
        self._zero_device()
        print("Done!")

        print(f"Creating/initializing a database at ~/experiments_*.db ... ")
        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        qc.dataset.initialise_or_create_database_at(
            f"~/experiments_{todays_date}.db"
            )
        print("Done!")

        print(f"Creating/initializing the experiment in the database ... ")
        self.initialization_exp = qc.dataset.load_or_create_experiment(
            'Initialization',
            sample_name=self.device['sample_name']
        )
        print("Done!")

    def bias_device(self, Vbias=0):
        gates = self.ohmics
        for gate_name in gates:
            self.voltage_source.set_smooth({gate_name: Vbias})

    def check_turn_on(self, minV=0, maxV=None, dV=0.001):

        # Checks if the gate voltages provided are what they should be
        # given the device charge carrier and operation mode.
        assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        assert np.sign(minV) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for your given device."

        # Set up gate sweeps
        num_steps = int(np.abs(maxV-minV) / dV) + 1
        gates_involved = self.barriers + self.leads

        self._zero_gates(gates_involved)
        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                qc.dataset.LinSweep(getattr(self.voltage_source, f'volt_{gate_name}'), 0, maxV, num_steps, 0.01, get_after_set=False)
            )

        # Execute the measurement
        result = qc.dataset.dond(
            qc.dataset.TogetherSweep(
                *sweep_list
            ),
            self.drain,
            break_condition=self._check_break_conditions,
            measurement_name='Turn On',
            exp=self.initialization_exp
        )

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
        axes.set_title('Global Turn-On')

        # Mask out any data that is above the minimum turn-on current
        mask = df_current['agilent_current'].abs() > self.global_turn_on_info['abs_min_current'] 
        X = df_current[f'{self.voltage_device}_volt_{gates_involved[0]}']
        Y = df_current[f'{self.multimeter_device}_current']
        X_masked = X[mask]
        Y_masked = Y[mask]

        # Do the fit if possible
        if len(mask) <= 4:
            print("Insufficient points above turn-on to do any fitting. Decrease dV or increase Vmax.")
        else:
            try:
                guess = (-max(Y_masked), 0.5, min(X_masked), max(Y_masked))
                fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFitter, self.global_turn_on_info['fit_function']), X_masked, Y_masked, guess)
                # Extract relevant data from fit params
                a, b, x0, y0 = fit_params
                V_turn_on =  np.log(-y0/a)/b + x0
                V_sat = df_current[f'{self.voltage_device}_volt_{gates_involved[0]}'].iloc[-2] # Saturation is the last voltage on the gates

                # Plot / print results to user
                axes.plot(X_masked, getattr(self.DataFitter, self.global_turn_on_info['fit_function'])(X_masked, a, b, x0, y0), 'r-')
                axes.axvline(x=V_turn_on, alpha=0.5, linestyle=':',c='b',label=r'$V_{\min}$')
                axes.axvline(x=V_sat,alpha=0.5, linestyle='--',c='b',label=r'$V_{\max}$')
                axes.legend(loc='best')

                print("Turn on: ", V_turn_on, "V")
                print("Saturation Voltage: ", V_sat, "V")
                print("Global Turn On Distance: ", V_sat - V_turn_on, "V")

                # Store in device dictionary for later
                self.device_info['Properties']['Turn On'] = np.round(V_turn_on, 3)
                self.device_info['Properties']['Saturation'] = np.round(V_sat, 3)
                self.device_info['Properties']['Turn On Distance'] = round(V_sat - V_turn_on, 3)

            except RuntimeError:
                print(f"Error - fitting to \"{self.global_turn_on_info['fit_function']}\" failed. Manually adjust device info.")

    def check_pinch_offs(self, minV=None, maxV=None):
        # Checks if the gate voltages provided are what they should be
        # given the device charge carrier and operation mode.
        if maxV == None:
            maxV = self.device_info['Properties']['Saturation']
        else:
            assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."

        assert np.sign(minV) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for your given device."

        # When checking pinch-off have the gates initially where the device saturated
        num_steps = int(np.abs(maxV-minV) / self.voltage_resolution) + 1
        gates_involved = self.barriers + self.leads
        self._set_gates_to_value(gates_involved, maxV)
        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                qc.dataset.LinSweep(getattr(self.voltage_source, f'volt_{gate_name}'), maxV, minV, num_steps, 0.01, get_after_set=False)
            )

        for sweep in sweep_list:
            print(f"Pinching off {str(sweep._param).split('_')[-1]}")
            result = qc.dataset.dond(
                sweep,
                self.drain,
                break_condition=self._check_break_conditions,
                measurement_name='{} Pinch Off'.format(str(sweep._param).split('_')[-1]),
                exp=self.initialization_exp
            )

            # Fit data to theoretical function

            # From fit determine whether appropriate pinch-off occured

            ## BELOW CHECKS IF GATE IS FRIED 
            # try:
            #     # Guess a line fit with zero slope.
            #     guess = (0,np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.abs_max_current)
            #     fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFitter, 'linear'), X, Y, guess)
            #     a, b, x0, y0 = fit_params
            #     print("Fits well to a line, most likely one of the gates broke.")
            # except RuntimeError:
            #     print("Error - fitting to \"linear\" failed.")

            # Report to user

            # Plot findings

    def barrier_barrier_sweep(self, B1: str, B2: str):
        pass

    def coulomb_blockade(self, P: str, S: str):
        pass

    def _check_break_conditions(self):
        # Go through device break conditions to see if anything is flagged,
        # should return a Boolean.
        
        isExceedingMaxCurrent = np.abs(self._get_drain_current()) > self.abs_max_current

        flag = []
        for gate_name in self.ohmics:
            gate_voltage = getattr(self.voltage_source, f'volt_{gate_name}')() 
            if np.abs(gate_voltage * self.voltage_divider) > self.abs_max_ohmic_bias:
                flag.append(True)
            else:
                flag.append(False)
        isExceedingMaxOhmicBias = np.array(flag).any()

        flag = []
        for gate_name in self.all_gates:
            gate_voltage = getattr(self.voltage_source, f'volt_{gate_name}')()
            if np.abs(gate_voltage) > self.abs_max_gate_voltage:
                flag.append(True)
            else:
                flag.append(False)
        isExceedingMaxGateVoltage = np.array(flag).any()

        flag = []
        for i in range(len(self.all_gates)):
            for j in range(i+1, len(self.all_gates)):
                gate_voltage_i = getattr(self.voltage_source, f'volt_{self.all_gates[i]}')()
                gate_voltage_j = getattr(self.voltage_source, f'volt_{self.all_gates[j]}')()
                # Check if the absolute difference between gate voltages is greater than 0.5
                if np.abs(gate_voltage_i - gate_voltage_j) > self.abs_max_gate_differential:
                    flag.append(True)
                else:
                    flag.append(False)
        isExceedingMaxGateDifferential = np.array(flag).any()

        isExceeded = np.array([
            isExceedingMaxCurrent,
            isExceedingMaxOhmicBias,
            isExceedingMaxGateDifferential,
            isExceedingMaxGateVoltage
        ]).any()
        return isExceeded

    def _get_drain_current(self):
        # Returns the true current measured in amps
        return self.sensitivity * (self.drain() - self.preamp_bias)

    def _set_gates_to_value(self, gates: list, value: float):
        self.voltage_source.set_smooth(
            dict(zip(gates, [value]*len(gates)))
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