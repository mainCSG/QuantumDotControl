import pandas as pd
import qcodes as qc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import yaml, datetime, sys
from pathlib import Path

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
        self.drain_mm_device = getattr(self.station, self.multimeter_device)
        self.drain_volt = getattr(self.station, self.multimeter_device).volt
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

        if maxV == None:
            maxV = self.abs_max_gate_voltage

        # Checks if the gate voltages provided are what they should be
        # given the device charge carrier and operation mode.
        assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        assert np.sign(minV) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for your given device."

        # Set up gate sweeps
        num_steps = int(np.abs(maxV-minV) / dV) + 1
        gates_involved = self.barriers + self.leads

        print("Zeroing all gates ... ")
        self._zero_gates(gates_involved)
        print("Done!")
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
            self.drain_volt,
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
        axes.set_title('Global Device Turn-On')

        # Mask out any data that is above the minimum turn-on current
        mask = df_current[f'{self.multimeter_device}_current'].abs() > self.global_turn_on_info['abs_min_current'] 
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
                V_sat = df_current[f'{self.voltage_device}_volt_{gates_involved[0]}'].iloc[-2] # saturation is the last voltage on the gates

                # Plot / print results to user
                axes.plot(X_masked, getattr(self.DataFitter, self.global_turn_on_info['fit_function'])(X_masked, a, b, x0, y0), 'r-')
                axes.axvline(x=V_turn_on, alpha=0.5, linestyle=':',c='b',label=r'$V_{\min}$')
                axes.axvline(x=V_sat,alpha=0.5, linestyle='--',c='b',label=r'$V_{\max}$')
                axes.legend(loc='best')

                print(f"Device turns on at {V_turn_on} V")
                print(f"Device saturates at {V_sat} V")

                # Store in device dictionary for later
                self.device_info['properties']['turn_on'] = np.round(V_turn_on, 3)
                self.device_info['properties']['saturation'] = np.round(V_sat, 3)

                self.deviceTurnsOn = True

            except RuntimeError:
                print(f"Error - fitting to \"{self.global_turn_on_info['fit_function']}\" failed. Manually adjust device info.")
                
                self.deviceTurnsOn = self._query_yes_no("Did the device actually turn-on?")
                
                if self.deviceTurnsOn:
                    V_turn_on = input("What was the turn on voltage (V)?")
                    V_sat = input("What was the saturation voltage (V)?")
                    # Store in device dictionary for later
                    self.device_info['properties']['turn_on'] = V_turn_on
                    self.device_info['properties']['saturation'] = V_sat
                    
    def check_pinch_offs(self, minV=None, maxV=None):
        # Checks if the gate voltages provided are what they should be
        # given the device charge carrier and operation mode.
        assert self.deviceTurnsOn, "Device does not turn on. Why are you pinching anything off?"

        if maxV == None:
            maxV = self.device_info['properties']['saturation']
        else:
            assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        
        if minV == None:
            minV = self.device_info['properties']['saturation'] - 0.9 * self.device_info['properties']['abs_max_gate_differential']
        else:
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
                self.drain_volt,
                break_condition=self._check_break_conditions,
                measurement_name='{} Pinch Off'.format(str(sweep._param).split('_')[-1]),
                exp=self.initialization_exp
            )

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
            axes.set_xlim(0, self.device_info['properties']['saturation'])
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.pinch_off_info['abs_min_current'], alpha=0.5,c='g', linestyle=':', label=r'$I_{\min}$')
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.abs_max_current, alpha=0.5,c='g', linestyle='--', label=r'$I_{\max}$')
                
            # Mask out any data that is above the minimum turn-on current
            mask = df_current[f'{self.multimeter_device}_current'].abs() > self.pinch_off_info['abs_min_current'] 
            X = df_current[f'{str(sweep._param)}']
            Y = df_current[f'{self.multimeter_device}_current']
            X_masked = X[mask]
            Y_masked = Y[mask]

            # Do the fit if possible
            if len(mask) <= 4:
                print("Insufficient points above turn-on to do any fitting. Decrease dV or increase Vmax.")
            else:
                # Try and fit to sigmoid and get fit params
                try: 
                    guess = (-5e-9,-100,self.device_info['properties']['turn_on'],5e-9)
                    fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFitter, self.pinch_off_info['fit_function']), X_masked, Y_masked, guess)
                    a, b, x0, y0 = fit_params

                    plt.plot(X, getattr(self.DataFitter, self.pinch_off_info['fit_function'])(X, a, b, x0, y0), 'r-')

                    V_pinchoff = round(min(
                        np.abs(x0 - np.sqrt(8) / b),
                        np.abs(x0 + np.sqrt(8) / b)
                    ),3)
                    V_pinchoff_width = abs(round(2 * np.sqrt(8) / b,3))

                    axes.axvline(x=V_pinchoff, alpha=0.5, linestyle=':', c='b', label=r'$V_{\min}$')
                    axes.legend(loc='best')

                    self.device_info['properties'][str(sweep._param).split('_')[-1]]['pinch_off']['value'] = V_pinchoff
                    self.device_info['properties'][str(sweep._param).split('_')[-1]]['pinch_off']['width'] = V_pinchoff_width #V

                    print(f"Width of sigmoid: {V_pinchoff_width} V")
                    print(f"{str(sweep._param).split('_')[-1]} Pinch off: {V_pinchoff} V")
                
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

    def barrier_barrier_sweep(self, B1: str, B2: str):
        
        if maxV == None:
            maxV = self.device_info['properties']['saturation']
        else:
            assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."

        num_steps_B1 = int(np.abs(maxV-self.device_info['properties'][B1]['pinch_off']['value']) / self.voltage_resolution) + 1
        num_steps_B2 = int(np.abs(maxV-self.device_info['properties'][B2]['pinch_off']['value']) / self.voltage_resolution) + 1
        gates_involved = self.leads + [B1, B2]
        self._set_gates_to_value(gates_involved, maxV)
        
        sweep_B1 = qc.dataset.LinSweep(getattr(self.voltage_source, f'volt_{B1}'), maxV, self.device_info['properties'][B1]['pinch_off']['value'], num_steps_B1, 0.01, get_after_set=False)
        sweep_B2 = qc.dataset.LinSweep(getattr(self.voltage_source, f'volt_{B2}'), maxV, self.device_info['properties'][B2]['pinch_off']['value'], num_steps_B2, 0.01, get_after_set=False)

        BB_sweep = qc.dataset.dond(
            sweep_B1,
            sweep_B2,
            self.drain, 
            show_progress=True, 
            break_condition=self._check_break_conditions,
            measurement_name='Barrier Barrier Sweep',
            exp=self.initialization_exp
        )

    def coulomb_blockade(self, P: str, S: str):
        pass

    def current_trace(self, f_sampling: int, t_capture: int, NPLC=None):

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
        
        # MAX CURRENT
        isExceedingMaxCurrent = np.abs(self._get_drain_current()) > self.abs_max_current

        # MAX BIAS
        flag = []
        for gate_name in self.ohmics:
            gate_voltage = getattr(self.voltage_source, f'volt_{gate_name}')() 
            if np.abs(gate_voltage * self.voltage_divider) > self.abs_max_ohmic_bias:
                flag.append(True)
            else:
                flag.append(False)
        isExceedingMaxOhmicBias = np.array(flag).any()

        # MAX GATE VOLTAGE
        flag = []
        for gate_name in self.all_gates:
            gate_voltage = getattr(self.voltage_source, f'volt_{gate_name}')()
            if np.abs(gate_voltage) > self.abs_max_gate_voltage:
                flag.append(True)
            else:
                flag.append(False)
        isExceedingMaxGateVoltage = np.array(flag).any()

        # MAX GATE DIFFERENTIAL
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
        return self.sensitivity * (self.drain_volt() - self.preamp_bias)

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