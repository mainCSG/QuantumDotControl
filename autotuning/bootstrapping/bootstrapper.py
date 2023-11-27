import pandas as pd
import numpy as np
import logging
import qcodes
import yaml

from scipy.optimize import curve_fit


class Bootstrapper:
    def __init__(self) -> None:
        self.funcfit = FunctionFitter()

        print("(0) Creating station ... ")
        self.station = qcodes.Station()
        self.station.load_config_file('station_config.yml')
        self.station.load_all_instruments()
        self.station.keithley.sense.function('current')
        print('Done!')

        print("(0a) Loading user params.yml file ... ")
        # Load YAML file
        with open('your_file.yaml', 'r') as file:
            params = yaml.safe_load(file)

        self.global_turn_on_params = params['global_turn_on']


    def initialize_dut(self, DUT_csv: str):
        assert 'nmos' in DUT_csv.lower() or 'pmos' in DUT_csv.lower(), "Please put device type in csv name!"

        if 'nmos' in DUT_csv.lower():
            self.isNMOS = True
        elif 'pmos' in DUT_csv.lower():
            self.isPMOS = True

        # Read CSV into proper mapping
        self.DUT_DAC_mapping = self._get_dac_mapping(DUT_csv)
        self.DUT_IDC_mapping = self._get_idc_mapping(DUT_csv)
        self.DUT_LIMITS = self._get_dac_limits(DUT_csv)

        # Zero DACS
        print("(1) Zeroing all DACS ... ")
        self.station.ivvi.set_dacs_zero()
        print("Done!\n")

        # Global Turn On
        print("(2) Determining turn on voltage ... ")
        self._global_turn_on()
        print("Done!")


    def _zero_dacs(self, gate: list[str]):
        self._set_voltage(gate, value=[Quantity(0, unit='V')])

    def _get_dac_mapping(self, DUT_csv: str):
        DUT_df = pd.read_csv(DUT_csv)
        DUT_mapping = dict(zip(DUT_df['GATE'], DUT_df['DAC']))

        assert DUT_mapping['ST_TOP'] == DUT_mapping['ST_BOT'], "Lead gates need identical DAC mapping!"

        # Follow qtNE protocol for ID names
        ST_DAC = DUT_mapping['ST_TOP']
        DUT_mapping['ST'] = ST_DAC

        SOURCE_DAC = DUT_mapping['S']
        DUT_mapping['VSDe'] = SOURCE_DAC

        DUT_mapping.pop('ST_BOT')
        DUT_mapping.pop('ST_TOP')
        DUT_mapping.pop('S')
        
        return DUT_mapping

    def _get_idc_mapping(self, DUT_csv: str):
        DUT_df = pd.read_csv(DUT_csv)
        DUT_mapping = dict(zip(DUT_df['GATE'], DUT_df['IDC']))

        return DUT_mapping
    
    def _get_dac_limits(self, DUT_csv: str):
        DUT_df = pd.read_csv(DUT_csv)
        DUT_mapping = {}
        for gate_name, min_val, max_val in dict(zip(DUT_df['GATE'], DUT_df['MIN'], DUT_df['MAX'])):
            DUT_mapping[gate_name] = {'MIN': min_val, 'MAX': max_val}
        return DUT_mapping
    
    def _global_turn_on(self):

        step_size = self.global_turn_on_params['step_size']
        min_current = self.global_turn_on_params['min_current_threshold']
        max_current = self.global_turn_on_params['max_current_threshold']
        bias_voltage = self.global_turn_on_params['bias_voltage']

        polarity = 1 if self.isNMOS else -1
        VBias = Quantity(bias_voltage.unit, bias_voltage.unit)
        dV = Quantity(polarity * step_size.value, step_size.unit)
        min_current = Quantity(min_current.value, min_current.unit)
        max_current = Quantity(max_current.value, max_current.unit)

        # Set VBias
        self._set_voltage(gate=['VSDe'], value=[VBias])

        # Read current
        Idc = self._measure_current(['D'])[0]

        # Store current
        currents = [Idc.value]

        max_voltage = Quantity(min(
                self.DUT_LIMITS['ST']['MAX'],
                self.DUT_LIMITS['LB']['MAX'],
                self.DUT_LIMITS['RB']['MAX']
            ), unit='mV')

        while Idc.value < min_current.value:
            self._increment_voltage(gate=['ST', 'RB', 'LB'], dV=[dV]*3)
            ST_LB_RB_voltage = Quantity(
                np.average(self._read_voltage(gate=['ST', 'RB', 'LB'])),
                unit='mV'
            )

            if ST_LB_RB_voltage.value > max_voltage.value:
                raise ValueError(f"ST_LB_RB saturated voltage at {max_voltage.value} V ... terminating.")

            Idc = self._measure_current(['D'])[0]
            currents.append(Idc.value)

        V_turn_on = Quantity(
            np.average(self._read_voltage(gate=['ST', 'RB', 'LB'])),
            unit='mV'
        )

        while Idc.value < max_current.value:
            self._increment_voltage(gate=['ST', 'RB', 'LB'], dV=[dV]*3)
            ST_LB_RB_voltage = Quantity(
                np.average(self._read_voltage(gate=['ST', 'RB', 'LB'])),
                unit='mV'
            )
            Idc = self._measure_current(['D'])[0]
            currents.append(Idc.value)

        V_saturation = Quantity(
            np.average(self._read_voltage(gate=['ST', 'RB', 'LB'])),
            unit='mV'
        )

        voltages = np.linspace(0, V_saturation.value, np.abs((V_saturation.value)//dV.value))
        global_turn_on_df = pd.DataFrame({'ST_RB_LB (V)': voltages, 'Idc (A)': currents})
        global_turn_on_df.to_csv('global_turn_on.csv', index=False)

        # Fit
        fit_mask = (voltages > V_turn_on)
        X,Y = voltages[fit_mask], currents[fit_mask]
        guess = (-max(Y), -1, min(X), max(Y))

        fit_params, fit_cov = curve_fit(self.funcfit.exp_fit, X, Y, guess)
        a, b, x0, y0 = fit_params

        self.global_turn_on_fit_params = [a, b, x0, y0]
        self.global_turn_on_dist = V_saturation.value - V_turn_on.value

    def _isolate_channel(self):
        pass

    def _finger_gate_pinch_off(self):
        pass

    def _tune_barrier_values(self):
        pass

    def _set_voltage(self, gate: list[str], value: list[object]):
        DAC_list = [self.DUT_DAC_mapping[g] for g in gate]
        for dac, val in zip(DAC_list, value):
            new_voltage = Quantity(val.value, unit='V', max_value=self.DUT_LIMITS[dac]['MAX'], min_value=self.DUT_LIMITS[dac]['MIN'])
            self.station.ivvi._set_dac(dac, new_voltage.value * 1e3) # mV

    def _increment_voltage(self, gate: list[str], dV: list[object]):
        DAC_list = [self.DUT_DAC_mapping[g] for g in gate]
        for dac, val in zip(DAC_list, dV):
            cur_voltage = self.station.ivvi._get_dac(dac) # mV
            new_voltage = Quantity(cur_voltage/1e3 + val.value, unit='V', max_value=self.DUT_LIMITS[dac]['MAX'], min_value=self.DUT_LIMITS[dac]['MIN'])
            self.station.ivvi._set_dac(dac, new_voltage.value * 1e3) # mV

    def _read_voltage(self, gate: list[str])->list[float]:
        DAC_list = [self.DUT_DAC_mapping[g] for g in gate]
        voltages = []
        for dac in DAC_list:
            cur_voltage = self.station.ivvi._get_dac(dac) # mV
            voltages.append(cur_voltage)
        return voltages

    def _measure_current(self, gate: list[str])->list[object]:
        # return [Quantity(self.station.keithley.current(), unit='A')]
        gate = ['ST']
        DAC_list = [self.DUT_DAC_mapping[g] for g in gate]
        for dac in DAC_list:
            cur_voltage = self.station.ivvi._get_dac(dac)
            if cur_voltage < 0.4:
                return [Quantity(0, unit='A')]
            else:
                return [Quantity(cur_voltage / 1e9, unit='A')]

class Quantity:
    def __init__(self, value: float, unit: str, min_value = -np.inf, max_value = np.inf) -> None:

        base_units = ["V", "A"]
        if unit not in base_units:
            prefixes = {
                'y': -24,
                'z': -21,
                'a': -18,
                'f': -15,
                'p': -12,
                'n': -9,
                'u': -6,
                'm': -3,
                'c': -2,
                'd': -1,
                '': 0,
                'da': 1,
                'h': 2,
                'k': 3,
                'M': 6,
                'G': 9,
                'T': 12,
                'P': 15,
                'E': 18,
                'Z': 21,
                'Y': 24,
            }

            exponent = prefixes[unit[0]]
            unit = unit[1:]
        else: 
            exponent = 0


        self.min_value = (min_value * 10 ** exponent) 
        self.max_value = (max_value * 10 ** exponent) 

        self.unit = unit
        self.value = self._clip_number((value * 10 ** exponent), self.min_value, self.max_value)

    def _clip_number(self, number, min_value, max_value):
        return max(min(number, max_value), min_value)
    
class FunctionFitter:
    def __init__(self) -> None:
        pass

    def sigmoid_fit(self, x, a, b, x0, y0):
        return a/(1+np.exp(b * (x-x0))) + y0

    def exp_fit(self, x, a, b, x0, y0):
        return a * np.exp(b * (x-x0)) + y0

    def log_fit(self, x, a, b, x0, y0):
        return a * np.log(b * (x-x0)) + y0

    def cosh_fit(self, x, a, b, x0, y0):
        return a * np.cosh(b * (x-x0))**-2 + y0
