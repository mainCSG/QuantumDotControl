import pandas as pd
import qcodes as qc
import yaml
from pathlib import Path
import datetime
import numpy as np

class DataFitter:
    def __init__(self) -> None:
        pass

class SingleQuantumDotTuner:
    def __init__(self, 
                 device_config: str, 
                 station_config: str) -> None:

        self.station_info = yaml.safe_load(Path(station_config).read_text())
        self.general_info = self.station_info['general_info']
        self.sensitivity = self.general_info['sensitivity']
        self.preamp_bias = self.general_info['preamp_bias']
        self.voltage_divider = self.general_info['voltage_divider']
        self.voltage_resolution = self.general_info['voltage_resolution']

        self.device_info = yaml.safe_load(Path(device_config).read_text())
        self.ohmics = self.device_info['characteristics']['ohmics']
        self.barriers = self.device_info['characteristics']['barriers']
        self.leads = self.device_info['characteristics']['leads']
        self.plungers = self.device_info['characteristics']['plungers']
        self.all_gates = self.barriers + self.leads + self.plungers

        self._zero_device()

        self.station = qc.Station(config_file=station_config)
        self.station.load_all_instruments()
        self.drain = self.station.agilent.volt

        print(f"Creating/initializing a database at ~/experiments_*.db ... ")
        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        qc.dataset.initialise_or_create_database_at(
            f"~/experiments_{todays_date}.db"
            )
        print("Done!")

        print(f"Creating/initializing an experiment in the database ... ")
        self.initialization_exp = qc.dataset.load_or_create_experiment(
            'Initialization',
            sample_name=self.device['sample_name']
        )
        print("Done!")

    def bias_devices(self, Vbias=0):
        gates = self.ohmics
        for gate_name in gates:
            self.station.sim900.set_smooth({gate_name: Vbias})

    def check_turn_on(self, minV=0, maxV=None):
        num_steps = int(np.abs(maxV-minV) / self.voltage_resolution) + 1
        gates = self.barriers + self.leads
        sweep_list = []

        for gate_name in gates:
            sweep_list.append(
                qc.dataset.LinSweep(getattr(self.station.sim900, f'volt_{gate_name}'), 0, maxV, num_steps, 0.01, get_after_set=True)
            )

        result = qc.dataset.dond(
            qc.dataset.TogetherSweep(
                *sweep_list
            ),
            self.drain,
            break_condition=self._check_break_condition,
            measurement_name='Turn On',
            exp=self.initialization_exp
        )

        # Fit data to desired function

        # From fit determine whether appropriate turn-on happened

        # Report to user

        # Plot findings

    def check_pinch_offs(self, minV=None, maxV=None):
        num_steps = int(np.abs(maxV-minV) / self.voltage_resolution) + 1
        gates = self.barriers 
        sweep_list = []

        for gate_name in gates:
            sweep_list.append(
                qc.dataset.LinSweep(getattr(self.station.sim900, f'volt_{gate_name}'), minV, maxV, num_steps, 0.01, get_after_set=True)
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

            # Report to user

            # Plot findings

    def barrier_barrier_sweep(self, B1: str, B2: str):
        pass

    def coulomb_blockade(self, P: str, S: str):
        pass

    def _check_break_conditions(self):
        # Go through device break conditions to see if anything is flagged,
        # should return a Boolean.
        return True

    def _get_drain_current(self):
        return self.drain() * self.sensitivity - self.preamp_bias

    def _zero_ohmics(self, ohmics: list):
        self.station.sim900.set_smooth(
            dict(zip(ohmics, [0]*len(ohmics)))
        )

    def _zero_gates(self, gates: list):
        self.station.sim900.set_smooth(
            dict(zip(gates, [0]*len(gates)))
        )

    def _zero_device(self):
        self.station.sim900.set_smooth(
            dict(zip(self.ohmics + self.all_gates, [0]*len(self.ohmics + self.all_gates)))
        )