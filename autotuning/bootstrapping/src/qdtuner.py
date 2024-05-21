import pandas as pd
import qcodes as qc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import yaml, datetime, sys, time, os, shutil, json
from pathlib import Path
from typing import List, Dict

from qcodes.dataset import AbstractSweep
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase
import numpy.typing as npt

import skimage
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import matplotlib.cm as cm

original_sys_path = sys.path.copy()

try:

    sys.path.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../coarse_tuning/src')
        )
    )

    from inference import *

finally:

    sys.path = original_sys_path

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

class DataFit:
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
    
class FET_DataAnalyzer:
    def __init__(self, tuner_config) -> None:
        self.tuner_info = yaml.safe_load(Path(tuner_config).read_text())

        self.model_path = self.tuner_info['barrier_barrier']['segmentation_model_path']
        self.model_config_path = self.tuner_info['barrier_barrier']['segmentation_model_config_path']
        self.model_name =self.tuner_info['barrier_barrier']['segmentation_model_name']
        self.model_processor = self.tuner_info['barrier_barrier']['segmentation_model_processor']
        self.confidence_threshold = self.tuner_info['barrier_barrier']['segmentation_confidence_threshold']
        self.polygon_threshold = self.tuner_info['barrier_barrier']['segmentation_polygon_threshold']
        self.segmentation_class = self.tuner_info['barrier_barrier']['segmentation_class']
        
    def extract_bias_point(self,
                           data: pd.DataFrame,
                           plot_process: bool,
                           axes: plt.Axes):
        
        # Inference image for anything above 0.5
        outputs, metadata, image, Xdata, Ydata = inference(
            data,
            self.model_path,
            self.model_config_path,
            self.model_name,
            self.model_processor,
            self.confidence_threshold,
            self.polygon_threshold,
            plot=False
        )

        # Only keep things with class 'CD' = 'Central Dot'
        outputs = outputs[outputs.pred_classes == metadata.thing_classes.index(self.segmentation_class)]

        # Get the bounding box with the best score
        bboxes = outputs.pred_boxes.tensor.numpy()
        max_score_index = np.argmax(outputs.scores)
        best_bbox = bboxes[max_score_index]
        best_score = outputs.scores[max_score_index]

        bbox_units = pixel_polygon_to_image_units(best_bbox, data)
        x, y = bbox_units[:,0], bbox_units[:,1]
        x1,x2 = x
        y1,y2 = y

        if plot_process:
            # Plot the bounding box
            axes.plot([x1, x2], [y1, y1], linewidth=3, alpha=0.5, linestyle='--', color='k')  # Top line
            axes.plot([x1, x2], [y2, y2], linewidth=3, alpha=0.5,  linestyle='--', color='k')  # Bottom line
            axes.plot([x1, x1], [y1, y2], linewidth=3, alpha=0.5,  linestyle='--', color='k')  # Left line
            axes.plot([x2, x2], [y1, y2], linewidth=3, alpha=0.5,  linestyle='--', color='k')  # Right line
            label_text = 'CD' +' ' + str(round(best_score.item() * 100)) + "%"
            axes.text(x1, y2, label_text, color='k', fontsize=10, verticalalignment='bottom')

        voltage_window = {Xdata.name.split('_')[-1]: (x1,x2), Ydata.name.split('_')[-1]:(y1,y2)}
        print(f"Suggested voltage window: {voltage_window}")

        range_X = voltage_window[Xdata.name.split('_')[-1]]
        range_Y = voltage_window[Ydata.name.split('_')[-1]]
        windowed_data = data[
            (data[Xdata.name] >= range_X[0]) & (data[Xdata.name] <= range_X[1]) &
            (data[Ydata.name] >= range_Y[0]) & (data[Ydata.name] <= range_Y[1])
        ]

        window_data_image, Xdata, Ydata = convert_data_to_image(windowed_data)
        window_data_image = window_data_image[:,:,0]
        edges = canny(window_data_image,sigma=0.5, low_threshold=0.1*np.iinfo(np.uint8).max, high_threshold=0.3 * np.iinfo(np.uint8).max)
        lines = probabilistic_hough_line(edges, threshold=0, line_length=3,
                                        line_gap=0)
        
        if plot_process:
            # Generating figure 2
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax = ax.ravel()

            ax[0].imshow(window_data_image, cmap=cm.gray, origin='lower', extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()],)
            ax[0].set_title('Input image')

            ax[1].imshow(edges, cmap=cm.gray, origin='lower',  extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()],)
            ax[1].set_title('Masked Canny edges')

        potential_points = {}
        angles_data = []
        slopes_data = []
        for line in lines:
            p0_pixel, p1_pixel = line
            p0, p1 = pixel_polygon_to_image_units(line, windowed_data)

            dy =  (p1[1]-p0[1])
            dx = (p1[0]-p0[0])
            if dx == 0:
                continue
            m = dy/dx
            theta = np.arctan(m)*(180/np.pi)
            if theta > -40 or theta < -60:
                continue
            angles_data.append(theta)
            slopes_data.append(m)
            midpoint_pixel = (np.array(p0_pixel) + np.array(p1_pixel))/2
            midpoint_units = (np.array(p0) + np.array(p1))/2
            # print(midpoint)
            midpoint_pixel = midpoint_pixel.astype(int)

            X_name,Y_name,Z_name = windowed_data.columns[:3]
            current_at_midpoint = windowed_data[Z_name].to_numpy().reshape(len(Xdata), len(Ydata))[midpoint_pixel[0],midpoint_pixel[1]]
            potential_points[tuple(midpoint_units)] = current_at_midpoint

            if plot_process: 
                ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
                ax[1].scatter([midpoint_units[0]],[midpoint_units[1]], marker='*',s=50)
                ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]))
                ax[0].scatter([midpoint_units[0]],[midpoint_units[1]], marker='*',s=50)

        if plot_process:      
            ax[1].set_title('Hough Transform')
            ax[1].set_axis_off()

            ax[1].set_title('Histogram of Detected Line Angles')
            ax[2].hist(angles_data, bins=2*int(np.sqrt(len(slopes_data))))
            ax[2].set_xlabel(r"$\theta^\circ$")
            ax[2].set_ylabel(r"$f$")

        max_key = np.array(max(potential_points, key=potential_points.get))
        bias_point = {Xdata.name.split('_')[-1]: max_key[0], Ydata.name.split('_')[-1]: max_key[1]}
        axes.scatter(*max_key, marker='*', s=30, c='k', label='Bias Point')
        axes.legend(loc='best')
        print(f"Suggested bias point: {bias_point}")

        return bias_point, voltage_window

    def extract_plunger_value(self,
                              data):
        pass

    def extract_lever_arms(self,
                           data):
        pass

class DataAcquisition:
    def __init__(self,
                 station_config_yaml: str = None,
                 setup_config_yaml: str = None) -> None:
        pass

    def one_dimensional_sweep(self):
        pass

    def two_dimensional_sweep(self): 
        pass

    def n_dimensional_sweep(self):
        pass

    def time_trace(self):
        pass

    def set_voltage(self):
        pass

    def read_voltage(self):
        pass

    def read_current(self):
        pass

class QuantumDotFET:
    """Dedicated class to tune simple FET devices.
    """

    def __init__(self, 
                 device_config: str, 
                 setup_config: str,
                 tuner_config: str,
                 station_config: str,
                 save_dir: str) -> None:
        """Initializes the tuner.

        Args:
            device_config (str): Path to .yaml file containing device information.
            setup_config (str): Path to .yaml file containing experimental setup information.
            tuner_config (str): Path to .yaml file containing tuner information.
            station_config (str): Path to .yaml file containing QCoDeS station information
            save_dir (str): Directory to save data and plots generated.
        """
        self.DataFit = DataFit()
        self.DataAnalyzer = FET_DataAnalyzer(tuner_config)

        # Save file names
        self.device_config = device_config
        self.setup_config = setup_config
        self.tuner_config = tuner_config
        self.station_config = station_config
        self.save_dir = save_dir

        # Read in tuner config information
        self.tuner_info = yaml.safe_load(Path(tuner_config).read_text())
        self.global_turn_on_info = self.tuner_info['global_turn_on']
        self.pinch_off_info = self.tuner_info['pinch_off']

        # Read in station config information
        self.station_info = yaml.safe_load(Path(setup_config).read_text())
        self.general_info = self.station_info['general_info']
        self.multimeter_device = self.general_info['multimeter_device']
        self.voltage_device = self.general_info['voltage_device']
        self.preamp_sensitivity = self.general_info['preamp_sensitivity']
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
        self.station = qc.Station(config_file=station_config)
        self.station.load_instrument(self.voltage_device)
        self.station.load_instrument(self.multimeter_device)
        self.voltage_source = getattr(self.station, self.voltage_device)
        self.drain_mm_device = getattr(self.station, self.multimeter_device)
        self.drain_volt = getattr(self.station, self.multimeter_device).volt
        print("done!")

        print("Grounding device ... ", end=' ')
        self.ground_device()
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
        shutil.copy(self.station_config, self.db_folder)
        shutil.copy(self.device_config, self.db_folder)
        shutil.copy(self.tuner_config, self.db_folder)
        shutil.copy(self.setup_config, self.db_folder)
        print("done!")

    def ground_device(self):
        """Grounds all of the gates in the device.
        """
        device_gates = self.ohmics + self.all_gates
        self._smoothly_set_gates_to_voltage(device_gates, 0.)

    def bias_ohmic(self, 
                   ohmic: str = None, 
                   V: float = 0):
        """Biases the ohmic to desired voltage. Reports back to the user what the
        device will "see" based on the voltage divider in setup_config.yaml.

        Args:
            ohmic (str, optional): Ohmic gate name. Defaults to None.
            V (float, optional): Desired voltage (V). Defaults to 0.

        Returns: 
            None

        **Example**

        Setting the source module 'S' on a  SET (Single Electron Transistor) QD device,

        >>> QD_FET_Tuner.bias_ohmic(ohmic='S', V=0.005)
        Setting ohmic 'S' to 0.005 V (which is 0.05 mV) ... done!
        """     
        print(
            f"Setting ohmic {ohmic} to {V} V (which is {round(V*self.voltage_divider*1e3,3)} mV) ... ", 
            end=" "
        )
        self._smoothly_set_gates_to_voltage(ohmic, V)
        self.ohmic_bias = V*self.voltage_divider
        print("done!")

    def turn_on(self, 
                minV: float = 0.0, 
                maxV: float = None, 
                dV: float = None, 
                delay: float = 0.01) -> pd.DataFrame:
        """Attempts to 'turn-on' the FET by sweeping barriers and leads
        in the FET channel until either,
         (1) Maximum allowed current is reached 
         (2) Maximum allowed gate voltage is hit.

        Args:
            minV (float, optional): Starting sweep voltage (V). Defaults to 0.0.
            maxV (float, optional): Final sweep voltage (V). Defaults to None.
            dV (float, optional): Step size (V). Defaults to None.
            delay (float, optional): Delay between each step in the sweep (s). Defaults to 0.01.

        Returns: 
            df (pd.DataFrame): Return measurement data. 

        **Example**

        Turning on an SET (Single Electron Transistor) QD device,
        >>> QD_FET_Tuner.turn_on_device(minV = 0., maxV = None, dV = 0.05)
        """

        # Default dV and maxV based on setup_config and device_config
        if dV is None:
            dV = self.voltage_resolution

        if maxV is None:
            maxV = self.voltage_sign * self.abs_max_gate_voltage

        # Ensure we stay in the allowed voltage space 
        assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, "Double check the sign of the gate voltage (minV) for your given device."

        # Set up gate sweeps
        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        gates_involved = self.barriers + self.leads

        print(f"Setting all gates involved ({gates_involved}) to {minV} V ... ", end=" ")
        self._smoothly_set_gates_to_voltage(gates_involved, minV)
        print("done!")

        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'{gate_name}'), minV, maxV, num_steps, delay, get_after_set=False)
            )

        # Execute the measurement
        print(f"Beginning sweep from {minV} V to {maxV} V ... ", end=" ")
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
        print(f"done!")

        # Get last dataset recorded, convert to units of current (A)
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. gates involved (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_{gates_involved[0]}', marker= 'o',s=10)
        axes.set_title('Global Device Turn-On')
        df_current.plot.line(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_{gates_involved[0]}', ax=axes, linewidth=1)
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.global_turn_on_info['abs_min_current'], alpha=0.5, c='g', linestyle=':', label=r'$I_{\min}$')
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_device}_current'].iloc[-1])*self.abs_max_current, alpha=0.5, c='g', linestyle='--', label=r'$I_{\max}$')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$V_{GATES}$ (V)')
        axes.legend(loc='best')

        # Keep any data above the minimum current threshold
        mask = df_current[f'{self.multimeter_device}_current'].abs() > self.global_turn_on_info['abs_min_current'] 
        X = df_current[f'{self.voltage_device}_{gates_involved[0]}']
        Y = df_current[f'{self.multimeter_device}_current']
        X_masked = X[mask]
        Y_masked = Y[mask]

        # Try fitting data to tuner config information
        if len(mask) <= 4 or len(X_masked) <= 4:
            print("Insufficient points above minimum current threshold to do any fitting!")
        else:
            try:
                guess = [Y_masked.iloc[0], self.voltage_sign, X_masked.iloc[-1] - self.voltage_sign, 0]

                fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFit, self.global_turn_on_info['fit_function']), X_masked, Y_masked, guess)
                
                # Extract turn on voltage and saturation voltage
                a, b, x0, y0 = fit_params
                if self.global_turn_on_info['fit_function'] == 'exponential':
                    V_turn_on =  np.round(np.log(-y0/a)/b + x0, 3)
                elif self.global_turn_on_info['fit_function'] == 'logarithmic':
                    V_turn_on = np.round(np.exp(-y0/a)/b + x0, 3)
                V_sat = df_current[f'{self.voltage_device}_{gates_involved[0]}'].iloc[-2] 

                # Plot / print results to user
                axes.plot(X_masked, getattr(self.DataFit, self.global_turn_on_info['fit_function'])(X_masked, a, b, x0, y0), 'r-')
                axes.axvline(x=V_turn_on, alpha=0.5, linestyle=':',c='b',label=r'$V_{\min}$')
                axes.axvline(x=V_sat,alpha=0.5, linestyle='--',c='b',label=r'$V_{\max}$')
                axes.legend(loc='best')

                print(f"FET turns on at {V_turn_on} V.")
                print(f"FET saturates at {V_sat} V.")

                # Store in device information for later
                self.device_info['properties']['turn_on']['voltage'] = float(V_turn_on)
                self.device_info['properties']['turn_on']['saturation_voltage'] = float(V_sat)
                I_measured = np.abs(self._get_drain_current())
                R_measured = round(self.ohmic_bias / I_measured,3)
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

        return df

    def pinch_off(self, 
                    gates: List[str] | str = None, 
                    minV: float = None, 
                    maxV: float = None, 
                    dV: float = None,
                    delay: float = 0.01,
                    voltage_configuration: dict = {}) -> pd.DataFrame:
        """Attempts to pinch off gates from maxV to minV in steps of dV. If gates
        is not provided, defaults to barriers and leads in the FET channel. If minV is not 
        provided, defaults to saturation voltage minus allowed gate differential. If maxV
        is not provided defaults to saturation voltage.

        Args:
            gates (list | str, optional): Gates to pinch off. Defaults to None.
            minV (float, optional): Minimum sweep voltage (V). Defaults to None.
            maxV (float, optional): Maximum sweep voltage (V). Defaults to None.
            dV (float, optional): Step size (V). Defaults to None.
            delay (float, optional): Delay between each step in the sweep (s). Defaults to 0.01.
            voltage_configuration (dict, optional): Desired voltage configuration. Defaults to None.

        Returns:
            df (pd.DataFrame): Return measurement data. 

        **Example**

        Pinching off barriers in a SET (Single Electron Transistor) QD device,
        >>> QD_FET_Tuner.pinch_off(gates=['LB','RB'], minV=1, maxV=None, dV=0.005)
        """

        assert self.deviceTurnsOn, "Device does not turn on. Why are you pinching anything off?"

        # Bring device to voltage configuration
        if voltage_configuration is not None:
            print("Setting voltage configuration ... ", end = " ")
            self._smoothly_set_voltage_configuration(voltage_configuration)
            print("Done!")

        if dV is None:
            dV = self.voltage_resolution

        if gates is None:
            gates = self.barriers + self.leads
        if type(gates) is str:
            gates = [gates]

        if maxV is None:
            maxV = self.device_info['properties']['turn_on']['saturation_voltage']
        else:
            assert np.sign(maxV) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for your given device."
        
        if minV is None:
            if self.voltage_sign == 1:
                # minV = max(0, round(self.device_info['properties']['turn_on']['saturation_voltage'] - self.abs_max_gate_differential, 3))
                min_allowed = 0
                max_allowed = None
            elif self.voltage_sign == -1:
                # minV = min(0, round(self.device_info['properties']['turn_on']['saturation_voltage'] - self.abs_max_gate_differential, 3))
                min_allowed = None
                max_allowed = 0
            minV = np.clip(
                    a=round(self.device_info['properties']['turn_on']['saturation_voltage'] - self.voltage_sign * self.abs_max_gate_differential, 3),
                    a_min=min_allowed,
                    a_max=max_allowed
                )
        else:
            assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, "Double check the sign of the gate voltage (minV) for your given device."

        print(f"Settings gates involved ({gates}) to {maxV} V ... ", end = " ")
        self._smoothly_set_gates_to_voltage(gates, maxV)
        print("done!")

        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        sweep_list = []
        for gate_name in gates:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'{gate_name}'), maxV, minV, num_steps, delay, get_after_set=False)
            )

        def adjusted_break_condition():
            return np.abs(self._get_drain_current()) < self.pinch_off_info['abs_min_current']

        for sweep in sweep_list:

            print(f"Attempting to pinch off gate {str(sweep._param).split('_')[-1]} from {maxV} V to {minV} V ... ", end=" ")
            result = qc.dataset.dond(
                sweep,
                self.drain_volt,
                # break_condition=adjusted_break_condition,
                measurement_name='{} Pinch Off'.format(str(sweep._param).split('_')[-1]),
                exp=self.initialization_exp,
                show_progress=True
            )   
            print(f"done!")

            print(f"Returning gate {str(sweep._param).split('_')[-1]} back to {maxV} V ... ", end=" ")
            self._smoothly_set_gates_to_voltage([str(sweep._param).split('_')[-1]], maxV)
            print(f"done!")

            # Get last dataset recorded, convert to current units
            dataset = qc.load_last_experiment().last_data_set()
            df = dataset.to_pandas_dataframe().reset_index()
            df_current = df.copy()
            df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
            df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

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
                
            # Keep any data that is above the minimum turn-on current
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
                    fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFit, 'linear'), X, Y, guess)
                    m, b = fit_params
                    plt.plot(X, self.DataFit.linear(X,m,b), 'r-')
                    print("Fits well to line, most likely barrier is shorted somewhere.")

                except RuntimeError:
                    print("Error - fitting to \"linear\" failed.")

            else:
                # Try and fit and get fit params
                try: 
                    if str(sweep._param).split('_')[-1] in self.leads:

                        fit_function = getattr(self.DataFit, 'logarithmic')
                        guess = [Y_masked.iloc[0], self.voltage_sign, X_masked.iloc[-1] - self.voltage_sign, 0]
                    
                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params
                        
                        V_pinchoff = float(round(np.exp(-y0/a)/b + x0,3))
                        self.device_info['properties'][str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                   
                    else:

                        fit_function = getattr(self.DataFit, self.pinch_off_info['fit_function'])
                        guess = (Y.iloc[0], -1 * self.voltage_sign * 100, self.device_info['properties']['turn_on']['voltage'], 0)

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
                        fit_params, fit_cov = sp.optimize.curve_fit(getattr(self.DataFit, 'linear'), X, Y, guess)
                        m, b = fit_params
                        plt.plot(X, self.DataFit.linear(X,m,b), 'r-')
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
                        dV: float | tuple = None,
                        delay: float = 0.01, 
                        voltage_configuration: dict = None,
                        extract_bias_point: bool = False) -> tuple[pd.DataFrame, plt.Axes]:
        """Performs a 2D sweep of the barriers in the FET channel. If user doesn't
        provide bounds, it will perform a 2D sweep based on the voltage pinch off window
        determined earlier.

        Args:
            B1 (str, optional): Barrier gate to step. Defaults to None.
            B2 (str, optional): Barrier gate to sweep. Defaults to None.
            B1_bounds (tuple, optional): Voltage bounds of B1 (V). Defaults to (None, None).
            B2_bounds (tuple, optional): Voltage bounds of B2 (V). Defaults to (None, None).
            dV (float | tuple, optional): Step size (V). Defaults to None.
            delay (float, optional): Delay between each step in the sweep (s). Defaults to 0.01.
            voltage_configuration (dict, optional): Desired voltage configuration. Defaults to None.
            extract_bias_point (bool, optional): Will attempt to extract bias point. Defaults to False.

        Returns: 
            df (pd.DataFrame): Return measurement data. 
            ax1 (plt.Axes): Axis for the raw current data.


        Examples:
            
        """

        # Bring device to voltage configuration
        if voltage_configuration is not None:
            print("Setting voltage configuration ... ", end = " ")
            self._smoothly_set_voltage_configuration(voltage_configuration)
            print("Done!")

        # Parse dV from user
        if dV is None:
            dV_B1 = self.voltage_resolution
            dV_B2 = self.voltage_resolution
        elif type(dV) is float:
            dV_B1 = dV
            dV_B2 = dV
        elif type(dV) is tuple:
            dV_B1, dV_B2 = dV
        else:
            print("Invalid dV.")
            return

        # Double check barrier validity
        if B1 is None or B2 is None and len(self.barriers) == 2:
            B1, B2 = self.barriers
        
        # Double check device bounds
        minV_B1, maxV_B1 = B1_bounds
        minV_B2, maxV_B2 = B2_bounds

        if minV_B1 is None:
            minV_B1 = self.device_info['properties'][B1]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B1) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for B1."

        if minV_B2 is None:
            minV_B2 = self.device_info['properties'][B2]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B2) == self.voltage_sign, "Double check the sign of the gate voltage (minV) for B2."

        if maxV_B1 is None:
            if self.voltage_sign == 1:
                maxV_B1 = min(self.device_info['properties'][B1]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B1]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
            elif self.voltage_sign == -1:
                maxV_B1 = max(self.device_info['properties'][B1]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B1]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
        else:
            assert np.sign(maxV_B1) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for B1."

        if maxV_B2 is None:
            if self.voltage_sign == 1:
                maxV_B2 = min(self.device_info['properties'][B2]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B2]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
            elif self.voltage_sign == -1:
                maxV_B2 = max(self.device_info['properties'][B2]['pinch_off']['voltage']+self.voltage_sign*self.device_info['properties'][B2]['pinch_off']['width'], self.device_info['properties']['turn_on']['saturation_voltage'])
        else:
            assert np.sign(maxV_B2) == self.voltage_sign, "Double check the sign of the gate voltage (maxV) for B2."

        print(f"Setting lead gates ({self.leads}) to saturation gate voltage ({self.device_info['properties']['turn_on']['saturation_voltage']}) ... ", end = " ")
        self._smoothly_set_gates_to_voltage(self.leads, self.device_info['properties']['turn_on']['saturation_voltage'])
        print("done!")

        print(f"Setting {B1} to {maxV_B1} V ... ", end = " ")
        print(f"Setting {B2} to {maxV_B2} V ...", end = " ")
        self._smoothly_set_voltage_configuration({B1: maxV_B1, B2: maxV_B2})
        print(f"done!")

        def smooth_reset():
            """Resets the inner loop variable smoothly back to the starting value
            """
            self._smoothly_set_gates_to_voltage([B2], maxV_B2)

        print("Beginning 2D sweep ... ")
        print(f"Stepping {B1} from {maxV_B1} V to {minV_B1} V ...")
        print(f"Sweeping {B2} from {maxV_B2} V to {minV_B2} V ...")

        num_steps_B1 = self._calculate_num_of_steps(minV_B1, maxV_B1, dV_B1)
        num_steps_B2 = self._calculate_num_of_steps(minV_B2, maxV_B2, dV_B2)
        result = qc.dataset.do2d(
            getattr(self.voltage_source, f'{B1}'), # outer loop
            maxV_B1,
            minV_B1,
            num_steps_B1,
            delay,
            getattr(self.voltage_source, f'{B2}'), # inner loop
            maxV_B2,
            minV_B2,
            num_steps_B2,
            delay,
            self.drain_volt,
            after_inner_actions = [smooth_reset],
            set_before_sweep=True, 
            show_progress=True, 
            measurement_name='Barrier Barrier Sweep',
            exp=self.initialization_exp
        )
        print("done!")

        print(f"Returning gates {B1}, {B2} to {maxV_B1} V, {maxV_B2} V respectively ... ", end = " ")
        self._smoothly_set_gates_to_voltage([B1], maxV_B1)
        self._smoothly_set_gates_to_voltage([B2], maxV_B2)
        print("done!")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_device}_current', index=[f'{self.voltage_device}_{B1}'], columns=[f'{self.voltage_device}_{B2}'])
        B1_data, B2_data = df_pivoted.columns, df_pivoted.index
        raw_current_data = df_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        B1_grad = np.gradient(raw_current_data, axis=1)
        B2_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Barrier Barrier Sweep")

        im_ratio = 1

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

        if extract_bias_point:

            # Inference data
            bias_point, voltage_window = self.DataAnalyzer.extract_bias_point(
                df,
                plot_process=True,
                axes=ax1
            )

            for barrier_gate_name, voltage in bias_point.items():
                self.device_info['properties'][barrier_gate_name]['bias_point'] = voltage

            self._update_device_config_yaml()

        return (df,ax1)

    def coulomb_blockade(self, 
                         P: str = None, 
                         P_bounds: tuple = (None, None), 
                         dV: float = None, 
                         delay: float = 0.01,
                         voltage_configuration: dict = None) -> pd.DataFrame:
        """Attempts to sweep plunger gate to see coulomb blockade features. Ideally
        the voltage configuration provided should be that which creates a well defined
        central dot between the two barriers.

        Args:
            P (str, optional): Plunger gate. Defaults to None.
            P_bounds (tuple, optional): Voltage bounds (V). Defaults to (None, None).
            dV (float, optional): Step size (V). Defaults to None.
            delay (float, optional): Delay between each step in the sweep (s). Defaults to 0.01.
            voltage_configuration (dict, optional): Device gate voltages. Defaults to None.

        Returns:
            df (pd.DataFrame): Return measurement data. 

        **Example**
        >>> QD_FET_Tuner.coulomb_blockade(
            P='P', 
            P_bounds=(0,0.75), 
            voltage_configuration={
                'S': 0.005, 'LB': 0.55, 'RB': 0.7, 'STL': 2.4
            }, 
            dV=0.005
        )
        """
        assert voltage_configuration is not None, "Please provide a voltage configuration!"

        print("Setting voltage configuration ... ", end = " ")
        self._smoothly_set_voltage_configuration(voltage_configuration)
        print("Done!")

        if P is None:
            P = self.plungers[0]

        if dV is None:
            dV = self.voltage_resolution

        minV_P, maxV_P = P_bounds
        if minV_P is None:
            minV_P = 0
        assert maxV_P is not None, "Please specify maximum gate voltage."
  
        num_steps_P = self._calculate_num_of_steps(minV_P, maxV_P, dV)
        P_sweep = LinSweep_SIM928(getattr(self.voltage_source, f'{P}'), minV_P, maxV_P, num_steps_P, delay, get_after_set=False)

        print(f"Setting gate {P} to {minV_P} V ...", end = " ")
        self._smoothly_set_gates_to_voltage([P], minV_P)
        print("done!")

        print(f"Sweeping gate {P} from {minV_P} V to {maxV_P} V ...", end = " ")
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
        print("done!")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_{P}', marker= 'o',s=10)
        df_current.plot.line(y=f'{self.multimeter_device}_current', x=f'{self.voltage_device}_{P}', ax=axes, linewidth=1)
        axes.set_title('Coulomb Blockade')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=P))
        axes.legend(loc='best')

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{P}_sweep_{config_str}')

        return df

    def coulomb_diamonds(self, 
                         ohmic: str = None, 
                         gate: str = None, 
                         ohmic_bounds: tuple = (None, None),
                         gate_bounds: tuple = (None, None),
                         dV_ohmic: float = None, 
                         dV_gate: float = None,
                         delay: float = 0.01,
                         voltage_configuration: dict = None) -> tuple[pd.DataFrame, plt.Axes]:
        """Performs a bias spectroscopy of the device recovering Coulomb diamonds.

        Args:
            ohmic (str, optional): Ohmic gate name. Defaults to None.
            gate (str, optional): Gate to sweep. Defaults to None.
            ohmic_bounds (tuple, optional): Voltage bounds (V). Defaults to (None, None).
            gate_bounds (tuple, optional): Voltage bounds (V). Defaults to (None, None).
            dV_ohmic (float, optional): Step size (V). Defaults to None.
            dV_gate (float, optional): Step size (V). Defaults to None.
            delay (float, optional): Delay between each step in the sweep (s). Defaults to 0.01.
            voltage_configuration (dict, optional): Device voltage configuration. Defaults to None.

        Returns: 
            df (pd.DataFrame): Return measurement data. 

        **Example**
        >>>QD_FET_Tuner.coulomb_diamonds(
            ohmic='S', 
            gate='P', 
            ohmic_bounds=(-0.015, 0.015),
            gate_bounds=(0,0.75),
            dV_gate=0.003, 
            dV_ohmic=0.001
            voltage_configuration={'LB': 0.55, 'RB': 0.7},
        )
        """

        assert voltage_configuration is not None, "Please provide a voltage configuration!"

        print("Setting voltage configuration ... ", end = " ")
        self._smoothly_set_voltage_configuration(voltage_configuration)
        print("Done!")

        if dV_ohmic is None:
            dV_ohmic = self.voltage_resolution
        
        if dV_gate is None:
            dV_gate = self.voltage_resolution
        
        minV_ohmic, maxV_ohmic = ohmic_bounds
        minV_gate, maxV_gate = gate_bounds

        if minV_gate is None:
            print(f"Please provide a maximum gate voltage for {gate}")
            return

        if maxV_gate is None:
            print(f"Please provide a maximum gate voltage for {gate}")
            return
        
        if minV_ohmic is None:
            minV_ohmic = -1 * float(getattr(self.voltage_source, f"{ohmic}")())

        if maxV_ohmic is None:
            maxV_ohmic = float(getattr(self.voltage_source, f"{ohmic}")())
   
        def smooth_reset():
            self._smoothly_set_gates_to_voltage([ohmic], maxV_ohmic)

        print("Beginning 2D sweep ... ")
        print(f"Stepping {gate} from {maxV_gate} V to {minV_gate}...")
        print(f"Sweeping {ohmic} from {maxV_ohmic} V to {minV_ohmic}...")
        num_steps_ohmic = self._calculate_num_of_steps(minV_ohmic, maxV_ohmic, dV_ohmic)
        num_steps_gate = self._calculate_num_of_steps(minV_gate, maxV_gate, dV_gate)
        result = qc.dataset.do2d(
            getattr(self.voltage_source, f'{gate}'), # outer loop
            maxV_gate,
            minV_gate,
            num_steps_gate,
            delay,
            getattr(self.voltage_source, f'{ohmic}'), # inner loop
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

        print(f"Settings gates {gate}, {ohmic} to {maxV_gate} V, {maxV_ohmic} V respectively ... ", end = " ")
        self._smoothly_set_gates_to_voltage([gate], maxV_gate)
        self._smoothly_set_gates_to_voltage([ohmic], maxV_ohmic)
        print("done!")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_device}_current', index=[f'{self.voltage_device}_{gate}'], columns=[f'{self.voltage_device}_{ohmic}'])
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

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{ohmic}_{gate}_sweep_{config_str}')
        
        return (pd.DataFrame, ax1)

    def current_trace(self, 
                      f_sampling: int, 
                      t_capture: int, 
                      plot_psd: bool = False) -> pd.DataFrame:
        """Records current data from multimeter device at a given sampling rate
        for a given amount of time. 

        Args:
            f_sampling (int): Sampling rate (Hz)
            t_capture (int): Capture time (s)
            plot_psd (bool, optional): Plots power spectral density spectrum. Defaults to "False".
        
        Returns:
            df (pd.DataFrame): Return measurement data. 

        **Example**
        >>>QD_FET_Tuner.current_trace(
            f_sampling=1000,
            t_capture=60, 
            plot_psd=True
        )
        """

        time_param = qc.parameters.ElapsedTimeParameter('time')
        meas = qc.dataset.Measurement(exp=self.initialization_exp)
        meas.register_parameter(time_param)
        meas.register_parameter(self.drain_mm_device.volt, setpoints=[time_param])

        with meas.run() as datasaver:
            time_param.reset_clock()
            elapsed_time = 0
            while elapsed_time < t_capture:
                time.sleep(1/f_sampling)
                datasaver.add_result((self.drain_mm_device.volt, self.drain_mm_device.volt()),
                                (time_param, time_param()))
                elapsed_time = time_param.get()
      
        df = datasaver.dataset.to_pandas_dataframe().reset_index()
        df_current = df.rename(columns={f'{self.multimeter_device}_volt': f'{self.multimeter_device}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_device}_current', x='time', marker= 'o',s=5)
        df_current.plot.line(y=f'{self.multimeter_device}_current', x=f'time', ax=axes, linewidth=1)
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$t$ (s)')
        axes.set_title(rf'Current noise, $f_s={f_sampling}$ Hz, $t_\max={t_capture}$ s')
        plt.show()
        
        if plot_psd:
            # Plot noise spectrum
            t = df_current[f'time']
            I = df_current[f'{self.multimeter_device}_current']
            f, Pxx = sp.signal.periodogram(I, fs=f_sampling, scaling='density')

            plt.loglog(f, Pxx)
            plt.xlabel(r'$\omega$ (Hz)')
            plt.ylabel(r'$S_I$ (A$^2$/Hz)')
            plt.title(r"Current noise spectrum")
            plt.show()

        return df

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

    def _get_drain_current(self):
        """Reads the multimeter and gets the current reading.

        Returns:
            float: Current reading (A)
        """
        raw_voltage = self.drain_volt()
        current = float(self.preamp_sensitivity * (raw_voltage - self.preamp_bias))
        return current

    def _smoothly_set_gates_to_voltage(self, 
                              gates: List[str] | str, 
                              voltage: float):
        """Sets gates to desired voltage

        Args:
            gates (list | str): List of gate names
            voltage (float): Desired voltage (V)
        """
        if isinstance(gates, str):
            gates = [gates]
        self._smoothly_set_voltage_configuration(
            dict(zip(gates, [voltage]*len(gates)))
        )

    def _smoothly_set_voltage_configuration(self,
                                    voltage_configuration: Dict[str, float] = {},
                                    equitime: bool = False):


        gates_to_check = self.barriers + self.leads
        for gate_name in gates_to_check:
            if gate_name in voltage_configuration.keys():
                assert np.sign(voltage_configuration[gate_name]) == np.sign(self.voltage_sign) or np.sign(voltage_configuration[gate_name]) == 0, f"Check voltage sign on {gate_name}"

        intermediate = []
        if equitime:
            maxsteps = 0
            deltav = {}
            for gate_name, voltage in voltage_configuration.items():
                deltav[gate_name] = voltage-getattr(self.voltage_source, gate_name).get()
                gate_step_param = getattr(self.voltage_source, f'{gate_name}_step', None)
                if gate_step_param is None:
                    # work around for IVVI
                    stepsize = getattr(self.voltage_source, f'{gate_name}', None).step
                else:
                    stepsize = gate_step_param()
                steps = abs(int(np.ceil(deltav[gate_name]/stepsize)))
                if steps > maxsteps:
                    maxsteps = steps
            for s in range(maxsteps):
                intermediate.append({})
                for gate_name, voltage in voltage_configuration.items():
                    intermediate[-1][gate_name] = voltage - \
                                          deltav[i]*(maxsteps-s-1)/maxsteps
        else:
            done = []
            prevvals = {}
            for gate_name, voltage in voltage_configuration.items():
                prevvals[gate_name] = getattr(self.voltage_source, gate_name).get()
                
            while len(done) != len(voltage_configuration):
                intermediate.append({})
                for gate_name, voltage in voltage_configuration.items():
                    if gate_name in done:
                        continue
                    gate_step_param = getattr(self.voltage_source, f'{gate_name}_step', None)
                    if gate_step_param is None:
                        # work around for IVVI
                        stepsize = getattr(self.voltage_source, f'{gate_name}', None).step
                    else:
                        stepsize = gate_step_param()
                    deltav = voltage-prevvals[gate_name]
                    if abs(deltav) <= stepsize:
                        intermediate[-1][gate_name] = voltage
                        done.append(gate_name)
                    elif deltav > 0:
                        intermediate[-1][gate_name] = prevvals[gate_name] + stepsize
                    else:
                        intermediate[-1][gate_name] = prevvals[gate_name] - stepsize
                    prevvals[gate_name] = intermediate[-1][gate_name]

        for voltages in intermediate:
            for gate_name in voltages:
                getattr(self.voltage_source, gate_name).set(voltages[gate_name])
            delay_param = getattr(self.voltage_source, 'smooth_timestep', None)
            if delay_param is None:
                time.sleep(getattr(self.voltage_source, 'dac_set_sleep', None)())
            else:
                time.sleep(delay_param())

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

    def _calculate_num_of_steps(self, 
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
    
    def _update_device_config_yaml(self):
        """Updates the device_config.yaml file based on findings.
        """
        with open(self.device_config, 'w') as outfile:
            yaml.dump(self.device_info, outfile, default_flow_style=True)

    def _save_figure(self, plot_info: str, plot_dpi: int = 300):
        """Saves plot generated from one of the stages.

        Args:
            plot_info (str): Plot information for the filename.
            plot_dpi (int, optional): Dots Per Inch (DPI) for the plot. Defaults to 300.
        """
        completition_time = str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) # the current minute
        plot_name = os.path.join(self.db_folder, f"{plot_info}_{completition_time}.png")
        plt.savefig(fname=plot_name, dpi=plot_dpi)