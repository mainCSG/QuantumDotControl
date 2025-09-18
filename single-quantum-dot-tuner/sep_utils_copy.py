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
from qcodes.dataset import AbstractSweep
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

original_sys_path = sys.path.copy()

try:

    sys.path.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../coarse_tuning/src')
        )
    )

    from inference import *

except ModuleNotFoundError: 

    print("Detectron2 not found on the system cannot run any segmentation models.")

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
    
class DataAcquisition:
    
    def __init__(self):
        pass

    def set_voltage(self):
        pass

    def measure_current(self):
        pass

    def one_dimension_sweep(self):
        pass

    def two_dimension_sweep(self):
        pass

    def N_dimension_sweep(self):
        pass

    def time_trace(self):
        pass

class DataAnalysis:
    
    def __init__(self, tuner_config) -> None:
        
        self.tuner_info = yaml.safe_load(Path(tuner_config).read_text())

        self.model_path = self.tuner_info['barrier_barrier']['segmentation_model_path']
        self.model_config_path = self.tuner_info['barrier_barrier']['segmentation_model_config_path']
        self.model_name =self.tuner_info['barrier_barrier']['segmentation_model_name']
        self.model_processor = self.tuner_info['barrier_barrier']['segmentation_model_processor']
        self.confidence_threshold = self.tuner_info['barrier_barrier']['segmentation_confidence_threshold']
        self.polygon_threshold = self.tuner_info['barrier_barrier']['segmentation_polygon_threshold']
        self.segmentation_class = self.tuner_info['barrier_barrier']['segmentation_class']
  
    def logarithmic(self, x, a, b, x0, y0):
        return a * np.log(b*(x-x0)) + y0

    def exponential(self, x, a, b, x0, y0):
        return a * np.exp(b * (x-x0)) + y0

    def sigmoid(self, x, a, b, x0, y0):
        return a/(1+np.exp(b * (x-x0))) + y0
    
    def linear(self, x, m, b):
        return m * x + b         
    
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
            plot=plot_process
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
            if theta > -40 or theta < -50:
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

    def extract_max_conductance_point(self,
                                      data: pd.DataFrame,
                                      plot_process: bool = False,
                                      sigma: float = 0.5) -> dict:

        V_name, I_name = data.columns

        data = data.rename(
            columns={I_name: '{}_current'.format(I_name.split('_')[0])}
            )
        data.iloc[:,-1] = data.iloc[:,-1].subtract(0).mul(1e-7) # sensitivity
        
        V_name, I_name = data.columns

        I_data = data[I_name]
        V_data = data[V_name]

        I_filtered = sp.ndimage.gaussian_filter1d(
            I_data, sigma
        )

        G_data = np.gradient(I_filtered, np.abs(V_data.iloc[-1] - V_data.iloc[-2]))
        G_filtered = sp.ndimage.gaussian_filter1d(
            G_data, sigma
        )

        threshold = max(G_filtered) / 10

        maxima = sp.signal.argrelextrema(G_filtered, np.greater)[0]
        maxima_indices = maxima[G_filtered[maxima] >= threshold]

        if len(maxima_indices) != 0:
            results = dict(zip(V_data.iloc[maxima_indices], G_filtered[maxima_indices]))

            results_sorted = dict(sorted(results.items(), key=lambda item: item[1]))

        if plot_process:

            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            # Plot for I_SD vs V_ST
            ax1.set_title(r"$I_{SD}$")
            ax1.set_ylabel(r"$I_{SD}\ (A)$")
            ax1.plot(V_data, I_data, 'k-', alpha=0.3,linewidth=0.75)
            ax1.plot(V_data, I_filtered, 'k-', linewidth=1.5)

            # Plot for G vs V_ST
            ax2.set_title(r"$G_{SD}$")
            ax2.set_ylabel(r"$G_{SD}\ (S)$")
            ax2.set_xlabel(r"$V_{P}\ (V)$")
            ax2.plot(V_data, G_data, 'k-', alpha=0.3, linewidth=0.75)
            ax2.plot(V_data, G_filtered, 'k-', linewidth=1.5)
            ax2.hlines(y=threshold, xmin=V_data.iloc[0], xmax=V_data.iloc[-1], color='black', linestyle='--', linewidth=1.5)

        def legend_without_duplicate_labels(ax):
                # Helper function to prevent duplicates in the legend.
                handles, labels = ax.get_legend_handles_labels()
                unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
                ax.legend(*zip(*unique))

        # Plot all the points of interest according to their sensitivity
        if len(maxima_indices) > 0:
            for i in maxima_indices:

                index = list(results_sorted.keys()).index(V_data.iloc[i])
                if index == len(results_sorted)-1:
                    label = "High Sensitivity"
                    color = 'b'
                elif index == 0: 
                    label = "Low Sensitivity"
                    color = 'r'
                else:
                    label = "Medium Sensitivity"
                    color = 'g'

                if plot_process:
                    ax1.text(V_data.iloc[i], I_filtered[i], f'{V_data.iloc[i]:.2f} mV', color='black', fontsize=6, fontweight=750, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.25))
                    ax2.scatter(V_data.iloc[i], G_filtered[i], c=color, label=label)
                    ax1.scatter(V_data.iloc[i], I_filtered[i], c=color, label=label)
                    ax1.legend(loc='best')
                    legend_without_duplicate_labels(ax1)

        V_P_high, G_high = list(results_sorted.items())[-1]
        V_P_med, G_med = list(results_sorted.items())[int(len(results_sorted.items())//2)]
        V_P_low, G_low = list(results_sorted.items())[0]

        gate_name = V_name.split('_')[-1]
        return {'high': {gate_name: V_P_high, 'conductance': G_high},
                'medium': {gate_name: V_P_med, 'conductance': G_med},
                'low': {gate_name: V_P_low, 'conductance': G_low}}

    def extract_lever_arms(self,
                           data: pd.DataFrame,
                           plot_process: bool = False) -> dict:
        
        # Load in data and seperate 
        X_name, Y_name, Z_name = data.columns
        Xdata, Ydata = np.unique(data[X_name]), np.unique(data[Y_name])

        df_pivoted = data.pivot_table(values=Z_name, index=Y_name, columns=X_name).fillna(0)
        Zdata = df_pivoted.to_numpy()

        # Calculate conductance where G = dI / dVp 
        G = np.gradient(Zdata)[1]

        if plot_process:
            plt.imshow(G, origin='lower', extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()], aspect=(Xdata.max() - Xdata.min())/(Ydata.max() - Ydata.min()))
            plt.title("Transconductance")
            plt.colorbar()
            plt.show()
            
        # Apply filter to bring out edges better
        def U(x,y):
            sigX, sigY = 5,5
            return (1/(2 * np.pi * sigX * sigY)) * np.exp(- 0.5* ((x/sigX)**2 + (y/sigY)**2))
        def adjusted(G,G0):
            return np.sign(G) * np.log((np.abs(G)/G0) + 1)
        def F(U, G, G0):
            # G = adjusted(G,G0)
            return (G - convolve(G,U)) / np.sqrt((convolve(G,U))**2 + G0**2)

        N=2
        U_kernal = np.array([[U(x, y) for y in range(-(N-1)//2,(N-1)//2 + 1)] for x in range(-(N-1)//2,(N-1)//2 + 1)])
        cond_quant = 3.25 * 1e-5
        filtered_G = np.abs(F(U_kernal, G, G0=10**-7 * cond_quant))

        if plot_process:
            plt.imshow(filtered_G, origin='lower', extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()], aspect=(Xdata.max() - Xdata.min())/(Ydata.max() - Ydata.min()))
            plt.title("Filtered Transconductance")
            plt.colorbar()
            plt.show()

        # Apply binary threshold to bring out diamonds better
        thresh = threshold_otsu(filtered_G)
        binary_image = filtered_G < thresh

        if plot_process:
            plt.imshow(binary_image, origin='lower', extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()], aspect=(Xdata.max() - Xdata.min())/(Ydata.max() - Ydata.min()))
            plt.title("Filtered Transconductance Binary")
            plt.colorbar()
            plt.show()

        # Erode any artifacts and keep just the diamond shapes
        footprint = rectangle(13, 6)
        erode = skimage.morphology.erosion(binary_image,footprint)

        footprint = diamond(1)
        erode = skimage.morphology.erosion(erode,footprint)
        
        if plot_process:
            plt.imshow(erode, origin='lower', extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()], aspect=(Xdata.max() - Xdata.min())/(Ydata.max() - Ydata.min()))
            plt.title("Filtered Transconductance Binary Eroded")
            plt.show()

        # Attempt to find contours
        contours = skimage.measure.find_contours(erode, 0.8)

        if len(contours) == 0:
            return 
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()

        ax.imshow(Zdata, origin='lower', extent=[Xdata.min(), Xdata.max(), Ydata.min() , Ydata.max()], aspect=(Xdata.max() - Xdata.min())/(Ydata.max() - Ydata.min()))
        ax.set_title(r'$I_{SD}$')
        ax.set_ylabel(r'$V_{SD}$ (V)')
        ax.set_xlabel(r'$V_{P}$ (V)')
        ax.set_aspect('auto')

        addition_voltages = []
        charging_voltages = []
        results = {}


        for i, contour in enumerate(contours):
            if len(contour) < 350: 
                continue

            # Convert to proper units for calculations
            image_units = []
            for coordinate in contour:
                image_units.append([Ydata[int(coordinate[0])], Xdata[int(coordinate[1])]])
            image_units = np.array(image_units)
            
            Y = image_units[:,0]
            X = image_units[:,1]

            Xmax = max(X)
            Xmin = min(X)
            Ymax = max(Y)
            Ymin = min(Y)

            # Get centroid
            centroidX, centroidY = 0.5*(Xmax + Xmin), 0.5 * (Ymax + Ymin)

            dX = Xmax - Xmin
            dY = Ymax - Ymin

            divider = 1e-3
            alpha= (Ymax * divider /2) / dX

            e = 1.60217663e-19 # C

            eps0 = 8.8541878128e-12 # F/m
            epsR = 11.7 # Silicon

            Vadd = Xmax - Xmin # V
            Vc = dY * divider /2 # V
            addition_voltages += [Vadd]
            charging_voltages += [Vc] 
            C_P = e / Vadd # F
            C_sigma = e / Vc # F
            dot_size = C_sigma / (8 * eps0 * epsR) # m
            alpha = (dY * divider /2) / dX # eV/V

            results[i]= {
                'centroid': (centroidX, centroidY), 
                'Vadd': Vadd, 
                'Vcharge': Vc, 
                'Cp': C_P,
                'CSigma': C_sigma,
                'lever arm': alpha,
                'dot size': dot_size
                }

            ax.plot(image_units[:, 1], image_units[:, 0], linewidth=1, linestyle='-', c='k')
            label_text = r'$\alpha$ =' + str(round(alpha,3))
            ax.text(0.98*centroidX, 1.2 * Ymax, label_text, color='k', fontsize=8, verticalalignment='bottom')

            label_text = r'$V_{add}$ =' + str(round(Vadd*1e3,1)) + 'mV'
            ax.text(0.95*centroidX, 1.3 * Ymin, label_text, color='k', fontsize=8, verticalalignment='bottom')

            label_text = r'$V_{charge}$ =' + str(round(Vc * 1e3,1)) + 'mV'
            ax.text(0.95*centroidX, 1.5 * Ymin, label_text, color='k', fontsize=8, verticalalignment='bottom')

            label_text = r'$C_{P}$ =' + str(round((e / Vadd) * 1e18,2)) + 'aF'
            ax.text(0.95*centroidX, 1.7 * Ymin, label_text, color='k', fontsize=8, verticalalignment='bottom')

            label_text = r'$C_{\Sigma}$ =' + str(round((e / Vc) * 1e18,2)) + 'aF'
            ax.text(0.95*centroidX, 1.9 * Ymin, label_text, color='k', fontsize=8, verticalalignment='bottom')
            ax.scatter([centroidX], [centroidY], marker='*', s=30, c='k')

            label_text = r'$R_{dot}$ =' + str(round(dot_size * 1e9,2)) + 'nm'
            ax.text(0.95*centroidX, 2.1 * Ymin, label_text, color='k', fontsize=8, verticalalignment='bottom')
            ax.scatter([centroidX], [centroidY], marker='*', s=30, c='k')

        plt.show()
        return results

class QuantumDotSEP(DataAnalysis, DataAcquisition):
    
    """Dedicated class to tune simple SEP devices.
    """

    def __init__(self, 
                 config: str, 
                 tuner_config: str,
                 station_config: str,
                 save_dir: str) -> None:
        
        """
        Initializes the tuner.

        Args:
            config (str): Path to .yaml file containing device information.
            setup_config (str): Path to .yaml file containing experimental setup information.
            tuner_config (str): Path to .yaml file containing tuner information.
            station_config (str): Path to .yaml file containing QCoDeS station information
            save_dir (str): Directory to save data and plots generated.
        """

        DataAnalysis.__init__(self, tuner_config=tuner_config)
        DataAcquisition.__init__(self)
    
        # First, we save the config file data and the data directory, as well as loading the config files

        self.config_file = config
        self.tuner_config_file = tuner_config
        self.station_config_file = station_config
        self.save_dir = save_dir

        self._load_config_files()
        
        # Then, we set up the date and file where data is stored

        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        self.db_folder = os.path.join(save_dir, f"{self.config['device']['characteristics']['name']}_{todays_date}")
        os.makedirs(self.db_folder, exist_ok=True)

        # This code creates new logging categories for the user to see while the code runs

        ATTEMPT, COMPLETE, IN_PROGRESS = logging.INFO - 2, logging.INFO - 1, logging.INFO

        logging.addLevelName(ATTEMPT, 'ATTEMPT')
        logging.addLevelName(COMPLETE, 'COMPLETE')
        logging.addLevelName(IN_PROGRESS, 'IN PROGRESS')

        def attempt(self, message, *args, **kwargs):
            if self.isEnabledFor(ATTEMPT):
                self._log(ATTEMPT, message, args, **kwargs)
        
        def complete(self, message, *args, **kwargs):
            if self.isEnabledFor(COMPLETE):
                self._log(COMPLETE, message, args, **kwargs)

        def in_progress(self, message, *args, **kwargs):
            if self.isEnabledFor(IN_PROGRESS):
                self._log(IN_PROGRESS, message, args, **kwargs)

        logging.Logger.attempt = attempt
        logging.Logger.complete = complete
        logging.Logger.in_progress = in_progress

        console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s %(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    'ATTEMPT': 'yellow',
                    'COMPLETE': 'green',
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'IN PROGRESS': 'white',
                    'WARNING': 'red',
                    'ERROR': 'bold_red',
                    'CRITICAL': 'bold_red'
                }
        )

        # This code sets the format of the messages displayed to the user

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s %(message)s"
        )

        # This code defines a handler, which writes the messages from the Logger

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        # This line creates an info log for the Logger

        file_handler = logging.FileHandler(
            os.path.join(self.db_folder, 'run_info.log')
        )

        file_handler.setFormatter(file_formatter)

        # This defines the Logger

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.logger.setLevel(min(self.logger.getEffectiveLevel(), ATTEMPT))

        # This code connects to the instruments and then defines the voltage source and multimeter

        self.logger.attempt("connecting to station")

        # We use the Station class from qcodes, which represents the physical setup of our experiment

        self.station = qc.Station(config_file=self.station_config_file)
        self.station.load_instrument(self.voltage_source_name)
        # self.station.load_instrument(self.voltage_source_name2)

        import qcodes
        from qcodes import instrument_drivers
        from qcodes.dataset import do0d, load_or_create_experiment
        from qcodes.instrument import Instrument
        from qcodes.instrument_drivers.stanford_research import SR830
        from qcodes.validators import Numbers
        from qcodes import Parameter

        sr = SR830("lockin", "GPIB0::8::INSTR")
        self.station.load_instrument(self.multimeter_name)
        self.voltage_source = getattr(self.station, self.voltage_source_name)
        #self.voltage_source2 = getattr(self.station, self.voltage_source_name2)
        self.drain_mm_device = getattr(self.station, self.multimeter_name)
        
        self.drain_volt = getattr(self.station, self.multimeter_name).volt

        from qcodes.instrument.base import Instrument

        if 'sim900' in Instrument._all_instruments:
            sim900 = Instrument.find_instrument('sim900')
        else:
            sim900 = station.load_instrument('sim900')
        
        self.logger.complete("\n")   
     
            # Assume sr = SR830("lockin", "GPIB0::8::INSTR")
            # Assume self.device_gates is a dict like:
            # {
            #     'sig_x': {'parameter': 'X', 'label': 'Lock-in X', 'unit': 'V'},
            #     'sig_y': {'parameter': 'Y', 'label': 'Lock-in Y', 'unit': 'V'},
            # }

        self.logger.info("Remapping SR830 parameters to match config.yaml")

        for alias, details in self.device_gates.items():
            param_name = details['contact']

            if not hasattr(sr, param_name):
                self.logger.error(f"Parameter {param_name} not found in SR830")
                continue

            # Add DelegateParameter
            sr.add_parameter(
                name=alias,
                parameter_class=qc.parameters.DelegateParameter,
                source=getattr(sr, param_name),
                label=details.get('label', alias),
                unit=details.get('unit', ''),
                step=details.get('step', None)
        )

        self.logger.info(f"Mapped SR830.{param_name} to {alias}")

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
        # self.voltage_source.timeout(10)

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

        # Copy all of the configs for safekeeping
        self.logger.info(f"copying all of the config.yml files")
        shutil.copy(self.station_config_file, self.db_folder)
        shutil.copy(self.tuner_config_file, self.db_folder)
        shutil.copy(self.config_file, self.db_folder)

        # Setup results dictionary
        self.results = {}

        self.results['turn_on'] = {
            'voltage': None,
            'current': None,
            'resistance': None,
            'saturation': None,
        }

        for gate in self.QPC + self.leads + self.accumulation + self.plungers:
            self.results[gate] = {
                'pinch_off': {'voltage': None, 'width': None}
            }
        
        for gate in self.QPC:
            self.results[gate]['bias_voltage'] = None

        # Ground device
        #self.ground_device()

    def ground_device(self):
        
        # Grounds all of the gates in the device
        
        device_gates = self.ohmics + self.all_gates
        self.logger.attempt("grounding device")
        self._smoothly_set_gates_to_voltage(device_gates, 0.)
        self.logger.complete("\n")

    def bias_ohmic(self, 
                   ohmic: str = None, 
                   V: float = 0):
        
        """
        Biases the ohmic to desired voltage. Reports back to the user what the
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
        
        self.logger.attempt(f"setting ohmic ({ohmic}) to {V} V")
        self.logger.info(f'device receives {round(V*self.voltage_divider*1e3,5)} mV based on divider')
        self._smoothly_set_gates_to_voltage(ohmic, V)
        self.ohmic_bias = V*self.voltage_divider
        self.logger.complete("\n")

    def turn_on(self, 
                minV: float = 0.0, 
                maxV: float = None, 
                dV: float = None, 
                delay: float = 0.01) -> pd.DataFrame:
        
        """
        Attempts to 'turn-on' the FET by sweeping barriers and leads
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

        # Default dV and maxV based on setup_config and config
        if dV is None:
            dV = self.voltage_resolution

        if maxV is None:
            maxV = self.voltage_sign * self.abs_max_gate_voltage

        # Ensure we stay in the allowed voltage space 
        assert np.sign(maxV) == self.voltage_sign, self.logger.error("Double check the sign of the gate voltage (maxV) for your given device.")
        assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, self.logger.error("Double check the sign of the gate voltage (minV) for your given device.")

        # Set up gate sweeps
        
        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        gates_involved = self.QPC + self.leads + self.accumulation + self.plungers

        self.logger.info(f"setting {gates_involved} to {minV} V")
        self._smoothly_set_gates_to_voltage(gates_involved, minV)

        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'{gate_name}'), minV, maxV, num_steps, delay, get_after_set=False)
            )

        # Execute the measurement
        self.logger.attempt(f"sweeping {gates_involved} together from {minV} V to {maxV} V")
        result = qc.dataset.dond(
            qc.dataset.TogetherSweep(
                *sweep_list
            ),
            self.drain_volt,
            #break_condition=self._check_break_conditions,
            measurement_name='Device Turn On',
            exp=self.initialization_exp,
            show_progress=True
        )

        self.logger.complete('\n')

        # Get last dataset recorded, convert to units of current (A)
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. gates involved (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gates_involved[0]}', marker= 'o',s=10)
        axes.set_title('Global Device Turn-On')
        df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gates_involved[0]}', ax=axes, linewidth=1)
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*self.global_turn_on_info['abs_min_current'], alpha=0.5, c='g', linestyle=':', label=r'$I_{\min}$')
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*float(self.abs_max_current), alpha=0.5, c='g', linestyle='--', label=r'$I_{\max}$')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$V_{GATES}$ (V)')
        axes.legend(loc='best')

        # Keep any data above the minimum current threshold
        mask = df_current[f'{self.multimeter_name}_current'].abs() > self.global_turn_on_info['abs_min_current'] 
        X = df_current[f'{self.voltage_source_name}_{gates_involved[0]}']
        Y = df_current[f'{self.multimeter_name}_current']
        X_masked = X[mask]
        Y_masked = Y[mask]

        # Try fitting data to tuner config information
        if len(mask) <= 4 or len(X_masked) <= 4:
            self.logger.error("insufficient points above minimum current threshold")
        else:
            try:
                guess = [Y_masked.iloc[-1], self.voltage_sign, X_masked.iloc[0] - self.voltage_sign, 0]

                fit_params, fit_cov = sp.optimize.curve_fit(getattr(self, self.global_turn_on_info['fit_function']), X_masked, Y_masked, guess)
                
                # Extract turn on voltage and saturation voltage
                a, b, x0, y0 = fit_params
                self.logger.info(f"Fit params: a = {a}, b = {b}, x0 = {x0}, y0 = {y0}")

                if self.global_turn_on_info['fit_function'] == 'exponential':
                    V_turn_on =  np.round(np.log(-y0/a)/b + x0, 3)
                elif self.global_turn_on_info['fit_function'] == 'logarithmic':
                    V_turn_on = np.round(np.exp(-y0/a)/b + x0, 3)
                V_sat = df_current[f'{self.voltage_source_name}_{gates_involved[0]}'].iloc[-2] 

                # Plot / print results to user
                axes.plot(X_masked, getattr(self, self.global_turn_on_info['fit_function'])(X_masked, a, b, x0, y0), 'r-')
                axes.axvline(x=V_turn_on, alpha=0.5, linestyle=':',c='b',label=r'$V_{\min}$')
                axes.axvline(x=V_sat,alpha=0.5, linestyle='--',c='b',label=r'$V_{\max}$')
                axes.legend(loc='best')

                

                # Store in device information for later
                self.results['turn_on']['voltage'] = float(V_turn_on)
                self.results['turn_on']['saturation'] = float(V_sat)
                I_measured = np.abs(self._get_drain_current())
                R_measured = round(self.ohmic_bias / I_measured,3)
                self.results['turn_on']['current'] = float(I_measured)
                self.results['turn_on']['resistance'] = float(R_measured)
                self.deviceTurnsOn = True
                self.logger.info(f"device turns on at {V_turn_on} V")
                self.logger.info(f"maximum allowed voltage is {V_sat} V")
                self.logger.info(f"measured current is {I_measured} A")
                self.logger.info(f"calculated resistance is {R_measured} Ohms")

            except RuntimeError:
                self.logger.error(f"fitting to \"{self.global_turn_on_info['fit_function']}\" failed")
                
                self.deviceTurnsOn = self._query_yes_no("Did the device actually turn-on?")
                
                if self.deviceTurnsOn:
                    V_turn_on = input("What was the turn on voltage (V)?")
                    V_sat = input("What was the saturation voltage (V)?")
                    # Store in device dictionary for later
                    self.results['turn_on']['voltage'] = V_turn_on
                    self.results['turn_on']['saturation'] = V_sat

        self._save_figure(plot_info='device_turn_on')


        return df

    def pinch_off(self, 
                    gates: List[str] | str = None, 
                    minV: float = None, 
                    maxV: float = None, 
                    dV: float = None,
                    delay: float = 0.01,
                    voltage_configuration: dict = {}) -> Dict[str, pd.DataFrame]:
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
            df (pd.DataFrame): Return measurement data as a dict of data frames. 

        **Example**

        Pinching off barriers in a SET (Single Electron Transistor) QD device,
        >>> QD_FET_Tuner.pinch_off(gates=['LB','RB'], minV=1, maxV=None, dV=0.005)
        """

        assert self.deviceTurnsOn, "Device does not turn on. Why are you pinching anything off?"

        # Bring device to voltage configuration
        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self._smoothly_set_voltage_configuration(voltage_configuration)

        if dV is None:
            dV = self.voltage_resolution

        if gates is None:
            gates = self.QPC + self.leads + self.accumulation + self.plungers
        if type(gates) is str:
            gates = [gates]

        if maxV is None:
            maxV = self.results['turn_on']['saturation']
        else:
            assert np.sign(maxV) == self.voltage_sign, self.logger.error("Double check the sign of the gate voltage (maxV) for your given device.")
        
        if minV is None:
            if self.voltage_sign == 1:
                min_allowed = 0
                max_allowed = None
            elif self.voltage_sign == -1:
                min_allowed = None
                max_allowed = 0
            minV = np.clip(
                    a=round(self.results['turn_on']['saturation'] - self.voltage_sign * self.abs_max_gate_differential, 3),
                    a_min=min_allowed,
                    a_max=max_allowed
                )
        else:
            assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, self.logger.error("Double check the sign of the gate voltage (minV) for your given device.")

        self.logger.info(f"setting {gates} to {maxV} V")
        self._smoothly_set_gates_to_voltage(gates, maxV)

        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        sweep_list = []
        for gate_name in gates:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'{gate_name}'), maxV, minV, num_steps, delay, get_after_set=False)
            )

        def adjusted_break_condition():
            return np.abs(self._get_drain_current()) < self.pinch_off_info['abs_min_current']

        results_dict = {}
        for sweep in sweep_list:

            self.logger.attempt(f"pinching off {str(sweep._param).split('_')[-1]} from {maxV} V to {minV} V")
            result = qc.dataset.dond(
                sweep,
                self.drain_volt,
                # break_condition=adjusted_break_condition,
                measurement_name='{} Pinch Off'.format(str(sweep._param).split('_')[-1]),
                exp=self.initialization_exp,
                show_progress=True
            )   
            self.logger.complete("\n")

            self.logger.info(f"returning {str(sweep._param).split('_')[-1]} to {maxV} V")
            self._smoothly_set_gates_to_voltage([str(sweep._param).split('_')[-1]], maxV)

            # Get last dataset recorded, convert to current units
            dataset = qc.load_last_experiment().last_data_set()
            df = dataset.to_pandas_dataframe().reset_index()

            results_dict[str(sweep._param).split('_')[-1]] = df

            df_current = df.copy()
            df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
            df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

            # Plot current v.s. param being swept
            axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x=f'{str(sweep._param)}', marker= 'o',s=10)
            df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'{str(sweep._param)}',  ax=axes, linewidth=1)
            axes.axhline(y=0,  alpha=.5, linewidth=0.5, c='k', linestyle='-')
            axes.axvline(x=0, alpha=.5, linewidth=0.5, c='k', linestyle='-')
            axes.set_ylabel(r'$I$ (A)')
            axes.set_xlabel(r'$V_{{{gate}}}$ (V)'.format(gate=str(sweep._param).split('_')[-1]))
            axes.set_title('{} Pinch Off'.format(str(sweep._param).split('_')[-1]))
            axes.set_xlim(0, float(self.results['turn_on']['saturation']))
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[0])*self.pinch_off_info['abs_min_current'], alpha=0.5,c='g', linestyle=':', label=r'$I_{\min}$')
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[0])*float(self.abs_max_current), alpha=0.5,c='g', linestyle='--', label=r'$I_{\max}$')
                
            mask_belowthreshold = df_current[f'{self.multimeter_name}_current'].abs() < self.pinch_off_info['abs_min_current'] 

            # Keep any data that is above the minimum turn-on current

            if str(sweep._param).split('_')[-1] in self.leads:
                mask = df_current[f'{self.multimeter_name}_current'].abs() > self.pinch_off_info['abs_min_current'] 
                X = df_current[f'{str(sweep._param)}']
                Y = df_current[f'{self.multimeter_name}_current']
                X_masked = X[mask]
                Y_masked = Y[mask]
                X_belowthreshold = X[mask_belowthreshold]

            else:
                X = df_current[f'{str(sweep._param)}']
                Y = df_current[f'{self.multimeter_name}_current']
                X_masked = X
                Y_masked = Y
                X_belowthreshold = X[mask_belowthreshold]

            X_masked = pd.to_numeric(X_masked, errors='coerce')
            Y_masked = pd.to_numeric(Y_masked, errors='coerce')

            X_masked = X_masked.dropna()
            Y_masked = Y_masked.dropna()

            # Do the fit if possible
            if len(X_masked) <= 4 or len(X_belowthreshold) == 0:
                self.logger.error(f"{str(sweep._param).split('_')[-1]}, insufficient points below minimum current threshold to do any fitting")
                self.logger.attempt(f"{str(sweep._param).split('_')[-1]}, fitting data to linear fit to see if barrier is broken.")

                try:
                    # Guess a line fit with zero slope.
                    guess = (0,np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*self.abs_max_current)
                    fit_params, fit_cov = sp.optimize.curve_fit(getattr(self, 'linear'), X, Y, guess)
                    m, b = fit_params
                    plt.plot(X, self.linear(X,m,b), 'r-')
                    self.logger.warning(f"{str(sweep._param).split('_')[-1]}, data fits well to line, most likely barrier is shorted somewhere.")
                    continue 

                except RuntimeError:
                    self.logger.error(f"{str(sweep._param).split('_')[-1]}, linear fit failed.")

            else:
                # Try and fit and get fit params
                try: 
                    if str(sweep._param).split('_')[-1] in self.leads:

                        fit_function = getattr(self, 'logarithmic')
                        guess = [Y_masked.iloc[0], self.voltage_sign, X_masked.iloc[0] - self.voltage_sign, 0]
                    
                        self.logger.info(f"{str(sweep._param).split('_')[-1]}, fitting data to logarithmic function")
                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params
                        self.logger.info(f"Fit params: a = {a}, b = {b}, x0 = {x0}, y0 = {y0}")

                        V_pinchoff = float(round(np.exp(-y0/a)/b + x0,3))
                        self.results[str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                   
                    else:

                        fit_function = getattr(self, self.pinch_off_info['fit_function'])
                        guess = (Y.iloc[0], -1 * self.voltage_sign * 100, float(self.results['turn_on']['voltage']), 0)

                        self.logger.info(f"{str(sweep._param).split('_')[-1]}, fitting data to {self.pinch_off_info['fit_function']}")
                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params
                        self.logger.info(f"Fit params: a = {a}, b = {b}, x0 = {x0}, y0 = {y0}")

                        V_pinchoff = self.voltage_sign * float(round(min(
                            np.abs(x0 - np.sqrt(8) / b),
                            np.abs(x0 + np.sqrt(8) / b)
                        ),3))
                        V_pinchoff_width = float(abs(round(2 * np.sqrt(8) / b,2)))
                        self.results[str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                        self.results[str(sweep._param).split('_')[-1]]['pinch_off']['width'] = V_pinchoff_width #V

                    plt.plot(X_masked, fit_function(X_masked, a, b, x0, y0), 'r-')
                    axes.axvline(x=V_pinchoff, alpha=0.5, linestyle=':', c='b', label=r'$V_{\min}$')
                    self.logger.info(f"{str(sweep._param).split('_')[-1]}, pinch off at {V_pinchoff}")
                    if str(sweep._param).split('_')[-1] not in self.leads:
                        axes.axvline(x=V_pinchoff + self.voltage_sign * V_pinchoff_width, alpha=0.5, linestyle='--', c='b', label=r'$V_{\max}$')
                        self.logger.info(f"{str(sweep._param).split('_')[-1]}, pinch off width of {V_pinchoff_width}")
                    axes.legend(loc='best')                
                
                except RuntimeError:
                    self.logger.error(f"{str(sweep._param).split('_')[-1]}, insufficient points below minimum current threshold to do any fitting")
                    self.logger.attempt(f"{str(sweep._param).split('_')[-1]}, fitting to linear fit to see if barrier is broken.")

                    try:
                        # Guess a line fit with zero slope.
                        guess = (0,np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*self.abs_max_current)
                        fit_params, fit_cov = sp.optimize.curve_fit(getattr(self, 'linear'), X, Y, guess)
                        m, b = fit_params
                        plt.plot(X, self.linear(X,m,b), 'r-')
                        self.logger.warning(f"{str(sweep._param).split('_')[-1]}, data fits well to line, most likely barrier is shorted somewhere.")

                    except RuntimeError:
                        self.logger.error(f"{str(sweep._param).split('_')[-1]}, linear fit failed.")         

            config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
            self._save_figure(plot_info=f"{str(sweep._param).split('_')[-1]}_pinch_{config_str}")


        return results_dict

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
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self._smoothly_set_voltage_configuration(voltage_configuration)
        else:
            self.logger.info(f"setting {self.leads} to {self.results['turn_on']['saturation']} V")
            self._smoothly_set_gates_to_voltage(self.leads, self.results['turn_on']['saturation'])

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
            self.logger.error("invalid dV")
            return

        # Double check barrier validity
        if B1 is None or B2 is None and len(self.QPC) == 2:
            B1, B2 = self.QPC
            self.logger.warning(f"no barriers provided, assuming {B1} and {B2} from device configuration")
        
        # Double check device bounds
        minV_B1, maxV_B1 = B1_bounds
        minV_B2, maxV_B2 = B2_bounds

        if minV_B1 is None:
            minV_B1 = self.results[B1]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B1) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (minV) for B1.")

        if minV_B2 is None:
            minV_B2 = self.results[B2]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B2) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (minV) for B2.")

        if maxV_B1 is None:
            if self.voltage_sign == 1:
                maxV_B1 = min(self.results[B1]['pinch_off']['voltage']+self.voltage_sign*self.results[B1]['pinch_off']['width'], self.results['turn_on']['saturation'])
            elif self.voltage_sign == -1:
                maxV_B1 = max(self.results[B1]['pinch_off']['voltage']+self.voltage_sign*self.results[B1]['pinch_off']['width'], self.results['turn_on']['saturation'])
        else:
            assert np.sign(maxV_B1) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (maxV) for B1.")

        if maxV_B2 is None:
            if self.voltage_sign == 1:
                maxV_B2 = min(self.results[B2]['pinch_off']['voltage']+self.voltage_sign*self.results[B2]['pinch_off']['width'], self.results['turn_on']['saturation'])
            elif self.voltage_sign == -1:
                maxV_B2 = max(self.results[B2]['pinch_off']['voltage']+self.voltage_sign*self.results[B2]['pinch_off']['width'], self.results['turn_on']['saturation'])
        else:
            assert np.sign(maxV_B2) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (maxV) for B2.")

        self.logger.info(f"setting {B1} to {maxV_B1} V")
        self.logger.info(f"setting {B2} to {maxV_B2} V")
        self._smoothly_set_voltage_configuration({B1: maxV_B1, B2: maxV_B2})

        def smooth_reset():
            """Resets the inner loop variable smoothly back to the starting value
            """
            self._smoothly_set_gates_to_voltage([B2], maxV_B2)

        num_steps_B1 = self._calculate_num_of_steps(minV_B1, maxV_B1, dV_B1)
        num_steps_B2 = self._calculate_num_of_steps(minV_B2, maxV_B2, dV_B2)
        self.logger.attempt("barrier barrier scan")
        self.logger.info(f"stepping {B1} from {maxV_B1} V to {minV_B1} V")
        self.logger.info(f"sweeping {B2} from {maxV_B2} V to {minV_B2} V")
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
        self.logger.complete("\n")

        self.logger.info(f"returning gates {B1}, {B2} to {maxV_B1} V, {maxV_B2} V respectively")
        self._smoothly_set_gates_to_voltage([B1], maxV_B1)
        self._smoothly_set_gates_to_voltage([B2], maxV_B2)

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_name}_current', index=[f'{self.voltage_source_name}_{B1}'], columns=[f'{self.voltage_source_name}_{B2}'])
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

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{B1}_{B2}_sweep_{config_str}')

        if extract_bias_point:

            # Inference data
            bias_point, voltage_window = self.extract_bias_point(
                df,
                plot_process=True,
                axes=ax1
            )

            for barrier_gate_name, voltage in bias_point.items():
                self.results[barrier_gate_name]['bias_voltage'] = voltage



        return (df,ax1)

    def coulomb_blockade(self, 
                         gate: str = None, 
                         gate_bounds: tuple = (None, None), 
                         dV: float = None, 
                         delay: float = 0.01,
                         voltage_configuration: dict = None) -> tuple[pd.DataFrame, plt.Axes]:
        """Attempts to sweep plunger gate to see coulomb blockade features. Ideally
        the voltage configuration provided should be that which creates a well defined
        central dot between the two barriers.

        Args:
            gate (str, optional): Plunger gate. Defaults to None.
            gate_bounds (tuple, optional): Voltage bounds (V). Defaults to (None, None).
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
        assert voltage_configuration is not None, self.logger.error("no voltage configuration provided")

        self.logger.info(f"setting voltage configuration: {voltage_configuration}")
        self._smoothly_set_voltage_configuration(voltage_configuration)

        if gate is None:
            gate = self.plungers[0]

        if dV is None:
            dV = self.voltage_resolution

        minV_gate, maxV_gate = gate_bounds
        if minV_gate is None:
            minV_gate = 0
        assert maxV_gate is not None, self.logger.error(f"{gate}, maximum voltage not provided")
  
        num_steps_P = self._calculate_num_of_steps(minV_gate, maxV_gate, dV)
        gate_sweep = LinSweep_SIM928(getattr(self.voltage_source, f'{gate}'), minV_gate, maxV_gate, num_steps_P, delay, get_after_set=False)

        self.logger.info(f"setting {gate} to {minV_gate} V")
        self._smoothly_set_gates_to_voltage([gate], minV_gate)

        self.logger.attempt(f"sweeping gate {gate} from {minV_gate} V to {maxV_gate} V")
        # Execute the measurement
        result = qc.dataset.dond(
            gate_sweep,
            self.drain_volt,
            write_period=0.1,
            # break_condition=self._check_break_conditions,
            measurement_name='Coulomb Blockade',
            exp=self.initialization_exp,
            show_progress=True
        )
        self.logger.complete("\n")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gate}', marker= 'o',s=10)
        df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gate}', ax=axes, linewidth=1)
        axes.set_title('Coulomb Blockade')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))
        axes.legend(loc='best')

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{gate}_sweep_{config_str}')

        return (df, axes,)

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

        assert voltage_configuration is not None, self.logger.error("no voltage configuration provided")

        self.logger.info(f"setting voltage configuration: {voltage_configuration}")
        self._smoothly_set_voltage_configuration(voltage_configuration)

        if dV_ohmic is None:
            dV_ohmic = self.voltage_resolution
        
        if dV_gate is None:
            dV_gate = self.voltage_resolution
        
        minV_ohmic, maxV_ohmic = ohmic_bounds
        minV_gate, maxV_gate = gate_bounds

        self.logger.info(f"setting {ohmic} to {maxV_ohmic} V")
        self.logger.info(f"setting {gate} to {maxV_gate} V")
        self._smoothly_set_voltage_configuration({ohmic: maxV_ohmic, gate: maxV_gate})

        assert minV_gate is not None, self.logger.error(f"no minimum voltage for {gate} provided")
        assert maxV_gate is not None, self.logger.error(f"no maximum voltage for {gate} provided")
        assert minV_ohmic is not None, self.logger.error(f"no minimum voltage for {ohmic} provided")
        assert maxV_ohmic is not None, self.logger.error(f"no maximum voltage for {ohmic} provided")
        
        def smooth_reset():
            self._smoothly_set_gates_to_voltage([ohmic], maxV_ohmic)

        self.logger.info(f"stepping {gate} from {maxV_gate} V to {minV_gate} V")
        self.logger.info(f"sweeping {ohmic} from {maxV_ohmic} V to {minV_ohmic} V")
        num_steps_ohmic = self._calculate_num_of_steps(minV_ohmic, maxV_ohmic, dV_ohmic)
        num_steps_gate = self._calculate_num_of_steps(minV_gate, maxV_gate, dV_gate)

        self.logger.attempt("Coulomb diamond scan")
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
        self.logger.complete("\n")

        self.logger.info(f"setting gates {gate}, {ohmic} to {maxV_gate} V, {maxV_ohmic} V respectively")
        self._smoothly_set_voltage_configuration({ohmic: maxV_ohmic, gate: maxV_gate})

  
        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity
        df_current.iloc[:,-2] = df_current.iloc[:,-2].mul(self.voltage_divider * 1e3) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_name}_current', index=[f'{self.voltage_source_name}_{ohmic}'], columns=[f'{self.voltage_source_name}_{gate}'])
        gate_data, ohmic_data = df_pivoted.columns, df_pivoted.index
        raw_current_data = df_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        gate_grad = np.gradient(raw_current_data, axis=1)
        ohmic_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Coulomb Diamonds")

        im_aspect = 'auto'

        cbar_ax1 = plt.colorbar(ax1.imshow(
            raw_current_data,
            extent=[gate_data[0], gate_data[-1], ohmic_data[0], ohmic_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_aspect
        ), ax=ax1,fraction=0.046, pad=0.04)

        cbar_ax1.set_label(r'$I_{SD}$ (nA)')
        ax1.set_title(r'$I_{SD}$')
        ax1.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))
        ax1.set_ylabel(r"$V_{{{gate_name}}}$ (mV)".format(gate_name=ohmic))

        # V grad is actually horizontal
        grad_vector = (1,0) 

        cbar_ax2 = plt.colorbar(ax2.imshow(
            np.sqrt(grad_vector[0] * gate_grad**2 +   grad_vector[1]* ohmic_grad**2),
            extent=[gate_data[0], gate_data[-1], ohmic_data[0], ohmic_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_aspect
        ), ax=ax2,fraction=0.046, pad=0.04)

        cbar_ax2.set_label(r'$\nabla I_{SD}$ (nA/V)')
        ax2.set_title(r'$\nabla I_{SD}$')
        ax2.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))
        ax2.set_ylabel(r"$V_{{{gate_name}}}$ (mV)".format(gate_name=ohmic))

        fig.tight_layout()

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{ohmic}_{gate}_sweep_{config_str}')
        
        return (df, ax1)

    def current_trace(self, 
                      f_sampling: int,
                      t_capture: int, 
                      voltage_configuration: dict = None,
                      meas_name: str = "Current Trace",
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

        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self._smoothly_set_voltage_configuration(voltage_configuration)

        time_param = qc.parameters.ElapsedTimeParameter('time')
        meas = qc.dataset.Measurement(exp=self.initialization_exp, name=meas_name)
        meas.register_parameter(time_param)
        meas.register_parameter(self.drain_mm_device.volt, setpoints=[time_param])

        self.logger.info(f"Running current trace for {t_capture} seconds")
        with meas.run() as datasaver:
            time_param.reset_clock()
            elapsed_time = 0
            while elapsed_time < t_capture:
                datasaver.add_result((self.drain_mm_device.volt, self.drain_mm_device.volt()),
                                (time_param, time_param()))
                elapsed_time = time_param.get()
      
        df = datasaver.dataset.to_pandas_dataframe().reset_index()
        df_current = df.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)

        axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x='time', marker= 'o',s=5)
        df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'time', ax=axes, linewidth=1)
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$t$ (s)')
        axes.set_title(rf'Current noise, $f_s={f_sampling}$ Hz, $t_\max={t_capture}$ s')
        plt.show()
        
        if plot_psd:

            # Plot noise spectrum
            
            t = df_current[f'time']
            I = df_current[f'{self.multimeter_name}_current']
            f, Pxx = sp.signal.periodogram(I, fs=f_sampling, scaling='density')

            plt.loglog(f, Pxx)
            plt.xlabel(r'$\omega$ (Hz)')
            plt.ylabel(r'$S_I$ (A$^2$/Hz)')
            plt.title(r"Current noise spectrum")
            plt.show()

        return df

    def _load_config_files(self):
        
        # Read the tuner config information

        self.tuner_info = yaml.safe_load(Path(self.tuner_config_file).read_text())
        self.global_turn_on_info = self.tuner_info['global_turn_on']
        self.pinch_off_info = self.tuner_info['pinch_off']

        # Read the config information

        self.config = yaml.safe_load(Path(self.config_file).read_text())
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
        self.QPC = []
        self.leads = []
        self.plungers = []
        self.accumulation = []
        self.screening = []
        self.source_ohmic = []
        
        for gate, details in self.device_gates.items():
            
            if details['type'] == 'ohmic':
                self.ohmics.append(gate)
            
            if details['type'] == 'barrier':
                self.QPC.append(gate)
            
            if details['type'] == 'lead':
                self.leads.append(gate)
            
            if details['type'] == 'plunger':
                self.plungers.append(gate)
            
            if details['type'] == 'accumulation':
                self.accumulation.append(gate)

            if details['type'] == 'screening':
                self.screening.append(gate)

            if details['type'] == 'source_ohmic':
                self.source_ohmic.append(gate)
        
        self.all_gates = list(self.device_gates.keys())



        self.abs_max_current = self.config['device']['constraints']['abs_max_current']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.config['device']['constraints']['abs_max_gate_differential']

        self.voltage_source_name = self.config['setup']['voltage_source']
        self.voltage_source_name2 = self.config['setup']['voltage_source2']
        self.multimeter_name = self.config['setup']['multimeter']
        self.voltage_divider = self.config['setup']['voltage_divider']
        self.preamp_bias = self.config['setup']['preamp_bias']
        self.preamp_sensitivity = self.config['setup']['preamp_sensitivity']
        self.voltage_resolution = self.config['setup']['voltage_resolution']

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

        # First, we determine which gates to are being swept and ensure the correct sign is maintained

        gates_to_check = self.leads + self.QPC + self.screening + self.accumulation + self.plungers + self.source_ohmic

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
                                          deltav[gate_name]*(maxsteps-s-1)/maxsteps
        else:
            
            # First, we set up the values of the sweeps and save them

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

        
        

        # Now, we loop over intermediate to set the voltages

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

    def _save_figure(self, plot_info: str, plot_dpi: int = 300, plot_format: str = 'svg'):
        """Saves plot generated from one of the stages.

        Args:
            plot_info (str): Plot information for the filename.
            plot_dpi (int, optional): Dots Per Inch (DPI) for the plot. Defaults to 300.
            plot_format (str, optional): Picture format file type. Defaults to 'svg'.
        """

        completition_time = str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) # the current minute
        plot_name = os.path.join(self.db_folder, f"{plot_info}_{completition_time}.{plot_format}")
        plt.savefig(fname=plot_name, dpi=plot_dpi, format=plot_format)

class QuantumDotFET(DataAnalysis, DataAcquisition):
  
    """Dedicated class to tune simple FET devices.
    """

    def __init__(self, 
                 config: str, 
                 tuner_config: str,
                 station_config: str,
                 save_dir: str) -> None:
        
        """
        Initializes the tuner.

        Args:
            config (str): Path to .yaml file containing device information.
            setup_config (str): Path to .yaml file containing experimental setup information.
            tuner_config (str): Path to .yaml file containing tuner information.
            station_config (str): Path to .yaml file containing QCoDeS station information
            save_dir (str): Directory to save data and plots generated.
        """

        DataAnalysis.__init__(self, tuner_config=tuner_config)
        DataAcquisition.__init__(self)
    
        # First, we save the config file data and the data directory, as well as loading the config files

        self.config_file = config
        self.tuner_config_file = tuner_config
        self.station_config_file = station_config
        self.save_dir = save_dir

        self._load_config_files()
        
        # Then, we set up the date and file where data is stored

        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        self.db_folder = os.path.join(save_dir, f"{self.config['device']['characteristics']['name']}_{todays_date}")
        os.makedirs(self.db_folder, exist_ok=True)

        # This code creates new logging categories for the user to see while the code runs

        ATTEMPT, COMPLETE, IN_PROGRESS = logging.INFO - 2, logging.INFO - 1, logging.INFO

        logging.addLevelName(ATTEMPT, 'ATTEMPT')
        logging.addLevelName(COMPLETE, 'COMPLETE')
        logging.addLevelName(IN_PROGRESS, 'IN PROGRESS')

        def attempt(self, message, *args, **kwargs):
            if self.isEnabledFor(ATTEMPT):
                self._log(ATTEMPT, message, args, **kwargs)
        
        def complete(self, message, *args, **kwargs):
            if self.isEnabledFor(COMPLETE):
                self._log(COMPLETE, message, args, **kwargs)

        def in_progress(self, message, *args, **kwargs):
            if self.isEnabledFor(IN_PROGRESS):
                self._log(IN_PROGRESS, message, args, **kwargs)

        logging.Logger.attempt = attempt
        logging.Logger.complete = complete
        logging.Logger.in_progress = in_progress

        console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s %(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    'ATTEMPT': 'yellow',
                    'COMPLETE': 'green',
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'IN PROGRESS': 'white',
                    'WARNING': 'red',
                    'ERROR': 'bold_red',
                    'CRITICAL': 'bold_red'
                }
        )

        # This code sets the format of the messages displayed to the user

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s %(message)s"
        )

        # This code defines a handler, which writes the messages from the Logger

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        # This line creates an info log for the Logger

        file_handler = logging.FileHandler(
            os.path.join(self.db_folder, 'run_info.log')
        )

        file_handler.setFormatter(file_formatter)

        # This defines the Logger

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.logger.setLevel(min(self.logger.getEffectiveLevel(), ATTEMPT))

        # This code connects to the instruments and then defines the voltage source and multimeter

        self.logger.attempt("connecting to station")

        # We use the Station class from qcodes, which represents the physical setup of our experiment

        self.station = qc.Station(config_file=self.station_config_file)
        self.station.load_instrument(self.voltage_source_name)
        self.station.load_instrument(self.multimeter_name)
        self.voltage_source = getattr(self.station, self.voltage_source_name)
        self.drain_mm_device = getattr(self.station, self.multimeter_name)
        
        self.drain_volt = getattr(self.station, self.multimeter_name).volt
        
        self.logger.complete("\n")

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
        # self.voltage_source.timeout(10)

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

        # Copy all of the configs for safekeeping
        self.logger.info(f"copying all of the config.yml files")
        shutil.copy(self.station_config_file, self.db_folder)
        shutil.copy(self.tuner_config_file, self.db_folder)
        shutil.copy(self.config_file, self.db_folder)

        # Setup results dictionary
        self.results = {}

        self.results['turn_on'] = {
            'voltage': None,
            'current': None,
            'resistance': None,
            'saturation': None,
        }

        for gate in self.barriers + self.leads + self.accumulation + self.plungers:
            self.results[gate] = {
                'pinch_off': {'voltage': None, 'width': None}
            }
        
        for gate in self.barriers:
            self.results[gate]['bias_voltage'] = None

        # Ground device
        #self.ground_device()

    def ground_device(self):
        
        # Grounds all of the gates in the device
        
        device_gates = self.ohmics + self.all_gates
        self.logger.attempt("grounding device")
        self._smoothly_set_gates_to_voltage(device_gates, 0.)
        self.logger.complete("\n")

    def bias_ohmic(self, 
                   ohmic: str = None, 
                   V: float = 0):
        
        """
        Biases the ohmic to desired voltage. Reports back to the user what the
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
        
        self.logger.attempt(f"setting ohmic ({ohmic}) to {V} V")
        self.logger.info(f'device receives {round(V*self.voltage_divider*1e3,5)} mV based on divider')
        self._smoothly_set_gates_to_voltage(ohmic, V)
        self.ohmic_bias = V*self.voltage_divider
        self.logger.complete("\n")

    def turn_on(self, 
                minV: float = 0.0, 
                maxV: float = None, 
                dV: float = None, 
                delay: float = 0.01) -> pd.DataFrame:
        
        """
        Attempts to 'turn-on' the FET by sweeping barriers and leads
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

        # Default dV and maxV based on setup_config and config
        if dV is None:
            dV = self.voltage_resolution

        if maxV is None:
            maxV = self.voltage_sign * self.abs_max_gate_voltage

        # Ensure we stay in the allowed voltage space 
        assert np.sign(maxV) == self.voltage_sign, self.logger.error("Double check the sign of the gate voltage (maxV) for your given device.")
        assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, self.logger.error("Double check the sign of the gate voltage (minV) for your given device.")

        # Set up gate sweeps
        
        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        gates_involved = self.barriers + self.leads + self.accumulation + self.plungers

        self.logger.info(f"setting {gates_involved} to {minV} V")
        self._smoothly_set_gates_to_voltage(gates_involved, minV)

        sweep_list = []
        for gate_name in gates_involved:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'{gate_name}'), minV, maxV, num_steps, delay, get_after_set=False)
            )

        # Execute the measurement
        self.logger.attempt(f"sweeping {gates_involved} together from {minV} V to {maxV} V")
        result = qc.dataset.dond(
            qc.dataset.TogetherSweep(
                *sweep_list
            ),
            self.drain_volt,
            #break_condition=self._check_break_conditions,
            measurement_name='Device Turn On',
            exp=self.initialization_exp,
            show_progress=True
        )

        self.logger.complete('\n')

        # Get last dataset recorded, convert to units of current (A)
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. gates involved (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gates_involved[0]}', marker= 'o',s=10)
        axes.set_title('Global Device Turn-On')
        df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gates_involved[0]}', ax=axes, linewidth=1)
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*self.global_turn_on_info['abs_min_current'], alpha=0.5, c='g', linestyle=':', label=r'$I_{\min}$')
        axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*float(self.abs_max_current), alpha=0.5, c='g', linestyle='--', label=r'$I_{\max}$')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$V_{GATES}$ (V)')
        axes.legend(loc='best')

        # Keep any data above the minimum current threshold
        mask = df_current[f'{self.multimeter_name}_current'].abs() > self.global_turn_on_info['abs_min_current'] 
        X = df_current[f'{self.voltage_source_name}_{gates_involved[0]}']
        Y = df_current[f'{self.multimeter_name}_current']
        X_masked = X[mask]
        Y_masked = Y[mask]

        # Try fitting data to tuner config information
        if len(mask) <= 4 or len(X_masked) <= 4:
            self.logger.error("insufficient points above minimum current threshold")
        else:
            try:
                guess = [Y_masked.iloc[-1], self.voltage_sign, X_masked.iloc[0] - self.voltage_sign, 0]

                fit_params, fit_cov = sp.optimize.curve_fit(getattr(self, self.global_turn_on_info['fit_function']), X_masked, Y_masked, guess)
                
                # Extract turn on voltage and saturation voltage
                a, b, x0, y0 = fit_params
                self.logger.info(f"Fit params: a = {a}, b = {b}, x0 = {x0}, y0 = {y0}")

                if self.global_turn_on_info['fit_function'] == 'exponential':
                    V_turn_on =  np.round(np.log(-y0/a)/b + x0, 3)
                elif self.global_turn_on_info['fit_function'] == 'logarithmic':
                    V_turn_on = np.round(np.exp(-y0/a)/b + x0, 3)
                V_sat = df_current[f'{self.voltage_source_name}_{gates_involved[0]}'].iloc[-2] 

                # Plot / print results to user
                axes.plot(X_masked, getattr(self, self.global_turn_on_info['fit_function'])(X_masked, a, b, x0, y0), 'r-')
                axes.axvline(x=V_turn_on, alpha=0.5, linestyle=':',c='b',label=r'$V_{\min}$')
                axes.axvline(x=V_sat,alpha=0.5, linestyle='--',c='b',label=r'$V_{\max}$')
                axes.legend(loc='best')

                

                # Store in device information for later
                self.results['turn_on']['voltage'] = float(V_turn_on)
                self.results['turn_on']['saturation'] = float(V_sat)
                I_measured = np.abs(self._get_drain_current())
                R_measured = round(self.ohmic_bias / I_measured,3)
                self.results['turn_on']['current'] = float(I_measured)
                self.results['turn_on']['resistance'] = float(R_measured)
                self.deviceTurnsOn = True
                self.logger.info(f"device turns on at {V_turn_on} V")
                self.logger.info(f"maximum allowed voltage is {V_sat} V")
                self.logger.info(f"measured current is {I_measured} A")
                self.logger.info(f"calculated resistance is {R_measured} Ohms")

            except RuntimeError:
                self.logger.error(f"fitting to \"{self.global_turn_on_info['fit_function']}\" failed")
                
                self.deviceTurnsOn = self._query_yes_no("Did the device actually turn-on?")
                
                if self.deviceTurnsOn:
                    V_turn_on = input("What was the turn on voltage (V)?")
                    V_sat = input("What was the saturation voltage (V)?")
                    # Store in device dictionary for later
                    self.results['turn_on']['voltage'] = V_turn_on
                    self.results['turn_on']['saturation'] = V_sat

        self._save_figure(plot_info='device_turn_on')


        return df

    def pinch_off(self, 
                    gates: List[str] | str = None, 
                    minV: float = None, 
                    maxV: float = None, 
                    dV: float = None,
                    delay: float = 0.01,
                    voltage_configuration: dict = {}) -> Dict[str, pd.DataFrame]:
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
            df (pd.DataFrame): Return measurement data as a dict of data frames. 

        **Example**

        Pinching off barriers in a SET (Single Electron Transistor) QD device,
        >>> QD_FET_Tuner.pinch_off(gates=['LB','RB'], minV=1, maxV=None, dV=0.005)
        """

        assert self.deviceTurnsOn, "Device does not turn on. Why are you pinching anything off?"

        # Bring device to voltage configuration
        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self._smoothly_set_voltage_configuration(voltage_configuration)

        if dV is None:
            dV = self.voltage_resolution

        if gates is None:
            gates = self.barriers + self.leads + self.accumulation + self.plungers
        if type(gates) is str:
            gates = [gates]

        if maxV is None:
            maxV = self.results['turn_on']['saturation']
        else:
            assert np.sign(maxV) == self.voltage_sign, self.logger.error("Double check the sign of the gate voltage (maxV) for your given device.")
        
        if minV is None:
            if self.voltage_sign == 1:
                min_allowed = 0
                max_allowed = None
            elif self.voltage_sign == -1:
                min_allowed = None
                max_allowed = 0
            minV = np.clip(
                    a=round(self.results['turn_on']['saturation'] - self.voltage_sign * self.abs_max_gate_differential, 3),
                    a_min=min_allowed,
                    a_max=max_allowed
                )
        else:
            assert np.sign(minV) == self.voltage_sign or np.sign(minV) == 0, self.logger.error("Double check the sign of the gate voltage (minV) for your given device.")

        self.logger.info(f"setting {gates} to {maxV} V")
        self._smoothly_set_gates_to_voltage(gates, maxV)

        num_steps = self._calculate_num_of_steps(minV, maxV, dV)
        sweep_list = []
        for gate_name in gates:
            sweep_list.append(
                LinSweep_SIM928(getattr(self.voltage_source, f'{gate_name}'), maxV, minV, num_steps, delay, get_after_set=False)
            )

        def adjusted_break_condition():
            return np.abs(self._get_drain_current()) < self.pinch_off_info['abs_min_current']

        results_dict = {}
        for sweep in sweep_list:

            self.logger.attempt(f"pinching off {str(sweep._param).split('_')[-1]} from {maxV} V to {minV} V")
            result = qc.dataset.dond(
                sweep,
                self.drain_volt,
                # break_condition=adjusted_break_condition,
                measurement_name='{} Pinch Off'.format(str(sweep._param).split('_')[-1]),
                exp=self.initialization_exp,
                show_progress=True
            )   
            self.logger.complete("\n")

            self.logger.info(f"returning {str(sweep._param).split('_')[-1]} to {maxV} V")
            self._smoothly_set_gates_to_voltage([str(sweep._param).split('_')[-1]], maxV)

            # Get last dataset recorded, convert to current units
            dataset = qc.load_last_experiment().last_data_set()
            df = dataset.to_pandas_dataframe().reset_index()

            results_dict[str(sweep._param).split('_')[-1]] = df

            df_current = df.copy()
            df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
            df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

            # Plot current v.s. param being swept
            axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x=f'{str(sweep._param)}', marker= 'o',s=10)
            df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'{str(sweep._param)}',  ax=axes, linewidth=1)
            axes.axhline(y=0,  alpha=.5, linewidth=0.5, c='k', linestyle='-')
            axes.axvline(x=0, alpha=.5, linewidth=0.5, c='k', linestyle='-')
            axes.set_ylabel(r'$I$ (A)')
            axes.set_xlabel(r'$V_{{{gate}}}$ (V)'.format(gate=str(sweep._param).split('_')[-1]))
            axes.set_title('{} Pinch Off'.format(str(sweep._param).split('_')[-1]))
            axes.set_xlim(0, float(self.results['turn_on']['saturation']))
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[0])*self.pinch_off_info['abs_min_current'], alpha=0.5,c='g', linestyle=':', label=r'$I_{\min}$')
            axes.axhline(y=np.sign(df_current[f'{self.multimeter_name}_current'].iloc[0])*float(self.abs_max_current), alpha=0.5,c='g', linestyle='--', label=r'$I_{\max}$')
                
            mask_belowthreshold = df_current[f'{self.multimeter_name}_current'].abs() < self.pinch_off_info['abs_min_current'] 

            # Keep any data that is above the minimum turn-on current

            if str(sweep._param).split('_')[-1] in self.leads:
                mask = df_current[f'{self.multimeter_name}_current'].abs() > self.pinch_off_info['abs_min_current'] 
                X = df_current[f'{str(sweep._param)}']
                Y = df_current[f'{self.multimeter_name}_current']
                X_masked = X[mask]
                Y_masked = Y[mask]
                X_belowthreshold = X[mask_belowthreshold]

            else:
                X = df_current[f'{str(sweep._param)}']
                Y = df_current[f'{self.multimeter_name}_current']
                X_masked = X
                Y_masked = Y
                X_belowthreshold = X[mask_belowthreshold]

            X_masked = pd.to_numeric(X_masked, errors='coerce')
            Y_masked = pd.to_numeric(Y_masked, errors='coerce')

            X_masked = X_masked.dropna()
            Y_masked = Y_masked.dropna()

            # Do the fit if possible
            if len(X_masked) <= 4 or len(X_belowthreshold) == 0:
                self.logger.error(f"{str(sweep._param).split('_')[-1]}, insufficient points below minimum current threshold to do any fitting")
                self.logger.attempt(f"{str(sweep._param).split('_')[-1]}, fitting data to linear fit to see if barrier is broken.")

                try:
                    # Guess a line fit with zero slope.
                    guess = (0,np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*self.abs_max_current)
                    fit_params, fit_cov = sp.optimize.curve_fit(getattr(self, 'linear'), X, Y, guess)
                    m, b = fit_params
                    plt.plot(X, self.linear(X,m,b), 'r-')
                    self.logger.warning(f"{str(sweep._param).split('_')[-1]}, data fits well to line, most likely barrier is shorted somewhere.")
                    continue 

                except RuntimeError:
                    self.logger.error(f"{str(sweep._param).split('_')[-1]}, linear fit failed.")

            else:
                # Try and fit and get fit params
                try: 
                    if str(sweep._param).split('_')[-1] in self.leads:

                        fit_function = getattr(self, 'logarithmic')
                        guess = [Y_masked.iloc[0], self.voltage_sign, X_masked.iloc[0] - self.voltage_sign, 0]
                    
                        self.logger.info(f"{str(sweep._param).split('_')[-1]}, fitting data to logarithmic function")
                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params
                        self.logger.info(f"Fit params: a = {a}, b = {b}, x0 = {x0}, y0 = {y0}")

                        V_pinchoff = float(round(np.exp(-y0/a)/b + x0,3))
                        self.results[str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                   
                    else:

                        fit_function = getattr(self, self.pinch_off_info['fit_function'])
                        guess = (Y.iloc[0], -1 * self.voltage_sign * 100, float(self.results['turn_on']['voltage']), 0)

                        self.logger.info(f"{str(sweep._param).split('_')[-1]}, fitting data to {self.pinch_off_info['fit_function']}")
                        fit_params, fit_cov = sp.optimize.curve_fit(fit_function, X_masked, Y_masked, guess)
                        a, b, x0, y0 = fit_params
                        self.logger.info(f"Fit params: a = {a}, b = {b}, x0 = {x0}, y0 = {y0}")

                        V_pinchoff = self.voltage_sign * float(round(min(
                            np.abs(x0 - np.sqrt(8) / b),
                            np.abs(x0 + np.sqrt(8) / b)
                        ),3))
                        V_pinchoff_width = float(abs(round(2 * np.sqrt(8) / b,2)))
                        self.results[str(sweep._param).split('_')[-1]]['pinch_off']['voltage'] = V_pinchoff
                        self.results[str(sweep._param).split('_')[-1]]['pinch_off']['width'] = V_pinchoff_width #V

                    plt.plot(X_masked, fit_function(X_masked, a, b, x0, y0), 'r-')
                    axes.axvline(x=V_pinchoff, alpha=0.5, linestyle=':', c='b', label=r'$V_{\min}$')
                    self.logger.info(f"{str(sweep._param).split('_')[-1]}, pinch off at {V_pinchoff}")
                    if str(sweep._param).split('_')[-1] not in self.leads:
                        axes.axvline(x=V_pinchoff + self.voltage_sign * V_pinchoff_width, alpha=0.5, linestyle='--', c='b', label=r'$V_{\max}$')
                        self.logger.info(f"{str(sweep._param).split('_')[-1]}, pinch off width of {V_pinchoff_width}")
                    axes.legend(loc='best')                
                
                except RuntimeError:
                    self.logger.error(f"{str(sweep._param).split('_')[-1]}, insufficient points below minimum current threshold to do any fitting")
                    self.logger.attempt(f"{str(sweep._param).split('_')[-1]}, fitting to linear fit to see if barrier is broken.")

                    try:
                        # Guess a line fit with zero slope.
                        guess = (0,np.sign(df_current[f'{self.multimeter_name}_current'].iloc[-1])*self.abs_max_current)
                        fit_params, fit_cov = sp.optimize.curve_fit(getattr(self, 'linear'), X, Y, guess)
                        m, b = fit_params
                        plt.plot(X, self.linear(X,m,b), 'r-')
                        self.logger.warning(f"{str(sweep._param).split('_')[-1]}, data fits well to line, most likely barrier is shorted somewhere.")

                    except RuntimeError:
                        self.logger.error(f"{str(sweep._param).split('_')[-1]}, linear fit failed.")         

            config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
            self._save_figure(plot_info=f"{str(sweep._param).split('_')[-1]}_pinch_{config_str}")


        return results_dict

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
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self._smoothly_set_voltage_configuration(voltage_configuration)
        else:
            self.logger.info(f"setting {self.leads} to {self.results['turn_on']['saturation']} V")
            self._smoothly_set_gates_to_voltage(self.leads, self.results['turn_on']['saturation'])

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
            self.logger.error("invalid dV")
            return

        # Double check barrier validity
        if B1 is None or B2 is None and len(self.barriers) == 2:
            B1, B2 = self.barriers
            self.logger.warning(f"no barriers provided, assuming {B1} and {B2} from device configuration")
        
        # Double check device bounds
        minV_B1, maxV_B1 = B1_bounds
        minV_B2, maxV_B2 = B2_bounds

        if minV_B1 is None:
            minV_B1 = self.results[B1]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B1) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (minV) for B1.")

        if minV_B2 is None:
            minV_B2 = self.results[B2]['pinch_off']['voltage']
        else:
            assert np.sign(minV_B2) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (minV) for B2.")

        if maxV_B1 is None:
            if self.voltage_sign == 1:
                maxV_B1 = min(self.results[B1]['pinch_off']['voltage']+self.voltage_sign*self.results[B1]['pinch_off']['width'], self.results['turn_on']['saturation'])
            elif self.voltage_sign == -1:
                maxV_B1 = max(self.results[B1]['pinch_off']['voltage']+self.voltage_sign*self.results[B1]['pinch_off']['width'], self.results['turn_on']['saturation'])
        else:
            assert np.sign(maxV_B1) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (maxV) for B1.")

        if maxV_B2 is None:
            if self.voltage_sign == 1:
                maxV_B2 = min(self.results[B2]['pinch_off']['voltage']+self.voltage_sign*self.results[B2]['pinch_off']['width'], self.results['turn_on']['saturation'])
            elif self.voltage_sign == -1:
                maxV_B2 = max(self.results[B2]['pinch_off']['voltage']+self.voltage_sign*self.results[B2]['pinch_off']['width'], self.results['turn_on']['saturation'])
        else:
            assert np.sign(maxV_B2) == self.voltage_sign, self.logger.error("double check the sign of the gate voltage (maxV) for B2.")

        self.logger.info(f"setting {B1} to {maxV_B1} V")
        self.logger.info(f"setting {B2} to {maxV_B2} V")
        self._smoothly_set_voltage_configuration({B1: maxV_B1, B2: maxV_B2})

        def smooth_reset():
            """Resets the inner loop variable smoothly back to the starting value
            """
            self._smoothly_set_gates_to_voltage([B2], maxV_B2)

        num_steps_B1 = self._calculate_num_of_steps(minV_B1, maxV_B1, dV_B1)
        num_steps_B2 = self._calculate_num_of_steps(minV_B2, maxV_B2, dV_B2)
        self.logger.attempt("barrier barrier scan")
        self.logger.info(f"stepping {B1} from {maxV_B1} V to {minV_B1} V")
        self.logger.info(f"sweeping {B2} from {maxV_B2} V to {minV_B2} V")
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
        self.logger.complete("\n")

        self.logger.info(f"returning gates {B1}, {B2} to {maxV_B1} V, {maxV_B2} V respectively")
        self._smoothly_set_gates_to_voltage([B1], maxV_B1)
        self._smoothly_set_gates_to_voltage([B2], maxV_B2)

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_name}_current', index=[f'{self.voltage_source_name}_{B1}'], columns=[f'{self.voltage_source_name}_{B2}'])
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

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{B1}_{B2}_sweep_{config_str}')

        if extract_bias_point:

            # Inference data
            bias_point, voltage_window = self.extract_bias_point(
                df,
                plot_process=True,
                axes=ax1
            )

            for barrier_gate_name, voltage in bias_point.items():
                self.results[barrier_gate_name]['bias_voltage'] = voltage



        return (df,ax1)

    def coulomb_blockade(self, 
                         gate: str = None, 
                         gate_bounds: tuple = (None, None), 
                         dV: float = None, 
                         delay: float = 0.01,
                         voltage_configuration: dict = None) -> tuple[pd.DataFrame, plt.Axes]:
        """Attempts to sweep plunger gate to see coulomb blockade features. Ideally
        the voltage configuration provided should be that which creates a well defined
        central dot between the two barriers.

        Args:
            gate (str, optional): Plunger gate. Defaults to None.
            gate_bounds (tuple, optional): Voltage bounds (V). Defaults to (None, None).
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
        assert voltage_configuration is not None, self.logger.error("no voltage configuration provided")

        self.logger.info(f"setting voltage configuration: {voltage_configuration}")
        self._smoothly_set_voltage_configuration(voltage_configuration)

        if gate is None:
            gate = self.plungers[0]

        if dV is None:
            dV = self.voltage_resolution

        minV_gate, maxV_gate = gate_bounds
        if minV_gate is None:
            minV_gate = 0
        assert maxV_gate is not None, self.logger.error(f"{gate}, maximum voltage not provided")
  
        num_steps_P = self._calculate_num_of_steps(minV_gate, maxV_gate, dV)
        gate_sweep = LinSweep_SIM928(getattr(self.voltage_source, f'{gate}'), minV_gate, maxV_gate, num_steps_P, delay, get_after_set=False)

        self.logger.info(f"setting {gate} to {minV_gate} V")
        self._smoothly_set_gates_to_voltage([gate], minV_gate)

        self.logger.attempt(f"sweeping gate {gate} from {minV_gate} V to {maxV_gate} V")
        # Execute the measurement
        result = qc.dataset.dond(
            gate_sweep,
            self.drain_volt,
            write_period=0.1,
            # break_condition=self._check_break_conditions,
            measurement_name='Coulomb Blockade',
            exp=self.initialization_exp,
            show_progress=True
        )
        self.logger.complete("\n")

        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)
        axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gate}', marker= 'o',s=10)
        df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'{self.voltage_source_name}_{gate}', ax=axes, linewidth=1)
        axes.set_title('Coulomb Blockade')
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))
        axes.legend(loc='best')

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{gate}_sweep_{config_str}')

        return (df, axes,)

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

        assert voltage_configuration is not None, self.logger.error("no voltage configuration provided")

        self.logger.info(f"setting voltage configuration: {voltage_configuration}")
        self._smoothly_set_voltage_configuration(voltage_configuration)

        if dV_ohmic is None:
            dV_ohmic = self.voltage_resolution
        
        if dV_gate is None:
            dV_gate = self.voltage_resolution
        
        minV_ohmic, maxV_ohmic = ohmic_bounds
        minV_gate, maxV_gate = gate_bounds

        self.logger.info(f"setting {ohmic} to {maxV_ohmic} V")
        self.logger.info(f"setting {gate} to {maxV_gate} V")
        self._smoothly_set_voltage_configuration({ohmic: maxV_ohmic, gate: maxV_gate})

        assert minV_gate is not None, self.logger.error(f"no minimum voltage for {gate} provided")
        assert maxV_gate is not None, self.logger.error(f"no maximum voltage for {gate} provided")
        assert minV_ohmic is not None, self.logger.error(f"no minimum voltage for {ohmic} provided")
        assert maxV_ohmic is not None, self.logger.error(f"no maximum voltage for {ohmic} provided")
        
        def smooth_reset():
            self._smoothly_set_gates_to_voltage([ohmic], maxV_ohmic)

        self.logger.info(f"stepping {gate} from {maxV_gate} V to {minV_gate} V")
        self.logger.info(f"sweeping {ohmic} from {maxV_ohmic} V to {minV_ohmic} V")
        num_steps_ohmic = self._calculate_num_of_steps(minV_ohmic, maxV_ohmic, dV_ohmic)
        num_steps_gate = self._calculate_num_of_steps(minV_gate, maxV_gate, dV_gate)

        self.logger.attempt("Coulomb diamond scan")
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
        self.logger.complete("\n")

        self.logger.info(f"setting gates {gate}, {ohmic} to {maxV_gate} V, {maxV_ohmic} V respectively")
        self._smoothly_set_voltage_configuration({ohmic: maxV_ohmic, gate: maxV_gate})

  
        # Get last dataset recorded, convert to current units
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()
        df_current = df_current.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity
        df_current.iloc[:,-2] = df_current.iloc[:,-2].mul(self.voltage_divider * 1e3) # sensitivity

        # Plot 2D colormap
        df_pivoted = df_current.pivot_table(values=f'{self.multimeter_name}_current', index=[f'{self.voltage_source_name}_{ohmic}'], columns=[f'{self.voltage_source_name}_{gate}'])
        gate_data, ohmic_data = df_pivoted.columns, df_pivoted.index
        raw_current_data = df_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        gate_grad = np.gradient(raw_current_data, axis=1)
        ohmic_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Coulomb Diamonds")

        im_aspect = 'auto'

        cbar_ax1 = plt.colorbar(ax1.imshow(
            raw_current_data,
            extent=[gate_data[0], gate_data[-1], ohmic_data[0], ohmic_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_aspect
        ), ax=ax1,fraction=0.046, pad=0.04)

        cbar_ax1.set_label(r'$I_{SD}$ (nA)')
        ax1.set_title(r'$I_{SD}$')
        ax1.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))
        ax1.set_ylabel(r"$V_{{{gate_name}}}$ (mV)".format(gate_name=ohmic))

        # V grad is actually horizontal
        grad_vector = (1,0) 

        cbar_ax2 = plt.colorbar(ax2.imshow(
            np.sqrt(grad_vector[0] * gate_grad**2 +   grad_vector[1]* ohmic_grad**2),
            extent=[gate_data[0], gate_data[-1], ohmic_data[0], ohmic_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_aspect
        ), ax=ax2,fraction=0.046, pad=0.04)

        cbar_ax2.set_label(r'$\nabla I_{SD}$ (nA/V)')
        ax2.set_title(r'$\nabla I_{SD}$')
        ax2.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=gate))
        ax2.set_ylabel(r"$V_{{{gate_name}}}$ (mV)".format(gate_name=ohmic))

        fig.tight_layout()

        config_str = json.dumps(voltage_configuration).replace(' ', '').replace('.', 'p').replace(',', '__').replace("\"", '').replace(":", '_').replace("{", "").replace("}", "")
        self._save_figure(plot_info=f'{ohmic}_{gate}_sweep_{config_str}')
        
        return (df, ax1)

    def current_trace(self, 
                      f_sampling: int,
                      t_capture: int, 
                      voltage_configuration: dict = None,
                      meas_name: str = "Current Trace",
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

        if voltage_configuration is not None:
            self.logger.info(f"setting voltage configuration: {voltage_configuration}")
            self._smoothly_set_voltage_configuration(voltage_configuration)

        time_param = qc.parameters.ElapsedTimeParameter('time')
        meas = qc.dataset.Measurement(exp=self.initialization_exp, name=meas_name)
        meas.register_parameter(time_param)
        meas.register_parameter(self.drain_mm_device.volt, setpoints=[time_param])

        self.logger.info(f"Running current trace for {t_capture} seconds")
        with meas.run() as datasaver:
            time_param.reset_clock()
            elapsed_time = 0
            while elapsed_time < t_capture:
                datasaver.add_result((self.drain_mm_device.volt, self.drain_mm_device.volt()),
                                (time_param, time_param()))
                elapsed_time = time_param.get()
      
        df = datasaver.dataset.to_pandas_dataframe().reset_index()
        df_current = df.rename(columns={f'{self.multimeter_name}_volt': f'{self.multimeter_name}_current'})
        df_current.iloc[:,-1] = df_current.iloc[:,-1].subtract(self.preamp_bias).mul(self.preamp_sensitivity) # sensitivity

        # Plot current v.s. random gate (since they are swept together)

        axes = df_current.plot.scatter(y=f'{self.multimeter_name}_current', x='time', marker= 'o',s=5)
        df_current.plot.line(y=f'{self.multimeter_name}_current', x=f'time', ax=axes, linewidth=1)
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$t$ (s)')
        axes.set_title(rf'Current noise, $f_s={f_sampling}$ Hz, $t_\max={t_capture}$ s')
        plt.show()
        
        if plot_psd:

            # Plot noise spectrum
            
            t = df_current[f'time']
            I = df_current[f'{self.multimeter_name}_current']
            f, Pxx = sp.signal.periodogram(I, fs=f_sampling, scaling='density')

            plt.loglog(f, Pxx)
            plt.xlabel(r'$\omega$ (Hz)')
            plt.ylabel(r'$S_I$ (A$^2$/Hz)')
            plt.title(r"Current noise spectrum")
            plt.show()

        return df

    def _load_config_files(self):
        
        # Read the tuner config information

        self.tuner_info = yaml.safe_load(Path(self.tuner_config_file).read_text())
        self.global_turn_on_info = self.tuner_info['global_turn_on']
        self.pinch_off_info = self.tuner_info['pinch_off']

        # Read the config information

        self.config = yaml.safe_load(Path(self.config_file).read_text())
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
        
        #self.ohmics = []
        #self.barriers = []
        #self.leads = []
        #self.plungers = []
        #self.accumulation = []
        #self.screening = []

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



        self.abs_max_current = self.config['device']['constraints']['abs_max_current']
        self.abs_max_gate_voltage = self.config['device']['constraints']['abs_max_gate_voltage']
        self.abs_max_gate_differential = self.config['device']['constraints']['abs_max_gate_differential']

        self.voltage_source_name = self.config['setup']['voltage_source']
        self.multimeter_name = self.config['setup']['multimeter']
        self.voltage_divider = self.config['setup']['voltage_divider']
        self.preamp_bias = self.config['setup']['preamp_bias']
        self.preamp_sensitivity = self.config['setup']['preamp_sensitivity']
        self.voltage_resolution = self.config['setup']['voltage_resolution']

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

        # First, we determine which gates to are being swept and ensure the correct sign is maintained

        gates_to_check = self.leads + self.barriers + self.screening + self.accumulation + self.plungers

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
                                          deltav[gate_name]*(maxsteps-s-1)/maxsteps
        else:
            
            # First, we set up the values of the sweeps and save them

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

        # Now, we loop over intermediate to set the voltages

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

    def _save_figure(self, plot_info: str, plot_dpi: int = 300, plot_format: str = 'svg'):
        """Saves plot generated from one of the stages.

        Args:
            plot_info (str): Plot information for the filename.
            plot_dpi (int, optional): Dots Per Inch (DPI) for the plot. Defaults to 300.
            plot_format (str, optional): Picture format file type. Defaults to 'svg'.
        """

        completition_time = str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute) # the current minute
        plot_name = os.path.join(self.db_folder, f"{plot_info}_{completition_time}.{plot_format}")
        plt.savefig(fname=plot_name, dpi=plot_dpi, format=plot_format)