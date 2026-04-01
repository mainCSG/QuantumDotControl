# Import modules

import yaml, datetime, sys, time, os, shutil, json,re
from pathlib import Path

import inspect

import pandas as pd

import numpy as np

import scipy as sp
from scipy.ndimage import convolve

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from typing import List, Dict, Callable

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


class DataAnalysis:
    
    def __init__(self, 
                 logger, 
                 tuner_config) -> None:
        
        self.logger = logger

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
    
    def relu(self, x, a, x0, b):
        return np.maximum(0, a * (x - x0) + b)

    def fit_to_function(self, 
                        x_data, 
                        y_data, 
                        function: Callable):
        
        popt, pcov = sp.optimize.curve_fit(function, x_data, y_data)
        perr = np.sqrt(np.diag(pcov))

        params = list(inspect.signature(function).parameters.keys())[1:]

        for name, val, err in zip(params, popt, perr):
            print(f"{name} = {val:.3f} ± {err:.3f}")

        return params, popt, pcov
    
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