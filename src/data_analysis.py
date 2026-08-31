# Standard library
import datetime
import inspect
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import copy
from pathlib import Path
from typing import Callable, Dict, List
from venv import logger

# Third-party libraries
import numpy as np
import numpy.typing as npt
import pandas as pd
import cv2
import scipy.signal as signal
from scipy.interpolate import make_smoothing_spline

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import ConnectionPatch, Rectangle

from IPython.display import display

from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.ndimage import convolve, map_coordinates, gaussian_filter1d, gaussian_filter

import skimage
from skimage import filters, transform
from skimage.feature import canny
from skimage.filters import threshold_otsu, sato
from skimage.morphology import diamond, rectangle
from skimage.transform import probabilistic_hough_line

import yaml
from colorlog import ColoredFormatter

import qcodes as qc
from qcodes.dataset import AbstractSweep, Measurement
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase

from nicegui import ui
from tunerlog import TunerLog
from gui_bridge import tuning_bridge

logger = TunerLog('Data Analysis')

#TODO: delete this
def test_run():
    logger.info('we were able to run a function from this thing')
    return
  
def logarithmic(x, a, b, x0, y0):
    """
    Description
    -----------
    Logarithmic model used for curve fitting.

    Parameters
    -----------
    x : np.array
        independent variable array
    a : float
        vertical scale
    b : float
        horizontal scale
    x0 : float
        horizontal shift
    y0 : float
        vertical shift
    """
    return a * np.log(b*(x-x0)) + y0

def exponential(x, a, b, x0, y0):
    """
    Description
    -----------
    Exponential model used for curve fitting.

    Parameters
    -----------
    x : np.array
        independent variable array
    a : float
        vertical scale
    b : float
        growth/decay rate
    x0 : float
        horizontal shift
    y0 : float
        horizontal asymptote
    """
    return a * np.exp(b * (x-x0)) + y0

def sigmoid(x, a, b, x0, y0):
    """
    Description
    -----------
    Sigmoid model used for pinch-off fitting.

    Parameters
    -----------
    x : np.array
        independent variable array
    a : float
        vertical amplitude
    b : float
        steepness rate
    x0 : float
        inflection point
    y0 : float
        lower horizontal asymptote
    """
    return a * expit(-b * (x - x0)) + y0

def linear(x, m, b):
    """
    Description
    -----------
    Simple linear model for fitting straight-line behavior.

    Parameters
    -----------
    x : np.array
        independent variable array
    m : float
        slope
    b : float
        x-intercept
    """
    return m * x + b         

def relu(x, a, x0):
    """
    Description
    -----------
    ReLU-style model that is zero below x0 and linear above it. Originally used for turn-on fitting.

    Parameters
    -----------
    x : np.array
        independent variable array
    a : float
        slope
    x0 : float
        hinge point
    """
    return np.maximum(0, a * (x - x0))

def gompertz(x, a, b, c):
    """
    Description
    -----------
    Gompertz model used as alternative to sigmoid for pinch-off fitting. Type of sigmoid used for highly asymmetric cases.

    Parameters
    -----------
    x : np.array
        independent variable array
    a : float
        upper asymptote
    b : float
        horizontal shift/delay
    c : float
        growth rate
    """
    return a * np.exp(-b * np.exp(-c * x))

def fit_to_function(x_data: np.array, 
                    y_data: np.array, 
                    function: Callable,
                    p0: list[float] = None,
                    print_results: bool = True
                    ):
    """
    Description
    -----------
    Fit a provided model function to x/y data using nonlinear least squares.

    Parameters
    -----------
    x_data : np.array
        independent variable values
    y_data : np.array
        dependent variable values
    function : Callable
        callable model to fit (e.g. sigmoid)
    p0 : list[float]
        optional initial guess for model parameters
    print_results : bool
        prints fitted parameter values

    Returns
    -----------
    params : list[str]
        model parameter names
    popt : np.array
        optimized parameter values
    perr : np.array
        Error of parameter estimates (square root of covariances)
    """
    
    if p0 is None:
        # If no initial guess for the function parameters are given by the user, apply no guess
        popt, pcov = curve_fit(function, x_data, y_data, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    else:
        # Otherwise use the given initial guess given by user
        popt, pcov = curve_fit(function, x_data, y_data, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    
    params = list(inspect.signature(function).parameters.keys())[1:] # List of fitting parameters

    if print_results:
        # For every parameter, print it's value and error
        for name, val, err in zip(params, popt, perr):
            print(f"{name} = {val:.3f} ± {err:.3f}")

    return params, popt, perr

def extract_turn_on_voltage(x_data: np.array,
                            y_data: np.array,
                            noisefloor: float,
                            filepath: str,
                            filename: str
                            ):
    """
    Description
    -----------
    Estimate the turn-on voltage from a gate-sweep current curve.
    This routine finds the first voltage where the current rises an
    order of magnitude above the provided noise floor.

    Parameters
    -----------
    x_data : np.array
        independent variable (voltage) values
    y_data : np.array
        dependent variable (current) values
    noisefloor : float
        baseline noise of the gate-sweep
    filepath : str
        name of directory to save turn-on plot in
    filename : str
        name of file to save the turn-on plot under

    Returns
    -----------
    turnon_voltage : float
        first voltage found an order of magnitude above the noise floor
    """

    # --- Data definitions ---
    
    # Ensures numpy array 
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    if y1[-1] < 0:
        # Flips current sign if SD bias was inversed
        y1 = -y1

    # --- Finding Turn-On Point ---
    turnon_voltage = 0
    turnon_current = 0

    threshold = noisefloor * 10 # When the turn-on will be detected, 10 times the noise floor

    for val in y1:
        # Going through every value in the current data, the first value above the threshold is considered turn-on
        if val > abs(threshold):
            idx_turnon = np.where(y1 == val)[0][0] # Get the index of the turn-on point
            
            logger.info(f"Index: {idx_turnon}")
            
            turnon_voltage = x1[idx_turnon - 1] # Get the voltage of the turn-on point
            
            logger.info(f"Turn_On Voltage: {turnon_voltage}")
            
            turnon_current = y1[idx_turnon - 1] # Get the current of the turn-on point
            break

    # --- Plot data ---

    # Raw data
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x1, y1, '-', color='C0', linewidth=2, label='I ($V_{gate}$)')

    # Save raw data
    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)

    logger.info(f"{filepath_raw_data}")

    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    # Turn-on point
    ax.scatter(turnon_voltage, turnon_current, color='red', s=100, zorder=5, label='Turn-On Point')
    ax.legend(fontsize=24, frameon=False, loc='upper left')

    # --- Labels and formatting ---

    ax.set_xlabel(r'V$_{gate}$ (V)', fontsize=35)
    ax.set_ylabel('I (nA)', fontsize=35)

    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
    ax.tick_params(direction='in', length=5, width=1.2, labelsize=18, top=True, right=True)

    xticks_span = np.linspace(x1.min(), x1.max(), 5)

    ax.set_xticks(xticks_span)
    ax.set_xticklabels([f'{xticks_span[0]:.1f}', '', f'{xticks_span[2]:.1f}', '', f'{xticks_span[-1]:.1f}'], fontsize=25)

    yticks_span = np.linspace(y1.min(), y1.max(), 5)

    ax.set_yticks(yticks_span)
    ax.set_yticklabels([f'{np.abs(yticks_span[0]):.1f}', '', '', '', f'{yticks_span[-1]:.4f}'], fontsize=25)

    plt.tight_layout()

    # Save final data
    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)

    logger.info(f"{filepath_analyzed}")

    fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    metrics = {'Turn-On Voltage (V)': turnon_voltage}

    payload = {
                'stage': 'Bootstrapping',
                'step_name': 'Turn-On',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)

    # --- Print summary ---
    #print(f"  Turn-on Voltage:  {turnon_voltage:.3f} V")

    return turnon_voltage

def extract_pinch_off_curve_ranges(x_data: np.array,
                                   y_data: np.array,
                                   noisefloor: float,
                                   gate_type: str,
                                   gate_name: str,
                                   filepath: str,
                                   filename: str
                                   ):
    """
    Description
    -----------
    Identify pinch-off and saturation voltage range for a gate-sweep. This function normalizes
    the sign of the current, selects the scan direction from the zero-voltage point, finds the
    pinch-off position using slope detection.
    
    The saturation voltage is found depending on the
    type of gate. For accumulation gates, it uses the point of slowest saturation from the gompertz
    fit. For plunger gates, it uses the upper asymptote from the gompertz fit as a threshold to
    determine the first voltage that goes above it. For barrier gates, it uses midpoint voltage
    calculated from the gompertz fit.

    Parameters
    -----------
    x_data : np.array
        independent variable (voltage) values
    y_data : np.array
        dependent variable (current) values
    noisefloor : float
        baseline noise of the gate-sweep
    gate_type : str
        the type of gate being pinched off, options are 'Accumulation', 'Barrier', and 'Plunger'
    gate_name: str
        name of the gate being pinched-off
    filepath : str
        name of directory to save pinch-off plot in
    filename : str
        name of file to save the pinch-off plot under

    Returns
    -----------
    voltage_window : tuple(float)
        for accumulation gates: (pinch-off voltage, saturation voltage),
        for plunger gates: (pinch-off voltage, saturation voltage),
        for barriers gates: (pinch-off voltage, midpoint voltage)
    """

    # --- Data definitions ---
    # Ensures numpy array 
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    if y1[0] < 0:
        # Flips current sign if SD bias was inversed
        y1 = -y1

    try:
        y1_norm = y1/np.max(y1) # Normalizes the data for fitting
    except Exception:
        y1_norm = y1


    # --- Finding Pinch-off Voltage ---

    # Ensures we scan the data from x-values closest to 0V to values away from it
    start_idx = int(np.argmin(np.abs(x1)))
    if start_idx == 0:
        step = 1
    elif start_idx == len(x1) - 1:
        step = -1
    else:
        left_abs = abs(x1[start_idx - 1])
        right_abs = abs(x1[start_idx + 1])
        step = 1 if right_abs >= left_abs else -1

    scan_indices = np.arange(start_idx, len(x1), step) if step > 0 else np.arange(start_idx, -1, -1) # Mask to scan over
    x_scan = x1[scan_indices]
    y_scan = y1[scan_indices]
    
    # Selects the first 5% of data points, closest to 0V, to see if there are oscillations from pinch-off
    baseline_window = max(5, int(0.05 * len(y_scan)))
    baseline_data = y_scan[:baseline_window]
    baseline_std = np.std(baseline_data)
    epsilon = 0.3
    total_signal_range = np.max(y_scan) - np.min(y_scan)

    # If the start already exhibits heavy oscillations relative to the total range,
    # it means the device turn-on is active right from the initial gate voltage.
    if baseline_std > 0.02 * total_signal_range:
        pinch_off_pos = 0
    else:
        # Otherwise, find where the current goes above the noise floor
        departure_threshold = noisefloor + epsilon
        pinch_off_pos = 0
        consecutive_points_needed = 3
        for i in range(len(y_scan) - consecutive_points_needed):
            if all(y_scan[i + j] > departure_threshold for j in range(consecutive_points_needed)):
                pinch_off_pos = i
                break

    early_rise_threshold = max(3, int(0.05 * len(x_scan)))
    if pinch_off_pos <= early_rise_threshold or pinch_off_pos >= len(x_scan) - 1:
        idx_pinch_off = int(scan_indices[0])
        pinch_off_pos = 0
    else:
        idx_pinch_off = int(scan_indices[pinch_off_pos])

    # Local peak adjustment fallback
    if 0 < pinch_off_pos < len(y_scan) - 1:
        if y_scan[pinch_off_pos] >= y_scan[pinch_off_pos - 1] and y_scan[pinch_off_pos] >= y_scan[pinch_off_pos + 1]:
            look_ahead = min(len(y_scan), pinch_off_pos + max(3, int(0.05 * len(y_scan))))
            post_peak = y_scan[pinch_off_pos + 1:look_ahead]
            if post_peak.size > 0:
                min_rel = np.argmin(post_peak)
                min_pos = pinch_off_pos + 1 + min_rel
                if y_scan[pinch_off_pos] - y_scan[min_pos] > max(1e-6, 0.05 * abs(y_scan[pinch_off_pos])):
                    pinch_off_pos = min_pos
                    idx_pinch_off = int(scan_indices[pinch_off_pos])

    # Goes 2 indices towards pinch-off for pinch-off (offset for epsilon)
    if idx_pinch_off == len(x1) - 1:
        idx_reduce = 0
    else:
        idx_reduce = 2
    
    # Finds the voltage and current
    pinch_off_voltage = x1[idx_pinch_off + idx_reduce]
    pinch_off_current = y1_norm[idx_pinch_off + idx_reduce]

    # --- Fit sigmoids (Gompertz function) ---

    params, popt, pcov = fit_to_function(x1, y1_norm, gompertz, print_results=False) # For plotting
    params2, popt2, pcov2 = fit_to_function(x1, y1, gompertz, print_results=False) # For fit calculations

    # --- Extract key points ---
   
    A, B, C = popt
    A2, B2, C2 = popt2

    # Calculate parameters from fit
    factor = np.log((3 + np.sqrt(5)) / 2)
    fit_midpoint_voltage = np.log(B2) / C2
    fit_pinch_off_voltage = fit_midpoint_voltage - (factor / C2)
    fit_saturation_voltage = 0.8 * A2

    sat_voltage = None
    sat_current = None

    # Ensures saturation voltage isn't above the maximum gate voltage
    if fit_saturation_voltage > 1.5:
        fit_saturation_voltage = 1.5

    y_fit = gompertz(x1, *popt) # Fit data for plotting

    # Saturation voltage and current determination based on Gate type
    if gate_type == 'Accumulation':
        sat_voltage = fit_saturation_voltage
        sat_current = y1_norm[np.abs(x1 - sat_voltage).argmin()]

    elif gate_type == 'Plunger':
        sat_voltage = x1[np.abs(y1 - A2).argmin()]
        sat_current = y1_norm[np.abs(y1 - A2).argmin()]

    elif gate_type == 'Barrier':
        a = A2 * 0.9
        sat_voltage = x1[np.abs(y1 - a).argmin()]
        sat_current = y1_norm[np.abs(x1 - sat_voltage).argmin()]

    else:
        # Switch to Logging error so protocol doesn't crash
        raise TypeError("The gate_type given isn't one of the following: 'Accumulation', 'Plunger', 'Barrier'")

    # --- Plot data ---

    # Raw data
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x1, y1_norm, '-', color='C0', linewidth=2, label='I ($V_{gate}$)')

    # Save raw data
    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)

    logger.info(f"{filepath_raw_data}")
    
    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    # Plot pinch-off and saturation points, and gompaertz sigmoid fit
    ax.scatter(pinch_off_voltage, pinch_off_current, color='red', s=100, zorder=5, label='Pinch-off Point')
    ax.scatter(sat_voltage, sat_current, color='green', s=100, zorder=5, label='Saturation Point')
    ax.plot(x1, y_fit, '--', color='red', linewidth=2, label='Fitted Gompertz')

    ax.legend(fontsize=20, frameon=False, loc='upper left')

    # --- Double-sided arrows showing full range ---

    # Define arrow y-positions (swap positions)
    y_arrow = ax.get_ylim()[1] + 0.05  # Device arrow

    # Device arrow (now above)
    ax.annotate(
        '', xy=(sat_voltage, y_arrow), xytext=(pinch_off_voltage, y_arrow),
        arrowprops=dict(arrowstyle='<->', color='C0', lw=3.0, shrinkA=0, shrinkB=0),
        annotation_clip=False
    )
    ax.text((sat_voltage + pinch_off_voltage)/2, y_arrow - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            s='', color='C0', ha='center', va='top', fontsize=20)

    # --- Characteristic vertical lines extending to the data points ---

    y_pinch = y1_norm[np.argmax(np.isclose(x1, pinch_off_voltage, atol=1e-3))]

    for color, po, sat, label, y_arrow, direction, y_pinch, y_sat in [
        # Device → arrow above, extend down to data
        ('C0', pinch_off_voltage, sat_voltage, 'Device 1', y_arrow, 'down', y_pinch, sat_current)
    ]:
        if direction == 'up':
            # Extend upward from arrow to the y-values of the fitted curve
            ax.vlines(po, ymin=y_arrow, ymax=y_pinch - 0.01, colors=color, linestyles='--', alpha=0.6)
            ax.vlines(sat, ymin=y_arrow, ymax=y_sat - 0.025, colors=color, linestyles='--', alpha=0.6)
        else:
            # Extend downward from arrow to the y-values of the fitted curve
            ax.vlines(po, ymin=y_pinch + 0.02, ymax=y_arrow, colors=color, linestyles='--', alpha=0.6)
            ax.vlines(sat, ymin=y_sat - 0.015, ymax=y_arrow, colors=color, linestyles='--', alpha=0.6)

    # --- Labels and formatting ---

    if gate_type == 'Plunger':
        ax.set_xlabel(r'V$_{Plunger}$ (V)', fontsize=35)

    elif gate_type == 'Accumulation':
        ax.set_xlabel(r'V$_{Accumulation}$ (V)', fontsize=35)

    elif gate_type == 'Barrier':
        ax.set_xlabel(r'V$_{Barrier}$ (V)', fontsize=35)
    
    ax.set_ylabel('I (nA)', fontsize=35)

    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
    ax.tick_params(direction='in', length=5, width=1.2, labelsize=18, top=True, right=True)

    xticks_span = np.linspace(x1.min(), x1.max(), 5)

    ax.set_xticks(xticks_span)
    ax.set_xticklabels([f'{xticks_span[0]:.2f}', '', f'{xticks_span[2]:.2f}', '', f'{xticks_span[-1]:.2f}'], fontsize=25)

    yticks_span = np.linspace(y1_norm.min(), y1_norm.max(), 5)

    ax.set_yticks(yticks_span)
    ax.set_yticklabels([f'{y1.min():.2f}', '', '', '', f'{y1.max():.3f}'], fontsize=25)

    # Extend y-limits slightly to make space for arrows

    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])

    plt.tight_layout()

    # Define a voltage/pinch-off window to return for next step in protocol
    voltage_window = (pinch_off_voltage, sat_voltage)

    # Save final plot
    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
    fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    metrics = {'Pinch-Off Voltage (V)': voltage_window[0],
               'Saturation Voltage (V)': voltage_window[1]}

    payload = {
                'stage': 'Bootstrapping',
                'step_name': f'{gate_name} Pinch-Off',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)
    # plt.show()

    return voltage_window

def extract_max_conductance_points(x_data: np.array,
                                   y_data: np.array,
                                   filepath: str,
                                   filename: str,
                                   peak_height_factor: list[float] = [None, None],
                                   abs_peak_height: list[float] = [None, None],
                                   peak_prominence_factor: list[float] = [None, None],
                                   peak_width: list[tuple] = [None, None]
                                   ):
    """
    Description
    -----------
    Analyze current data to identify the largest conductance features.

    This function plots the current and its derivative, then highlights
    the most extreme conductance peaks and valleys.

    Parameters
    -----------
    x_data : np.array
        independent variable (voltage) values
    y_data : np.array
        dependent variable (current) values
    filepath : str
        name of directory to save conductance plot in
    filename : str
        name of file to save the conductance plot under
    peak_height_factor : list[float]
        percentage of maximum conductance to set the minimum absolute peak height threshold in conductance, for [positive peaks, negative peaks]
    abs_peak_height : list[float]
        absolute minimum conductance threshold for peak detection
    peak_prominence_factor : list[float]
        percentage of maximum conductance to set the minimum peak prominence threshold in conductance, for [positive peaks, negative peaks]
    peak_width : list[tuple]
        peak width range in pixel space, for [positive peaks, negative peaks]

    Returns
    -----------
    best_sens_pts : list[tuple]
        the most positive and negative conductance peaks in current space
    all_sens_pts : tuple(np.array, np.array)
        first np.array is the voltages where all the peaks occur, second np.array is their corresponding conductances 
    """

    # --- Data definitions ---
    
    # Ensures numpy array 
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    # Calculate the derivative
    dIdV = np.gradient(y1, x1)

    # Ensures only 1 kind of height threshold is given
    if abs_peak_height != [None, None] and peak_height_factor != [None, None]:
        # Switch to Logging error so protocol doesn't crash
        raise ValueError("Both abs_peak_height and peak_height_factor values were given. Please ensure only one of these arguments are given!")

    # Implements default if no user-defined values are given

    if abs_peak_height != [None, None]:
        peak_height = [abs_peak_height[0], abs_peak_height[1]]
    elif peak_height_factor == [None, None]:
        peak_height = [0.25 * np.max(dIdV), 0.25 * np.max(dIdV)]
    else:
        peak_height = [peak_height_factor[0] * np.max(dIdV), peak_height_factor[1] * np.max(dIdV)]

    if peak_prominence_factor == [None, None]:
        peak_prominence = [0.3 * np.max(dIdV), 0.3 * np.max(dIdV)]
    else:
        peak_prominence = [peak_prominence_factor[0] * np.max(dIdV), peak_prominence_factor[1] * np.max(dIdV)]

    # --- Find two largest and two smallest conductance points (positive + negative extremes) ---

    peak_idx_pos, _ = signal.find_peaks(dIdV, height = peak_height[0], prominence = peak_prominence[0], width=peak_width[0])
    peak_idx_neg, _ = signal.find_peaks(-dIdV, height = peak_height[1], prominence = peak_prominence[1], width=peak_width[1])

    peak_idx = np.sort(np.concatenate([peak_idx_pos, peak_idx_neg]))

    # Check if no peaks are found, and safely end the function whilst saving the raw data

    if peak_idx.size == 0:

        logger.info("No conductance peaks were found")

        # Saves data even if no peaks are found, so the user can see why the peak detection failed
        fig = plt.figure()
        plt.plot(x1, y1)
        plt.close(fig)

        filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
        fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

        raise ValueError("No conductance peaks were found") # Switch to Logging error so protocol doesn't crash

    # Extract data points from peaks

    x_top = x1[peak_idx]
    I_top = y1[peak_idx]
    G_top = dIdV[peak_idx]

    max_idx = peak_idx[np.argmax(dIdV[peak_idx])]
    min_idx = peak_idx[np.argmin(dIdV[peak_idx])]

    x_max = x1[max_idx]
    x_min = x1[min_idx]
    I_max = y1[max_idx]
    I_min = y1[min_idx]
    G_max = dIdV[max_idx]
    G_min = dIdV[min_idx]

    best_sens_pts = [(x_max, I_max), (x_min, I_min)]
    all_sens_pts = (x_top, G_top)

    # Create two subplots that share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6))

    # --- Top panel: Current Data ---
    ax1.plot(x1, y1, color='#2c5aa0', linewidth=1)

    # --- Bottom panel: Conductance Data ---
    ax2.plot(x1, dIdV, color='#2c5aa0', linewidth=1)

    # Save raw data
    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    # --- Top panel: Current Analysis and Labelling ---
    for i in range(len(x_top)):
        if I_top[i] == I_max:
            ax1.scatter(x_top[i], I_top[i], facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Max I')
        elif I_top[i] == I_min:
            ax1.scatter(x_top[i], I_top[i], facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Min I')
        else:
            ax1.scatter(x_top[i], I_top[i], facecolors='none', edgecolors="#FF5500", s=100, linewidths=2, zorder=5, label='High Sensitivity Points')
    ax1.set_ylabel('I (nA)', fontsize=35)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(min(x1), max(x1))
    ax1.tick_params(labelbottom=True)

    # --- Bottom panel: Conductance Analysis and Labelling ---
    for i in range(len(x_top)):
        if G_top[i] == G_max:
            ax2.scatter(x_top[i], G_top[i], facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Max G')
        elif G_top[i] == G_min:
            ax2.scatter(x_top[i], G_top[i], facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Min G')
        else:
            ax2.scatter(x_top[i], G_top[i], facecolors='none', edgecolors="#FF5500", s=100, linewidths=2, zorder=5, label='High Sensitivity Points')
    ax2.set_xlabel(r'$V_P$ (V)', fontsize=35)
    ax2.set_ylabel('G (nS)', fontsize=35)
    ax1.set_ylim(bottom=0)
    ax2.set_xlim(min(x1), max(x1))

    # --- Create the connection line between top and bottom panels ---
    for i in range(len(x_top)):
        con = ConnectionPatch(
            xyA=(x_top[i], I_top[i]), coordsA=ax1.transData,
            xyB=(x_top[i], G_top[i]), coordsB=ax2.transData,
            color='#FF5500', linestyle='--', linewidth=0.7
        )
        fig.add_artist(con)

    # --- Create custom legend entries (hollow circles) ---
    legend_marker = mlines.Line2D([], [], color='#FF5500', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='High Sensitivity Points')
    
    legend_marker_2 = mlines.Line2D([], [], color='#01FF05', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='Best Sensitivity Points')

    # --- Custom tick labels: only min and max shown ---

    # Get existing ticks
    for ax in [ax1, ax2]:

        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
        ax.tick_params(direction='in', length=5, width=1.2, labelsize=20, top=True, right=True)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        
    ax1.set_xticks([np.round(x1.min(), 3), np.round((x1.min() + x1.max()) / 2, 3), np.round(x1.max(), 3)])
    ax1.set_xticklabels([str(np.round(x1.min(), 3)), str(np.round((x1.min() + x1.max()) / 2, 3)), str(np.round(x1.max(), 3))], fontsize=25)

    ax1.set_yticks([0, np.round(y1.max(), 3)])
    ax1.set_yticklabels(['0', str(np.round(y1.max(), 3))], fontsize=25)

    ax2.set_xticks([np.round(x1.min(), 3), np.round((x1.min() + x1.max()) / 2, 3),  np.round(x1.max(), 3)])
    ax2.set_xticklabels([str(np.round(x1.min(), 3)), str(np.round((x1.min() + x1.max()) / 2, 3)), str(np.round(x1.max(), 3))], fontsize=25)

    ax2.set_yticks([np.round(dIdV.min(), 1), 0, np.round(dIdV.max(), 1)])
    ax2.set_yticklabels([str(np.round(dIdV.min(), 1)), '0', str(np.round(dIdV.max(), 1))], fontsize=25)

    ax1.legend(handles=[legend_marker, legend_marker_2], loc='upper left', fontsize=16, frameon=False)

    # --- Adjust layout ---
    plt.subplots_adjust(hspace=0.40)
    # plt.show()
    
    # Saves final plot
    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
    fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    metrics = {'Best Sensitivity Point': best_sens_pts}

    payload = {
                'stage': 'Bootstrapping',
                'step_name': 'All Coulomb Blockade Peaks with All Sensivity Points',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)

    return best_sens_pts, all_sens_pts

def extract_working_point(lb_data: np.array,
                          rb_data: np.array,
                          current_data: np.array,
                          gates: list[str],
                          DotTuning: str,
                          barrier_pinch_offs: list[float],                          
                          filepath: str,
                          filename: str,
                          minAngleDeg: float = -60,
                          maxAngleDeg: float = -30,
                          minLineLength: int = 50,
                          maxLineGap: int = 200,
                          debug: bool = False
                          ):
    """
    Description
    -----------
    Find working-point lines in a 2D barrier sweep image.

    This function converts raw barrier voltage and current data into an image,
    applies ridge detection and Hough transform filtering, and returns the
    extracted working-points and lines that correspond to relevant device ridges.

    Parameters
    -----------
    lb_data : np.array
        independent left barrier voltage values
    rb_data : np.array
        independent right barrier voltage values
    current_data : np.array
        dependent current values
    gates : list[str]
        barrier gate names, [left barrier, right barrier]
    DotTuning : str
        'SET' for charge sensor tuning, and 'Triple Dot' for triple dot tuning
    barrier_pinch_offs : list[float]
        barrier gate pinch-off voltages, [left barrier, right barrier]
    filepath : str
        name of directory to save barrier-barrier plot in
    filename : str
        name of file to save the barrier-barrier plot under
    minAngleDeg : float
        minimum angle threshold to find working-point/hough lines in degrees
    maxAngleDeg : float
        maximum angle threshold to find working-point/hough lines in degrees
    minLineLength : int
        minimum length of hough lines in pixel space
    maxLineGap : int
        maximum gap between 2 hough lines in pixel space
    debug : bool
        toggles debug mode, which plots images of each step in the process

    Returns
    -----------
    best_shifted_point : tuple(float)
        the working-point closest to both barriers pinch-off voltage (bottom-left corner of plot).
    shifted_points : list
        all working-point candidates found for DotTuning of 'Triple Dot'
    perp_bias_points : list
        all working-point candidates found for DotTuning of 'SET'
    perp_traces_for_plot : list
        the perpendicular lines found at each working-point candidate
    """

    # --- Data definitions ---
    
    # Ensures numpy array
    lb_data = np.array(lb_data)
    rb_data = np.array(rb_data)
    current_data = np.array(current_data)
    barrier_pinch_offs = np.array(barrier_pinch_offs)
    device_type = 'electron'

    ux = np.unique(lb_data)
    uy = np.unique(rb_data)

    # Define your clipping thresholds here
    lb_min, lb_max = barrier_pinch_offs[0], ux.max()
    rb_min, rb_max = barrier_pinch_offs[1], uy.max()

    # Create a boolean mask matching the original 1D data structure
    clip_mask = (
        (lb_data >= lb_min) & (lb_data <= lb_max) & 
        (rb_data >= rb_min) & (rb_data <= rb_max)
    )

    # Apply the clipping mask to all arrays
    lb_data = lb_data[clip_mask]
    rb_data = rb_data[clip_mask]
    current_data = current_data[clip_mask]

    # Handle polarity
    if current_data[0] < 0:
        current_data = -current_data

    # Determines device type
    device_type = 'hole'
    if np.average(lb_data) > 0 and np.average(rb_data) > 0:
        current_data = np.flip(current_data, axis=None)
        device_type = 'electron'

    # Calculate new dimensions based on unique clipped values
    nx_new = len(np.unique(lb_data))
    ny_new = len(np.unique(rb_data))

    # Reshape the 1D clipped data into a 2D grid
    if current_data.ndim == 1:
        # Verify the clipped size matches the expected 2D grid dimensions
        if current_data.size == nx_new * ny_new:
            current_data = current_data.reshape((ny_new, nx_new))
        else:
            # Switch to Logging error so protocol doesn't crash
            raise ValueError("Clipped data size does not form a perfect rectangular grid.")

    ny, nx = current_data.shape

    # Here, we define the voltage ranges for the barriers
    lb_voltages = np.linspace(lb_data.min(), lb_data.max(), nx)
    rb_voltages = np.linspace(rb_data.min(), rb_data.max(), ny)
    
    logger.info("calculation starting...")

    # --- Gradient Calculation and Ridge Detection ---
    
    # Now, compute the gradient and the log of the gradient
    Gx, Gy = np.gradient(current_data)
    G = (1.0 / np.sqrt(2.0)) * np.sqrt(Gx**2 + Gy**2)

    g_lo, g_hi = np.percentile(G, [2, 98])
    G_clipped = np.clip(G, g_lo, g_hi)
    G_scaled = (G_clipped - g_lo) / (g_hi - g_lo)

    G_uint = (255 * G_scaled).astype(np.uint8)

    low = int(0.10 * 255) # Discard noise
    high = int(0.35 * 255) # Discard strongest boundaries

    # These next lines threshold above and below to keep a certain color band

    _, low_passed = cv2.threshold(G_uint, 10, low, cv2.THRESH_TOZERO)
    _, band_passed = cv2.threshold(low_passed, high, 255, cv2.THRESH_TOZERO_INV)

    # Here, we apply ridge detection, meaning we are detecting peaks within the image, then finding the middles of those peaks, widthwise
    epsilon = 1e-12

    ridge = sato(band_passed, sigmas=[1, 2, 3], black_ridges=False)
    ridge_norm = (ridge - ridge.min()) / (np.ptp(ridge) + epsilon)
    ridge_filtered = ridge_norm > 0.18

    # Close very small gaps in the ridge image so the Hough transform sees longer continuous lines.
    ridge_filtered = cv2.morphologyEx(
        ridge_filtered.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    ).astype(bool)

    # Now, we limit our analysis to the red zone based on device type
    # Convert barrier_pinch_offs to pixel coordinates
    
    # Pixel index arrays for interpolation
    x_index_arr = np.arange(nx)
    y_index_arr = np.arange(ny)

    # Selects midpoint of x and y axes
    x_idx_mid = np.interp((lb_voltages.min() + lb_voltages.max()) / 2, lb_voltages, x_index_arr)
    y_idx_mid = np.interp((rb_voltages.min() + rb_voltages.max()) / 2, rb_voltages, y_index_arr)
    x_idx_mid = int(np.clip(x_idx_mid, 0, nx - 1))
    y_idx_mid = int(np.clip(y_idx_mid, 0, ny - 1))
    
    ridge_masked = np.zeros_like(ridge_filtered)
    
    if device_type == 'electron':
        # Electron: analyze bottom-left
        ridge_masked[:y_idx_mid, :x_idx_mid] = ridge_filtered[:y_idx_mid, :x_idx_mid]
    else:
        # Hole: analyze top-right
        ridge_masked[y_idx_mid:, x_idx_mid:] = ridge_filtered[y_idx_mid:, x_idx_mid:]

    # From these edges, we detect lines using a probabilistic hough transform.
    # Uses a slightly lower threshold and tunes the minimum required segment length so long lines are prioritized.
    hough_threshold = max(5, int(0.02 * max(nx, ny)))
    hough_length = max(12, int(minLineLength * 0.15))
    hough_gap = max(1, int(maxLineGap * 0.03))

    logger.info("hough lines drawing...")

    lines = transform.probabilistic_hough_line(
        ridge_masked,
        threshold=hough_threshold,
        line_length=hough_length,
        line_gap=hough_gap
    )

    # if not lines:
    #     return []

    # Now, we filter for lines within a certain angle range

    line_candidates = []
    min_length = max(0.10 * max(nx, ny), 15)
    for p0, p1 in lines:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        angle = np.degrees(np.arctan2(dy, dx))
        length = np.hypot(dx, dy)
        if minAngleDeg <= angle <= maxAngleDeg and length >= min_length:
            midx = 0.5 * (p0[0] + p1[0])
            midy = 0.5 * (p0[1] + p1[1])
            if device_type == 'electron':
                score = length - 0.35 * (midx + midy)
            else:
                score = length - 0.35 * ((nx - midx) + (ny - midy))
            line_candidates.append((score, (*p0, *p1)))

    line_candidates.sort(key=lambda item: -item[0])
    filtered_lines = [entry[1] for entry in line_candidates]

    logger.info("filtering...")

    if not filtered_lines:
        # Relax angle range slightly if no good long line was found.
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            length = np.hypot(dx, dy)
            if (minAngleDeg - 10) <= angle <= (maxAngleDeg + 10) and length >= min_length:
                midx = 0.5 * (p0[0] + p1[0])
                midy = 0.5 * (p0[1] + p1[1])
                if device_type == 'electron':
                    score = length - 0.35 * (midx + midy)
                else:
                    score = length - 0.35 * ((nx - midx) + (ny - midy))
                line_candidates.append((score, (*p0, *p1)))
        line_candidates.sort(key=lambda item: -item[0])
        filtered_lines = [entry[1] for entry in line_candidates]

        line_candidates = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            length = np.hypot(dx, dy)
            if (minAngleDeg - 10) <= angle <= (maxAngleDeg + 10) and length >= min_length:
                midx = 0.5 * (p0[0] + p1[0])
                midy = 0.5 * (p0[1] + p1[1])
                if device_type == 'electron':
                    score = length - 0.35 * (midx + midy)
                else:
                    score = length - 0.35 * ((nx - midx) + (ny - midy))
                line_candidates.append((score, (*p0, *p1)))
        line_candidates.sort(key=lambda item: -item[0])
        filtered_lines = [entry[1] for entry in line_candidates]

    # if not filtered_lines:
    #     return []

    # Here we're defining the voltage range for that quadrant
    # Using the pinch-off voltages from barrier_pinch_offs parameter

    lb_mid_volt = (lb_voltages.min() + lb_voltages.max()) / 2  # First value: x-axis (left barrier gate)
    rb_mid_volt = (rb_voltages.min() + rb_voltages.max()) / 2  # Second value: y-axis (right barrier gate)

    perp_candidates = []
    perp_traces_for_plot = []

    perp_length_pixels = max(40, int(min(nx, ny) * 0.5))
    perp_samples = 400
    smooth_sigma = 2.0

    # Now, for each filtered line, we define a line perpendicular to it

    logger.info("finding traces...")

    for x1, y1, x2, y2 in filtered_lines:
        
        # Midpoints
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        # Distances
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
            continue

        # Perpendicular direction
        pxu, pyu = -dy / L, dx / L

        # Length along the perpendicular lines in pixel space
        t = np.linspace(-perp_length_pixels / 2,
                        perp_length_pixels / 2,
                        perp_samples)

        # Limiting the values of the array to the quadrant in analysis
        samp_x = np.clip(mx + pxu * t, 0, nx - 1)
        samp_y = np.clip(my + pyu * t, 0, ny - 1)
        trace_id = len(perp_traces_for_plot)

        # Defining current trace
        trace = map_coordinates(
            current_data,
            [samp_y, samp_x],
            order=3,
            mode="reflect"
        )

        # Smooth the trace
        trace_smooth = gaussian_filter1d(trace, smooth_sigma)
        
        # Calculating conductance
        ds = np.sqrt(np.diff(samp_x)**2 + np.diff(samp_y)**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        conductance = np.gradient(trace_smooth, s)

        trace_info = {
            "px": samp_x.copy(),
            "py": samp_y.copy(),
            "trace_id": trace_id,
            "trace": trace_smooth.copy(),
            "conductance": conductance.copy(),
            "s": s.copy()
        }

        # Find local maxima of current
        noise_sigma = 1.4826 * np.median(np.abs(conductance - np.median(conductance)))
        prominence_thresh = 4.0 * noise_sigma
        peaks, _ = signal.find_peaks(conductance,
                                     prominence=prominence_thresh,
                                     distance=15)

        if len(peaks) == 0:
            continue

        # Defining the the maxima in voltage space from pixel space
        peak_idx = peaks
        px = samp_x[peak_idx]
        py = samp_y[peak_idx]

        vx = np.interp(px, x_index_arr, lb_voltages)
        vy = np.interp(py, y_index_arr, rb_voltages)

        # Restrict to red zone based on device type
        if device_type == 'electron':
            valid = (vx < lb_mid_volt) & (vy < rb_mid_volt)
        else:  # Hole
            valid = (vx > lb_mid_volt) & (vy > rb_mid_volt)
        peak_idx = peak_idx[valid]
        px = px[valid]
        py = py[valid]
        vx = vx[valid]
        vy = vy[valid]

        if len(peak_idx) == 0:
            continue

        # Compiling data into trace info
        for k, p in enumerate(peak_idx):
            prom = signal.peak_prominences(trace_smooth, peak_idx)[0]
            score = prom[k]

            perp_candidates.append(
                (
                    score,
                    vx[k],
                    vy[k],
                    px[k],
                    py[k],
                    trace_id
                )
            )

        trace_info["peak_idx"] = peak_idx.copy()
        trace_info["vx_peak"] = vx
        trace_info["vy_peak"] = vy

        perp_traces_for_plot.append(trace_info)

    # if not perp_candidates:
    #     return []


    # ---------- Selecting Final Bias Points ----------

    logger.info("selecting bias points...")

    logger.info(f"filtered_lines: {len(filtered_lines)}")
    logger.info(f"perp_candidates: {len(perp_candidates)}")

    # First, we sort the points in order of increasing current
    perp_candidates.sort(key=lambda x: -x[0])

    # Then, we pick the top 4 points of highest current
    N_FINAL = 4
    top_candidates = perp_candidates[:N_FINAL]

    selected_trace_ids = {c[5] for c in top_candidates}

    selected_peaks_by_trace = {
        c[5]: (c[1], c[2]) for c in top_candidates
    }

    for tr in perp_traces_for_plot:
        tid = tr["trace_id"]
        if tid in selected_peaks_by_trace:
            tr["peaks"] = [selected_peaks_by_trace[tid]]

    perp_bias_points = [
        (round(vx, 3), round(vy, 3))
        for (_, vx, vy, _, _, _) in top_candidates
    ]

    logger.info("selecting working points...")

    dist_to_pinch_off_corner = {}
    shift_voltage = 0.15 # Amount we want to shift the triple dot working point by in voltage space

    # Compute selected working points. For Triple Dot, shift each point 0.15 V in the
    # yellow trace direction (perpendicular)
    selected_working_points = []
    traces_by_id = {tr["trace_id"]: tr for tr in perp_traces_for_plot}
    for i, cand in enumerate(top_candidates):
        _, vx_c, vy_c, px_c, py_c, tid = cand
        tr = traces_by_id.get(tid, None)
        cand_point = np.array([vx_c, vy_c])
        if tr is not None and str(DotTuning).strip().lower() == 'triple dot':
            try:
                vx_trace = np.interp(tr["px"], x_index_arr, lb_voltages)
                vy_trace = np.interp(tr["py"], y_index_arr, rb_voltages)
                direction = np.array([vx_trace[-1] - vx_trace[0], vy_trace[-1] - vy_trace[0]])
                norm_dir = np.hypot(direction[0], direction[1])
                if norm_dir > 0:
                    unit_dir = direction / norm_dir
                    shift_vec = -unit_dir * shift_voltage
                    selected_working_points.append((float(round(vx_c + shift_vec[0], 3)), float(round(vy_c + shift_vec[1], 3))))
                else:
                    selected_working_points.append((round(vx_c, 3), round(vy_c, 3)))
            except Exception:
                selected_working_points.append((round(vx_c, 3), round(vy_c, 3)))
        else:
            selected_working_points.append((round(vx_c, 3), round(vy_c, 3)))
        
        dist_to_pinch_off_corner[tuple(cand_point)] = np.linalg.norm(cand_point - barrier_pinch_offs)

    logger.info("defining perp traces...")

    perp_traces_for_plot = [
        tr for tr in perp_traces_for_plot
        if tr["trace_id"] in selected_trace_ids
    ]

    # Now, we overlay perpendicular traces (strictly clipped to the quadrant in analysis)
    for tr in perp_traces_for_plot:

        logger.info("for loop!")

        vx = np.interp(tr["px"], x_index_arr, lb_voltages)
        vy = np.interp(tr["py"], y_index_arr, rb_voltages)

        if device_type == 'electron':
            in_quad = (vx < lb_mid_volt) & (vy < rb_mid_volt)
        else:  # hole
            in_quad = (vx > lb_mid_volt) & (vy > rb_mid_volt)
        if not np.any(in_quad):
            continue

        logger.info("finding index!")

        idx = np.where(in_quad)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        blocks = np.split(idx, splits + 1)

        logger.info("getting peak!")

        peak_idx = tr.get("peak_idx", None)
        chosen_block = None
        if peak_idx is not None:
            best_count = -1
            best_block = None
            for b in blocks:
                count = np.intersect1d(peak_idx, b).size
                if count > best_count:
                    best_count = count
                    best_block = b
            if best_count > 0:
                chosen_block = best_block
        if chosen_block is None:
            chosen_block = blocks[0]

        tr["chosen_block"] = chosen_block

    logger.info("finding closest candidate...")

    logger.info(f"top_candidates: {len(top_candidates)}")
    logger.info(f"dist_to_pinch_off_corner: {len(dist_to_pinch_off_corner)}")

    closest_candidate, closest_value = min(dist_to_pinch_off_corner.items(), key=lambda kv: kv[1])

    # ---------- Final Plotting ----------

    logger.info("Plotting Results...")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10,10))

    # Show image
    im = ax.imshow(
        current_data,
        extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
        origin='lower',
        aspect='auto',
        cmap='coolwarm'
    )

    # Save raw data
    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    # Set axis limits
    ax.set_xlim(lb_data.min(), lb_data.max())
    ax.set_ylim(rb_data.min(), rb_data.max())

    # Round and Set ticks
    step = 0.001
    def round_to_step(x, step): return step * np.round(x / step)

    x0, x1 = round_to_step(lb_data.min(), step), round_to_step(lb_data.max(), step)
    y0, y1 = round_to_step(rb_data.min(), step), round_to_step(rb_data.max(), step)

    ax.set_xticks([np.round(lb_data.min(), 3), np.round(lb_data.max(), 3)])
    ax.set_yticks([np.round(rb_data.min(), 3), np.round(rb_data.max(), 3)])
    ax.set_xticklabels([str(np.round(lb_data.min(), 3)), str(np.round(lb_data.max(), 3))], fontsize=30)
    ax.set_yticklabels([str(np.round(rb_data.min(), 3)), str(np.round(rb_data.max(), 3))], fontsize=30)

    ax.tick_params(
        which="major",
        direction="in",
        length=6,
        width=1.2,
        top=True,
        right=True
    )

    ax.minorticks_on()

    ax.xaxis.set_minor_locator(AutoMinorLocator(9))
    ax.yaxis.set_minor_locator(AutoMinorLocator(9))

    # Style minor ticks (no labels by default)
    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=1.0,
        top=True,
        right=True
    )

    # Axis labels
    ax.set_xlabel(rf'V$_{{{gates[0]}}}$ (V)', fontsize=35, labelpad = -25)
    ax.set_ylabel(rf'V$_{{{gates[1]}}}$ (V)', fontsize=35)

    ax.yaxis.set_label_coords(-0.025, 0.40)

    # Create horizontal colorbar above the plot
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(
        ax,
        width="100%",
        height="50%",
        loc="upper center",
        bbox_to_anchor=(0, 1.08, 1, 0.1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("I (nA)", fontsize=35, labelpad=10)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("top")
    cbar_ticks = np.linspace(0, current_data.max(), 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks])
    cbar.ax.tick_params(labelsize=25, direction="in", length=6)

    cbar.ax.minorticks_on()

    cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    # Style minor ticks
    cbar.ax.tick_params(
        which="minor",
        direction="in",
        length=4,
        width=1.0
    )

    # Block Boundary and shaded region based on device type
    if device_type == 'electron':
        # Bottom-left quadrant boundary

        # Top edge of the bottom-left quadrant
        ax.plot(
            [lb_data.min(), lb_mid_volt],   # left edge -> midpoint
            [rb_mid_volt, rb_mid_volt],     # horizontal line at y midpoint
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Right edge of the bottom-left quadrant
        ax.plot(
            [lb_mid_volt, lb_mid_volt],     # vertical line at x midpoint
            [rb_data.min(), rb_mid_volt],   # bottom edge -> midpoint
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Shade only the bottom-left quadrant
        rect = Rectangle(
            (lb_data.min(), rb_data.min()),
            lb_mid_volt - lb_data.min(),
            rb_mid_volt - rb_data.min(),
            facecolor='red',
            alpha=0.2,
            edgecolor=None,
            zorder=2
        )
        ax.add_patch(rect)

    else:  # Hole
        # Top-right quadrant boundary

        # Bottom edge of the top-right quadrant
        ax.plot(
            [lb_mid_volt, lb_data.max()],   # midpoint -> right edge
            [rb_mid_volt, rb_mid_volt],     # horizontal line at y midpoint
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Left edge of the top-right quadrant
        ax.plot(
            [lb_mid_volt, lb_mid_volt],     # vertical line at x midpoint
            [rb_mid_volt, rb_data.max()],   # midpoint -> top edge
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Shade only the top-right quadrant
        rect = Rectangle(
            (lb_mid_volt, rb_mid_volt),
            lb_data.max() - lb_mid_volt,
            rb_data.max() - rb_mid_volt,
            facecolor='red',
            alpha=0.2,
            edgecolor=None,
            zorder=2
        )
        ax.add_patch(rect)

    # Hough lines
    for x1, y1, x2, y2 in filtered_lines:
        # Compute voltage coordinates
        v1x = np.interp(x1, x_index_arr, lb_voltages)
        v1y = np.interp(y1, y_index_arr, rb_voltages)
        v2x = np.interp(x2, x_index_arr, lb_voltages)
        v2y = np.interp(y2, y_index_arr, rb_voltages)
        
        # Uncomment below to see the detected Hough lines
        # ax.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2) 

    # Perpendicular traces and peaks
    dot_tuning_shift = shift_voltage if str(DotTuning).strip().lower() == 'triple dot' else 0.0

    shifted_points = []
    best_shifted_point = None
    green_circle = False

    for tr in perp_traces_for_plot:
        idx = tr.get("chosen_block", None)
        if idx is None or len(idx) == 0: continue
        vx = np.interp(tr["px"], np.arange(nx), lb_voltages)
        vy = np.interp(tr["py"], np.arange(ny), rb_voltages)
        ax.plot(vx[idx], vy[idx], c='yellow', lw=1.5, alpha=0.9)
        valid_peaks = np.intersect1d(tr.get("peak_idx", []), idx)

        if valid_peaks.size > 0:
            # Plot each peak and shift it along the perpendicular normal towards origin
            try:
                for p in np.atleast_1d(valid_peaks):
                    i = int(p)
                    vx_p = float(vx[i])
                    vy_p = float(vy[i])

                    # Compute a local tangent along the yellow trace and shift along it
                    if 1 <= i < (len(vx) - 1):
                        ddx = float(vx[i + 1]) - float(vx[i - 1])
                        ddy = float(vy[i + 1]) - float(vy[i - 1])
                    else:
                        # Fallback to using the chosen block endpoints
                        ddx = float(vx[idx][-1]) - float(vx[idx][0])
                        ddy = float(vy[idx][-1]) - float(vy[idx][0])

                    norm_dir = np.hypot(ddx, ddy)
                    if norm_dir > 0:
                        shift_dir = np.array([ddx / norm_dir, ddy / norm_dir])
                        shift_vec = -shift_dir * dot_tuning_shift
                    else:
                        shift_dir = np.array([0.0, 1.0])
                        shift_vec = shift_dir * dot_tuning_shift

                    sx = vx_p + float(shift_vec[0])
                    sy = vy_p + float(shift_vec[1])

                    # Debug: report computed shift direction and shift vector
                    if debug:
                        try:
                            print(f"DEBUG_SHIFT peak={i} vx={vx_p:.6f} vy={vy_p:.6f} dir=({shift_dir[0]:.6f},{shift_dir[1]:.6f}) shift_vec=({shift_vec[0]:.6f},{shift_vec[1]:.6f})")
                            # Draw a cyan debug line showing the shift direction
                            ax.plot([vx_p, sx], [vy_p, sy], c='cyan', lw=1.5, alpha=0.9, zorder=9)
                        except Exception:
                            pass

                    # Shifted star (exactly dot_tuning_shift along the yellow perp)
                    ax.scatter(sx, sy, s=200, c='white', marker='*', edgecolors='black', zorder=10)
                    shifted_points.append((sx, sy))

                    if vx_p == closest_candidate[0] and vy_p == closest_candidate[1]:
                        best_shifted_point = (sx, sy)
                        green_circle = True

                    if green_circle:
                        # Hollow green circle at original peak position for best candidate
                        ax.scatter(vx_p, vy_p, s=80, c='none', edgecolors='white', linewidths=1.5, zorder=11)
                        green_circle = False
                    else:
                        # Hollow red circle at original peak position for every other peak
                        ax.scatter(vx_p, vy_p, s=80, c='none', edgecolors='red', linewidths=1.5, zorder=11)

                    # Arrow from original to shifted star
                    ax.annotate('', xy=(sx, sy), xytext=(vx_p, vy_p),
                                arrowprops=dict(arrowstyle='->', color='black', lw=1.0), zorder=12)
            except Exception:
                pass

        ax.set_box_aspect(0.775)

        # Save final plot
        filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
        fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

        logger.info("Figure saved!")

        metrics = {'Best Working Point': best_shifted_point}

        payload = {
                    'stage': 'Bootstrapping',
                    'step_name': 'Barrier-Barrier Scan',
                    'figure_object': fig,
                    'results': metrics,
                }
        tuning_bridge.plot_queue.put(payload)

        plt.close(fig)
        # plt.show()

        # These are 1D perpendicular trace plots
        if debug:
            for tr in perp_traces_for_plot:
                s = tr["s"]
                I = tr["trace"]                 # Smoothed current
                dIds = tr["conductance"]        # dI/ds
                peak_idx = tr.get("peak_idx", [])
                chosen_block = tr.get("chosen_block", None)

                fig, axs = plt.subplots(
                    2, 1, figsize=(7, 5), sharex=True
                )

                # We plot the current traces here
                axs[0].plot(s, I, color="black", lw=1.3)
                axs[0].set_ylabel("Current")
                axs[0].set_title(
                    f"Perpendicular trace {tr['trace_id']}"
                )

                # Shade chosen block (if present)
                if chosen_block is not None and len(chosen_block) > 0:
                    axs[0].axvspan(
                        s[chosen_block[0]],
                        s[chosen_block[-1]],
                        color="orange",
                        alpha=0.15,
                        label="Selected block"
                    )

                axs[0].legend(loc="best")

                # Here, we plot the conductance as well
                axs[1].plot(s, dIds, color="tab:blue", lw=1.2)
                axs[1].set_xlabel("Arc length s (pixels)")
                axs[1].set_ylabel("dI/ds")

                # Mark current peaks
                if len(peak_idx) > 0:
                    axs[1].scatter(
                        s[peak_idx],
                        dIds[peak_idx],
                        c="red",
                        s=40,
                        zorder=5,
                        label="Current peaks"
                    )

                plt.tight_layout()
                plt.show()

    # ---------- Debugging Code ----------

    if debug:
        
        # We first plot the original current data
        plt.figure(figsize=(10, 6))
        plt.imshow(current_data,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.xlabel(rf'V$_{{{gates[0]}}}$ (V)', fontsize=45)
        plt.ylabel(rf'V$_{{{gates[1]}}}$ (V)', fontsize=45)
        plt.title("Original Current Data") 
        plt.show()
        
        # Next, we plot the gradient of the data, i.e. the conductance
        plt.figure(figsize=(10, 6))
        plt.imshow(G,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("G (Conductance)") 
        plt.show()

        # Then, we plot G_log normalized to 255, or G_uint
        plt.figure(figsize = (10, 6))
        plt.imshow(G_uint, 
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title(r"$G Normalized$")
        plt.show()

        # Here is the plot of the band-passed G_uint, i.e. after being thresholded
        plt.figure(figsize = (10, 6))
        plt.imshow(band_passed, 
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title(r"$G Normalized and Thresholded$")
        plt.show()

        # Here is a plot of the ridges
        plt.figure(figsize=(10, 6))
        plt.imshow(ridge,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Ridges detected")
        plt.show()

        # Here is a plot of the normalized ridges
        plt.figure(figsize=(10, 6))
        plt.imshow(ridge_norm,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Normalized Ridges Detected")
        plt.show()

        # Here is a plot of the ridges filtered for strength
        plt.figure(figsize=(10, 6))
        plt.imshow(ridge_masked,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.title("Filtered Ridges Detected")
        plt.show()


        # ---------- Hough Line Plotting ----------

        # Now, we'll plot a set of lines from the Hough Transform at each preprocessing stage
        # We start with lines detected from the original data
        plt.figure(figsize = (10, 6))
        plt.imshow(
            current_data,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        current_data,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Original Data")
        plt.show()

        # Now, we detect lines from the gradient
        plt.figure(figsize = (10, 6))
        plt.imshow(
            G,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        G,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Gradient")
        plt.show()

        # Now, we detect lines from the normalized gradient
        plt.figure(figsize = (10, 6))
        plt.imshow(
            G_uint,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        G_uint,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from G Normalized")
        plt.show()

        # Now, we detect lines from the G_log normalized after thresholding
        plt.figure(figsize = (10, 6))
        plt.imshow(
            band_passed,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        band_passed,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from band-passed data")
        plt.show()

        # Now, we detect lines from the ridges
        plt.figure(figsize = (10, 6))
        plt.imshow(
            ridge,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        ridge,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Ridges")
        plt.show()

        # Now, we detect lines from the normalized ridges
        plt.figure(figsize = (10, 6))
        plt.imshow(
            ridge_norm,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        ridge_norm,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Normalized Ridges")
        plt.show()

        # Now, we detect lines from the ridges after filtering
        plt.figure(figsize = (10, 6))
        plt.imshow(
            ridge_filtered,
            extent=[lb_data.min(), lb_data.max(), rb_data.min(), rb_data.max()],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )

        lines = transform.probabilistic_hough_line(
        ridge_filtered,
        threshold=15,
        line_length=max(2, int(minLineLength * 0.1)),
        line_gap=max(1, int(maxLineGap * 0.02))
        )

        if not lines:
            return []

        # Now, we filter for lines within a certain angle range
        filtered_lines = []
        for p0, p1 in lines:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle = np.degrees(np.arctan2(dy, dx))
            if minAngleDeg <= angle <= maxAngleDeg:
                filtered_lines.append((*p0, *p1))

        if not filtered_lines:
            return []

        for x1, y1, x2, y2 in filtered_lines:
            hx = 0.5 * (x1 + x2)
            hy = 0.5 * (y1 + y2)

            # Check if line midpoint is near any selected block peaks
            keep_line = False
            for tr in perp_traces_for_plot:
                idx = np.arange(len(tr["s"]))
                if idx is None or len(idx) == 0:
                    continue
                valid_peaks = np.intersect1d(tr["peak_idx"], idx)
                if len(valid_peaks) > 0:
                    mx = np.mean(tr["px"][valid_peaks])
                    my = np.mean(tr["py"][valid_peaks])
                    if np.hypot(hx - mx, hy - my) < max(perp_length_pixels, 5):
                        keep_line = True
                        break

            if keep_line:
                v1x = np.interp(x1, x_index_arr, lb_voltages)
                v1y = np.interp(y1, y_index_arr, rb_voltages)
                v2x = np.interp(x2, x_index_arr, lb_voltages)
                v2y = np.interp(y2, y_index_arr, rb_voltages)
                plt.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2)
            
        plt.title("Hough Transform Lines from Filtered Ridges")
        plt.show()

    logger.info("Returning...")

    if DotTuning == 'Triple Dot':
        return best_shifted_point, shifted_points, perp_traces_for_plot
    elif DotTuning == 'SET':
        return best_shifted_point, perp_bias_points, perp_traces_for_plot

def extract_tunnel_barrier_latching(dp_data: np.array,
                                    tb_data: np.array,
                                    current_data: np.array,
                                    filepath: str,
                                    filename: str,
                                    peak_height: list[float] = [None, None],
                                    peak_prominence: list[float] = [None, None],
                                    peak_width: list[float] = [None, None]
                                    ):
    """
    Description
    -----------
    Analyze current data to identify if latching is occuring during dot-lead tuning.

    This function plots the derivative of the current at multiple traces and analyzes the peaks to see if latching is occuring.

    Parameters
    -----------
    dp_data : np.array
        independent dot plunger voltage values
    tb_data : np.array
        independent tunnel barrier voltage values
    current_data : np.array
        dependent current values
    peak_height : list[float]
        minimum absolute peak height threshold in conductance, for [positive peaks, negative peaks]
    peak_prominence : list[float]
        minimum peak prominence threshold in conductance, for [positive peaks, negative peaks]
    peak_width : list[float]
        minimum peak width threshold in pixel space, for [positive peaks, negative peaks]

    Returns
    -----------
    best_sens_pts_list : list
        the most positive and negative conductance peaks in current space found at every trace
    all_sens_pts_list : list
        all conductance peaks in current space found at every trace 
    barrier_voltage_set_point : float
        the voltage where the tunnel barrier should be set to to avoid latching
    """

    dp_data = np.array(dp_data)
    tb_data = np.array(tb_data)
    current_data = np.array(current_data)
    device_type = 'electron'
    num_traces = 25

    if np.average(dp_data) < 0 and np.average(tb_data) < 0:
        current_data = np.flip(current_data, axis=None)
        device_type = 'hole'
    
    # Extract unique, sorted coordinate values for axis references
    unique_dp = np.sort(np.unique(dp_data))
    unique_tb = np.sort(np.unique(tb_data))
    
    nx = len(unique_dp)
    ny = len(unique_tb)

    if current_data.ndim == 1:
        if current_data.size == nx * ny:
            current_data = current_data.reshape((ny, nx))
        else:
            raise ValueError("Data size does not form a perfect rectangular grid.")
        
    ny, nx = current_data.shape

    # Calculate evenly spaced indices across the Y axis (tb_data)
    y_indices = np.linspace(0, ny - 1, num_traces, dtype=int)

    # Extract the current rows corresponding to those Y indices
    # Shape is (num_traces, nx)
    sliced_current = current_data[y_indices, :]

    # Duplicate X axis (dp_data) to pair with every extracted trace row
    x_broadcasted = np.repeat(unique_dp[np.newaxis, :], num_traces, axis=0)

    # Stack X and Current along the last axis -> shape (num_traces, nx, 2)
    # Trace 0 matches the lowest unique_tb value, Trace -1 matches the highest.
    final_traces = np.stack((x_broadcasted, sliced_current), axis=-1)

    # =========================================================================
    # PLOTTING BLOCK: Render the derivative subplots inside the function
    # =========================================================================
    cols = 5
    rows = int(np.ceil(num_traces / cols))
    
    # # Adjusted sharey=False because derivative scales can vary across the map
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=False)
    axes = axes.flatten()  # Flatten grid into a 1D list for easy looping

    # Match the exact physical tb_data coordinate value for each indexed row slice
    trace_y_values = unique_tb[y_indices]

    best_sens_pts_list = []
    all_sens_pts_list = []

    for i in range(num_traces):
        # Extract X (dp_data) and Z (current) data for this specific trace
        x_vals = final_traces[i, :, 0]
        z_vals = final_traces[i, :, 1]

        spline = make_smoothing_spline(x_vals, z_vals, lam=1e-9)
        smoothed_z_vals = spline(x_vals)

        deriv = np.gradient(smoothed_z_vals, x_vals)
        neg_deriv = -deriv

        if peak_height == [None, None]:
            peak_height = [0.14 * deriv.max(), 0.3 * deriv.max()]
        if peak_prominence == [None, None]: 
            peak_prominence = [0.1 * deriv.max(), 0.3 * neg_deriv.max()]
        if peak_width == [None, None]:
            peak_width = [(0, 40), (0, None)]

        print(f"Tunnel Barrier Voltage = {trace_y_values[i]:.3f} V")

        best_pts, all_pts = extract_max_conductance_points(x_vals, smoothed_z_vals, peak_height=peak_height, peak_prominence=peak_prominence, peak_width=peak_width)
        best_sens_pts_list.append(best_pts)
        all_sens_pts_list.append(all_pts)

        # 1. Compute the derivative (dI/d_dp) using the spatial coordinate grid spacing
        derivative_vals = np.gradient(z_vals, x_vals)
        
        # 2. Plot the derivative line to the corresponding grid cell
        axes[i].plot(x_vals, derivative_vals, color='tab:purple', linewidth=1.5)
        
        # Place LaTeX formatted text overlay relative to the subplot viewport bounds
        axes[i].text(
            0.05, 0.93, 
            rf"$V_B$ = {trace_y_values[i]:.3f}", 
            transform=axes[i].transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

    # Clean up empty subplots if any
    for j in range(num_traces, len(axes)):
        fig.delaxes(axes[j])

    # Add global canvas labels and updated main title
    fig.supxlabel(r"$V_P$ (V)", fontsize=12)
    fig.supylabel("G (nS)", fontsize=12)
    fig.suptitle("Conductance Traces Extracted Along the Tunnel Barrier", fontsize=14, fontweight='bold')
    
    plt.tight_layout()

    metrics = {'Best Sensitivity Point': best_sens_pts_list}

    payload = {
                'stage': 'Global Charge Tuning',
                'step_name': 'Dot-Lead Tuning',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)
    # plt.show()

    negative_peak_count = 0
    negative_peak_count_list = []

    for arrs in all_sens_pts_list:
        for items in arrs:
            if items < 0:
                negative_peak_count += 1
                break
        negative_peak_count_list.append(negative_peak_count)
    
    if negative_peak_count == 0:
        barrier_voltage_set_point = tb_data.min()
        return best_sens_pts_list, all_sens_pts_list, barrier_voltage_set_point
    if negative_peak_count == num_traces:
        barrier_voltage_set_point = tb_data.max()
        return best_sens_pts_list, all_sens_pts_list, barrier_voltage_set_point

    final_number = negative_peak_count_list[-1]
    first_index = negative_peak_count_list.index(final_number)
    second_index = negative_peak_count_list.index(final_number, first_index + 1)
    barrier_voltage_set_point = trace_y_values[second_index]

    return best_sens_pts_list, all_sens_pts_list, barrier_voltage_set_point

def extract_max_conductance_pair(x_data: np.array,
                                 y_data: np.array,
                                 filepath: str,
                                 filename: str,
                                 peak_height_factor: list[float] = [None, None],
                                 peak_prominence_factor: list[float] = [None, None],
                                 peak_width: list[float] = [None, None]
                                ):
    """
    Description
    -----------
    Analyze current data to identify the largest conductance peak and it's pair feature on the same current peak.

    This function plots the current and its derivative, then highlights
    the most extreme conductance peak along with the pair that's on the same current peak.

    Parameters
    -----------
    x_data : np.array
        independent gate voltage values
    y_data : np.array
        dependent SET current values
    filepath : str
        name of directory to save conductance plot in
    filename : str
        name of file to save the conductance plot under
    peak_height_factor : list[float]
        percentage of maximum conductance to set the minimum absolute peak height threshold in conductance, for [positive peaks, negative peaks]
    peak_prominence_factor : list[float]
        percentage of maximum conductance to set the minimum peak prominence threshold in conductance, for [positive peaks, negative peaks]
    peak_width : list[float]
        minimum peak width threshold in pixel space, for [positive peaks, negative peaks]

    Returns
    -----------
    conductance_pair : tuple(float)
        the voltages of the largest absolute conductance and its pair on the same current peak
    """

    x1 = np.array(x_data)
    y1 = np.array(y_data)

    # Now, we calculate the derivative and replot

    dIdV = np.gradient(y1, x1)

    posdIdV = abs(dIdV)

    if peak_height_factor == [None, None]:
        peak_height = [0.25 * np.max(dIdV), 0.25 * np.max(dIdV)]
    else:
        peak_height = [peak_height_factor[0] * np.max(dIdV), peak_height_factor[1] * np.max(dIdV)]

    if peak_prominence_factor == [None, None]:
        peak_prominence = [0.3 * np.max(dIdV), 0.3 * np.max(dIdV)]
    else:
        peak_prominence = [peak_prominence_factor[0] * np.max(dIdV), peak_prominence_factor[1] * np.max(dIdV)]

    peak_idx_pos, _ = signal.find_peaks(dIdV, height = peak_height[0], prominence = peak_prominence[0], width=peak_width[0])
    peak_idx_neg, _ = signal.find_peaks(-dIdV, height = peak_height[1], prominence = peak_prominence[1], width=peak_width[1])

    peak_idx = np.sort(np.concatenate([peak_idx_pos, peak_idx_neg]))

    x_top = x1[peak_idx]
    I_top = y1[peak_idx]
    G_top = dIdV[peak_idx]

    max_idx = peak_idx[np.argmax(posdIdV[peak_idx])]

    x_max = x1[max_idx]
    I_max = y1[max_idx]
    G_max = dIdV[max_idx]

    if G_max > 0:
        pair_idx = peak_idx[np.where(max_idx == peak_idx)[0][0] + 1]
    else:
        pair_idx = peak_idx[np.where(max_idx == peak_idx)[0][0] - 1]

    x_pair = x1[pair_idx]
    I_pair = y1[pair_idx]
    G_pair = dIdV[pair_idx]

    if G_max > 0:
        conductance_pair = (x_max, x_pair)
    else:
        conductance_pair = (x_pair, x_max)
    
    # Create two subplots that share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6))

    # --- Top panel: Current ---
    ax1.plot(x1, y1, color='#2c5aa0', linewidth=1)

    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight') 

    ax1.scatter(x_max, I_max, facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Max I')
    ax1.scatter(x_pair, I_pair, facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Max I')
    ax1.set_ylabel('I (nA)', fontsize=35)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(min(x1), max(x1))
    ax1.tick_params(labelbottom=True)

    # --- Bottom panel: Conductance ---
    ax2.plot(x1, dIdV, color='#2c5aa0', linewidth=1)
    ax1.scatter(x_max, G_max, facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Max G')
    ax1.scatter(x_pair, G_pair, facecolors='none', edgecolors="#01FF05", s=100, linewidths=2, zorder=5, label='Max G')
    ax2.set_xlabel(r'$V_P$ (V)', fontsize=35)
    ax2.set_ylabel('G (nS)', fontsize=35)
    ax1.set_ylim(bottom=0)
    ax2.set_xlim(min(x1), max(x1))

    # --- Create the connection line ---
    con = ConnectionPatch(
        xyA=(x_max, I_max), coordsA=ax1.transData,
        xyB=(x_max, G_max), coordsB=ax2.transData,
        color='#01FF05', linestyle='--', linewidth=0.7
    )
    fig.add_artist(con)

    con = ConnectionPatch(
        xyA=(x_pair, I_pair), coordsA=ax1.transData,
        xyB=(x_pair, G_pair), coordsB=ax2.transData,
        color='#01FF05', linestyle='--', linewidth=0.7
    )
    fig.add_artist(con)

    # --- Create a custom legend entry (hollow circle) ---
    legend_marker = mlines.Line2D([], [], color='#01FF05', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='Best Conductance Points')

    # --- Custom tick labels: only min and max shown ---

    # Get existing ticks (so tick marks stay)
    for ax in [ax1, ax2]:

        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
        ax.tick_params(direction='in', length=5, width=1.2, labelsize=20, top=True, right=True)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        
    ax1.set_xticks([np.round(x1.min(), 3), np.round((x1.min() + x1.max()) / 2, 3), np.round(x1.max(), 3)])
    ax1.set_xticklabels([str(np.round(x1.min(), 3)), str(np.round((x1.min() + x1.max()) / 2, 3)), str(np.round(x1.max(), 3))], fontsize=25)

    ax1.set_yticks([0, np.round(y1.max(), 3)])
    ax1.set_yticklabels(['0', str(np.round(y1.max(), 3))], fontsize=25)

    ax2.set_xticks([np.round(x1.min(), 3), np.round((x1.min() + x1.max()) / 2, 3),  np.round(x1.max(), 3)])
    ax2.set_xticklabels([str(np.round(x1.min(), 3)), str(np.round((x1.min() + x1.max()) / 2, 3)), str(np.round(x1.max(), 3))], fontsize=25)

    ax2.set_yticks([np.round(dIdV.min(), 1), 0, np.round(dIdV.max(), 1)])
    ax2.set_yticklabels([str(np.round(dIdV.min(), 1)), '0', str(np.round(dIdV.max(), 1))], fontsize=25)

    ax1.legend(handles=[legend_marker],
           loc='upper left',
           fontsize=16,
           frameon=False)

    # --- Adjust layout ---
    plt.subplots_adjust(hspace=0.40)

    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
    fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    metrics = {'Best Conductance Point Pair': conductance_pair}

    payload = {
                'stage': 'Bootstrapping',
                'step_name': 'Coulomb Blockade Peaks with Best Sensitivity Points',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)

    return conductance_pair

def extract_charge_transitions(x_data: np.array,
                               y_data: np.array,
                               filepath: str,
                               filename: str,
                               peak_height_factor: list[float] = [None, None],
                               abs_peak_height: list[float] = [None, None],
                               peak_prominence_factor: list[float] = [None, None],
                               peak_width: list[tuple] = [None, None]
                               ):
    """
    Description
    -----------
    Analyze current data to identify charge transition (as heavy-side looking functions).

    This function plots the current and its derivative, then highlights
    the all charge transitions, returning the best one to the user.

    Parameters
    -----------
    x_data : np.array
        independent gate voltage values
    y_data : np.array
        dependent SET current values
    filepath : str
        name of directory to save charge transitions plot in
    filename : str
        name of file to save the charge transitions plot under
    peak_height_factor : list[float]
        percentage of maximum conductance to set the minimum absolute peak height threshold in conductance, for [positive peaks, negative peaks]
    abs_peak_height : list[float]
        absolute minimum conductance threshold for peak detection
    peak_prominence_factor : list[float]
        percentage of maximum conductance to set the minimum peak prominence threshold in conductance, for [positive peaks, negative peaks]
    peak_width : list[tuple]
        peak width range in pixel space, for [positive peaks, negative peaks]

    Returns
    -----------
    best_charge_transition_voltage : float
        the voltage where the strongest conductance of any charge transition exists
    """

    # --- Data definitions ---
    
    # Ensures numpy array 
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    # Calculate the derivative
    dIdV = np.gradient(y1, x1)

    # Implements default if no user-defined values are given
    if abs_peak_height != [None, None]:
        peak_height = [abs_peak_height[0], abs_peak_height[1]]
    elif peak_height_factor == [None, None]:
        peak_height = [0.25 * np.max(dIdV), 0.25 * np.max(dIdV)]
    else:
        peak_height = [peak_height_factor[0] * np.max(dIdV), peak_height_factor[1] * np.max(dIdV)]

    if peak_prominence_factor == [None, None]:
        peak_prominence = [0.3 * np.max(dIdV), 0.3 * np.max(dIdV)]
    else:
        peak_prominence = [peak_prominence_factor[0] * np.max(dIdV), peak_prominence_factor[1] * np.max(dIdV)]

    # --- Find two largest and two smallest conductance points (positive + negative extremes) ---

    peak_idx_pos, _ = signal.find_peaks(dIdV, height = peak_height[0], prominence = peak_prominence[0], width=peak_width[0])
    peak_idx_neg, _ = signal.find_peaks(-dIdV, height = peak_height[1], prominence = peak_prominence[1], width=peak_width[1])

    peak_idx = np.sort(np.concatenate([peak_idx_pos, peak_idx_neg]))

    # Check if no peaks are found, and safely end the function whilst saving the raw data

    if peak_idx.size == 0:

        # logger.info("No conductance peaks were found")

        # Saves data even if no peaks are found, so the user can see why the peak detection failed
        fig = plt.figure()
        plt.plot(x1, y1)

        filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
        fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

        plt.close(fig)

        raise ValueError("No conductance peaks were found") # Switch to Logging error so protocol doesn't crash

    # Extract data points from peaks

    x_top = x1[peak_idx]
    I_top = y1[peak_idx]
    G_top = dIdV[peak_idx]

    max_idx = peak_idx[np.argmax(dIdV[peak_idx])]
    min_idx = peak_idx[np.argmin(dIdV[peak_idx])]

    x_max = x1[max_idx]
    x_min = x1[min_idx]
    I_max = y1[max_idx]
    I_min = y1[min_idx]
    G_max = dIdV[max_idx]
    G_min = dIdV[min_idx]

    num_transitions = 3
    if len(peak_idx) < 3:
        num_transitions = len(peak_idx)
    G_best = []
    x_best = []
    I_best = []
    indices = np.argpartition(abs(G_top), -num_transitions)[-num_transitions:]
    for i in indices:
        x_best.append(x_top[i])
        I_best.append(I_top[i])
        G_best.append(G_top[i])

    all_sens_pts = (x_top, G_top)

    # Create two subplots that share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6))

    # --- Top panel: Current Data ---
    ax1.plot(x1, y1, color='#2c5aa0', linewidth=1)

    # --- Bottom panel: Conductance Data ---
    ax2.plot(x1, dIdV, color='#2c5aa0', linewidth=1)

    # Save raw data
    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    # --- Top panel: Current Analysis and Labelling ---
    for i in range(len(x_best)):
        ax1.scatter(x_best[i], I_best[i], facecolors='none', edgecolors="#FF5500", s=100, linewidths=2, zorder=5, label='Detected Charge Transitions')
    ax1.set_ylabel('I (nA)', fontsize=35)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(min(x1), max(x1))
    ax1.tick_params(labelbottom=True)

    # --- Bottom panel: Conductance Analysis and Labelling ---
    for i in range(len(x_best)):
        ax2.scatter(x_best[i], G_best[i], facecolors='none', edgecolors="#FF5500", s=100, linewidths=2, zorder=5, label='Detected Charge Transitions')
    ax2.set_xlabel(r'$V_P$ (V)', fontsize=35)
    ax2.set_ylabel('G (nS)', fontsize=35)
    ax1.set_ylim(bottom=0)
    ax2.set_xlim(min(x1), max(x1))

    # --- Create the connection line between top and bottom panels ---
    for i in range(len(x_best)):
        con = ConnectionPatch(
            xyA=(x_best[i], I_best[i]), coordsA=ax1.transData,
            xyB=(x_best[i], G_best[i]), coordsB=ax2.transData,
            color='#FF5500', linestyle='--', linewidth=0.7
        )
        fig.add_artist(con)

    # --- Create custom legend entries (hollow circles) ---
    legend_marker = mlines.Line2D([], [], color='#FF5500', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='Detected Charge Transitions')

    # --- Custom tick labels: only min and max shown ---

    # Get existing ticks
    for ax in [ax1, ax2]:

        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
        ax.tick_params(direction='in', length=5, width=1.2, labelsize=20, top=True, right=True)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        
    ax1.set_xticks([np.round(x1.min(), 3), np.round((x1.min() + x1.max()) / 2, 3), np.round(x1.max(), 3)])
    ax1.set_xticklabels([str(np.round(x1.min(), 3)), str(np.round((x1.min() + x1.max()) / 2, 3)), str(np.round(x1.max(), 3))], fontsize=25)

    ax1.set_yticks([0, np.round(y1.max(), 3)])
    ax1.set_yticklabels(['0', str(np.round(y1.max(), 3))], fontsize=25)

    ax2.set_xticks([np.round(x1.min(), 3), np.round((x1.min() + x1.max()) / 2, 3),  np.round(x1.max(), 3)])
    ax2.set_xticklabels([str(np.round(x1.min(), 3)), str(np.round((x1.min() + x1.max()) / 2, 3)), str(np.round(x1.max(), 3))], fontsize=25)

    ax2.set_yticks([np.round(dIdV.min(), 1), 0, np.round(dIdV.max(), 1)])
    ax2.set_yticklabels([str(np.round(dIdV.min(), 1)), '0', str(np.round(dIdV.max(), 1))], fontsize=25)

    ax1.legend(handles=[legend_marker], loc='upper left', fontsize=16, frameon=False)

    # --- Adjust layout ---
    plt.subplots_adjust(hspace=0.40)
    # plt.show()
    
    # Saves final plot
    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
    fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    metrics = {'Charge Transition Voltage (V)': best_charge_transition_voltage}

    payload = {
                'stage': 'Global Charge Tuning',
                'step_name': 'Charge Transitions',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)
    # plt.show()

    G_max = abs(all_sens_pts[1]).max()
    max_idx = np.where(G_max == abs(all_sens_pts[1]))[0][0]
    best_charge_transition_voltage = all_sens_pts[0][max_idx]

    return best_charge_transition_voltage

def extract_lever_arms(x_data: np.array,
                       y_data: np.array,
                       current_data: np.array,
                       filepath: str,
                       filename: str,
                       gate_names: tuple,
                       stage: str,
                       transform_trim: list[int] = [0, -1]
                       ):
    """
    Description
    -----------
    Analyze voltage data to find the slope of the line seen by cross-talk measurements using the hough transform.
    Can be used as regualr hough transform as well.

    This function plots the voltage, then highlights the slope of the cross-talk line seen.
    The transform_trim argument allows you to remove points from the hough tranform analysis to make the fit better (mainly for debugging)

    Parameters
    -----------
    x_data : np.array
        independent gate voltage values
    y_data : np.array
        independent gate voltage values
    current_data : np.array
        dependent current values
    filepath : str
        name of directory to save hough transform plot in
    filename : str
        name of file to save the hough transform plot under
    gate_names: tuple(str)
        name of the 2 gates being swept as strings
    stage: str
        stage in protocol where this function is being called from
    transform_trim : list[int]
        the cut-off values for making the transform a better fit to the data (mainly for manual debugging)

    Returns
    -----------
    slope : float
        the slope of the line found using the transform
    intercept : float
        the y-intercept of the line found using the transform
    """

    # Ensure arguments are arrays
    x1 = np.array(x_data)
    y1 = np.array(y_data)
    I1 = np.array(current_data)

    unique_x = np.unique(x1)
    unique_y = np.unique(y1)

    # Get x voltage spacing
    dx = x1[1] - x1[0]

    trace_len = len(unique_x)   # Number of columns (x1)
    num_traces = len(unique_y)  # Number of rows (y1)

    X_matrix, Y_matrix = np.meshgrid(unique_x, unique_y) # Get the X and Y grids to be used for analysis and plotting
    Z_matrix = np.full((num_traces, trace_len), np.nan) # Create an empty Z matrix
    # Fill in the Z matrix with current data for proper plotting
    for x, y, I in zip(x1, y1, I1):
        idx_x = np.where(unique_x == x)
        idx_y = np.where(unique_y == y)
        Z_matrix[idx_y, idx_x] = I
    x_vals = []
    y_vals = []

    # Plot the 2D color map and save the raw data prior to any analysis

    fig, ax = plt.subplots(figsize=(8,6))

    mesh = ax.pcolormesh(
        unique_x,
        unique_y,
        Z_matrix,
        shading='auto',
        cmap='viridis'
    )
    fig.colorbar(mesh, ax=ax, label="I (nA)")
    ax.set_xlabel("P1 (V)")
    ax.set_ylabel("P2 (V)")

    # Save Raw Data
    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
    fig.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    # Analysis using hough transform line detection

    for j in range(num_traces):
        # Go through each trace along the y-axis
        trace_z = Z_matrix[j, :]
        z_der = signal.savgol_filter(trace_z, window_length=11, polyorder=3, deriv=1, delta=dx)
        der_min = np.min(z_der)
        # if avg == True:
        der_max = np.max(z_der)
        idx1 = np.where(z_der == der_min)[0][0]
        idx2 = np.where(z_der == der_max)[0][0]
        xpt1 = float(X_matrix[j, idx1])
        xpt2 = float(X_matrix[j, idx2])
        ypt1 = float(Y_matrix[j, idx1])
        ypt2 = float(Y_matrix[j, idx2])
        x_vals.append(np.mean([xpt1, xpt2]))
        y_vals.append(np.mean([ypt1, ypt2]))
        # else:
        #     idx = np.where(z_der == der_min)[0][0]
        #     x_vals.append(float(X_matrix[j, idx]))
        #     y_vals.append(float(Y_matrix[j, idx]))

    if len(x_vals) != len(y_vals):
        raise ValueError("lengths of x and y values don't match")
    else:
        start = transform_trim[0]
        end = transform_trim[1]
        x_vals = x_vals[start:end]
        y_vals = y_vals[start:end]
        slope, intercept = np.polyfit(x_vals, y_vals, 1)

    ax.plot(x_vals, np.polyval([slope, intercept], x_vals), color='r', linestyle='-')

    # Set the limits and show
    ax.set_ylim(min(y_vals), max(y_vals))

    # Save Analyzed Data
    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
    fig.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    metrics = {'Slope': slope, 'Y-Intercept': intercept}

    payload = {
                'stage': stage,
                'step_name': f'{gate_names[0]}-{gate_names[1]} Cross talk',
                'figure_object': fig,
                'results': metrics,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig)
    # plt.show()

    return slope, intercept

def bias_range_coulomb_diamond(x_data: np.array,
                               current_data: np.array,
                               bias: list[float],
                               debug: bool = False
                               ):
    """
    Description
    -----------
    Determine the range, bounds, of SD bias necessary to conduct a coulomb diamond experiment.

    This function uses 4 traces given by the user to outline rough diamonds that will be seen for a full high-res scan to be taken later.
    Ensure the order of the current traces and bias is the same.

    Parameters
    -----------
    x_data : np.array
        Charge sensor plunger gate data, all traces should have the same x range
    current_data : np.array
        4 arrays of charge sensor current data are contained within this array, 1 for each bias traces
    bias : list[float]
        The 4 biases at which the traces are taken at
    debug : bool
        Shows the traces and its derivative alongside the where the peaks were found

    Returns
    -----------
    bias_bounds : tuple(float)
        The minimum and maximum bounds for the SD bias for a high-res scan (symmetric around 0)
    line_list : list[list[float]]
        Contains the properties of each line that outlines the coulomb diamonds
    dia_list : list[list[float]]
        Contains the properties of the points that make up each coulomb diamond
    """

    # Check if 4 current_data traces were given
    if len(current_data) != 4:
        raise ValueError("Not enough or too many current traces were given. Ensure that only 4 traces are given.")

    # Check if 4 biases were given
    if len(bias) != 4:
        raise ValueError("Not enough or too many biases were given. Ensure that only 4 biases are given.")

    # Set all data to numpy arrays
    curr = []

    for g in range(len(current_data)):
        curr.append(np.array(current_data[g]))

    x1 = np.array(x_data)
    curr = np.array(curr)

    # Create a list template for containing all the data for the diamonds
    dia_temp = [
                [(), ()], # top left line
                [(), ()], # top right line
                [(), ()], # bottom left line
                [(), ()]  # bottom right line
                        ]

    # Initialze all variables
    dia_list = []
    x_vals = []
    y_vals = []
    dia_create = True
    num_dia = None
    der_analysis = []
    peak_list = []

    # Getting the derivative of each bias trace and finding the peaks in each derivative trace
    for h in range(len(bias)):
        der_analysis.append(np.gradient(curr[h], x1))
        peak_idx, _ = signal.find_peaks(abs(der_analysis[h]), distance=6)
        peak_list.append(peak_idx)

    # Ensures that there are only an even number of peaks (only full diamonds being taken into account)
    for i in range(len(bias)):
        if len(peak_list[i]) % 2 == 1:
            peak_list[i] = peak_list[i][:-1]

    # Group the traces by bias polarity. Every positive-bias trace shares one
    # peak set and every negative-bias trace shares another, so a check that
    # fails on any one trace trims the peaks of every trace in that group.
    pos_group = [idx for idx in range(len(bias)) if bias[idx] > 0]
    neg_group = [idx for idx in range(len(bias)) if bias[idx] <= 0]

    def _peak_sign(trace_idx, peak_pos):
        """Sign of the derivative at a single peak, indexed directly.

        Avoids rebuilding the full ``der_analysis[t][peak_list[t]]`` fancy-index
        array (which the previous implementation did once per comparison).
        """
        return float(np.sign(der_analysis[trace_idx][peak_list[trace_idx][peak_pos]]))

    def _trim_group(group, del_pos):
        """Delete the given peak position(s) from every trace in the group.

        Traces in a group can hold different numbers of peaks, so a member that
        has already run out of peaks is skipped instead of being allowed to raise
        IndexError from ``np.delete``.
        """
        needed = len(del_pos) if isinstance(del_pos, list) else 1
        for member in group:
            if len(peak_list[member]) >= needed:
                peak_list[member] = np.delete(peak_list[member], del_pos)

    # Checks every bias trace for full diamonds. `polarity` encodes the expected
    # leading edge: +1 for positive-bias traces (positive peak then negative)
    # and -1 for negative-bias traces (negative peak then positive).
    for group, polarity in ((pos_group, 1.0), (neg_group, -1.0)):
        for trace_idx in group:

            # Every check reads the first/last two peaks, so stop once the trace
            # no longer has two peaks left to inspect.
            if len(peak_list[trace_idx]) < 2:
                continue

            # First two peaks share a sign, so they do not bound a full diamond
            if _peak_sign(trace_idx, 0) == _peak_sign(trace_idx, 1):
                _trim_group(group, [0, 1])
                if len(peak_list[trace_idx]) < 2:
                    continue

            # Last two peaks share a sign, so they do not bound a full diamond
            if _peak_sign(trace_idx, -1) == _peak_sign(trace_idx, -2):
                _trim_group(group, [-1, -2])
                if len(peak_list[trace_idx]) < 2:
                    continue

            # Leading edge runs the wrong way round for this bias polarity
            if _peak_sign(trace_idx, 0) == -polarity and _peak_sign(trace_idx, 1) == polarity:
                _trim_group(group, 0)
                if len(peak_list[trace_idx]) < 2:
                    continue

            # Trailing edge runs the wrong way round for this bias polarity
            if _peak_sign(trace_idx, -1) == polarity and _peak_sign(trace_idx, -2) == -polarity:
                _trim_group(group, -1)

    # Number of full diamonds found in all traces
    num_dia = int(np.floor(len(peak_list[0])/2))

    for j in range(len(bias)):

        # Creates a copy of the template for each diamond only on the first iteration
        if dia_create:
            for _ in range(num_dia):
                dia_list.append(copy.deepcopy(dia_temp))
            dia_create = False

        # Populates the list with the points to outline each diamond
        for k in range(num_dia):
            start_idx = k*2
            end_idx = start_idx + 1

            if j == 0 or j == 2:
                dia_list[k][j][0] = (x1[peak_list[j][start_idx]], bias[j])
                x_vals.append(x1[peak_list[j][start_idx]])
                y_vals.append(bias[j])
                dia_list[k][j+1][0] = (x1[peak_list[j][end_idx]], bias[j])
                x_vals.append(x1[peak_list[j][end_idx]])
                y_vals.append(bias[j])
            elif j == 1 or j == 3:
                dia_list[k][j-1][1] = (x1[peak_list[j][start_idx]], bias[j])
                x_vals.append(x1[peak_list[j][start_idx]])
                y_vals.append(bias[j])
                dia_list[k][j][1] = (x1[peak_list[j][end_idx]], bias[j])
                x_vals.append(x1[peak_list[j][end_idx]])
                y_vals.append(bias[j])

        # Allows the plotting of the traces and its derivative and where the peaks were found
        if debug == True:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(x1, curr[j], label=f"SD Bias = {bias[j]} V")
            for l in peak_list[j]:
                    ax.axvline(x=x1[l], color='red', linestyle='--')
            ax.set_xlabel("P20 (V)")
            ax.set_ylabel("I (nA)")
            ax.legend()
            plt.show()

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(x1, der_analysis[j], label=f"SD Bias = {bias[j]} V")
            ax.scatter(x1[peak_list[j]], der_analysis[j][peak_list[j]], color='red', label='Peaks')
            ax.set_xlabel("P20 (V)")
            ax.set_ylabel("dI/dP20 (nA/V)")
            ax.legend()
            plt.show()

    line_list = []

    # Finding the lines to outline each diamond
    for m in range(num_dia):
        line_list.append([])

        for n in range(4):
            # If the slope is undefined (vertical line), then slope is set to nan and y_int is the x value that satisfies the equation y = x
            if (dia_list[m][n][1][0] - dia_list[m][n][0][0]) == 0:
                slope = np.nan
                y_int = dia_list[m][n][0][0]
            else:
                slope = (dia_list[m][n][1][1] - dia_list[m][n][0][1]) / (dia_list[m][n][1][0] - dia_list[m][n][0][0])
                y_int = dia_list[m][n][0][1] - (slope*dia_list[m][n][0][0])

            line_list[m].append((slope, y_int))

    # Finding the intersection point between the top and bottom lines
    intersect_list = []

    # Top intersection point (+'ve bias)
    for o in range(num_dia):
        intersect_list.append([])

        # Checks if both lines are vertical, and if so, then set the intersection point to nan
        if np.isnan(line_list[o][1][0]) and np.isnan(line_list[o][0][0]):
            intersect_list[o].append(np.nan)
            continue

        # Checks if the slopes if they are inverted (left line is negatively sloped and right line is positively sloped), and if so, then set the intersection point to nan
        if line_list[o][0][0] < 0 and line_list[o][1][0] > 0:
            intersect_list[o].append(np.nan)
            continue

        # Checks if the lines are parallel, and if so, then set the intersection point to nan
        if line_list[o][0][0] == line_list[o][1][0]:
            intersect_list[o].append(np.nan)
            continue

        # Checks if the right line is vertical and calculates the intersection point accordingly
        if np.isnan(line_list[o][1][0]):
            x_int_0 = line_list[o][1][1]
            y_int_0 = (line_list[o][0][0]*x_int_0) + line_list[o][0][1]

        # Checks if the left line is vertical and calculates the intersection point accordingly
        elif np.isnan(line_list[o][0][0]):
            x_int_0 = line_list[o][0][1]
            y_int_0 = (line_list[o][1][0]*x_int_0) + line_list[o][1][1]

        # Calculates the intersection point
        else:
            x_int_0 = (line_list[o][1][1] - line_list[o][0][1]) / (line_list[o][0][0] - line_list[o][1][0])
            y_int_0 = (line_list[o][0][0]*x_int_0) + line_list[o][0][1]

        intersect_list[o].append((x_int_0, y_int_0))

    # Bottom intersection point (-'ve bias)
    for p in range(num_dia):

        # Checks if both lines are vertical, and if so, then set the intersection point to nan
        if np.isnan(line_list[p][3][0]) and np.isnan(line_list[p][2][0]):
            intersect_list[p].append(np.nan)
            continue

        # Checks if the slopes if they are inverted (left line is positively sloped and right line is negatively sloped), and if so, then set the intersection point to nan
        if line_list[p][2][0] > 0 and line_list[p][3][0] < 0:
            intersect_list[p].append(np.nan)
            continue

        # Checks if the lines are parallel, and if so, then set the intersection point to nan
        if line_list[p][2][0] == line_list[p][3][0]:
            intersect_list[p].append(np.nan)
            continue

         # Checks if the right line is vertical and calculates the intersection point accordingly
        if np.isnan(line_list[p][3][0]):
            x_int_1 = line_list[p][3][1]
            y_int_1 = (line_list[p][2][0]*x_int_1) + line_list[p][2][1]

        # Checks if the left line is vertical and calculates the intersection point accordingly
        elif np.isnan(line_list[p][2][0]):
            x_int_1 = line_list[p][2][1]
            y_int_1 = (line_list[p][3][0]*x_int_1) + line_list[p][3][1]

        # Calculates the intersection point
        else:
            x_int_1 = (line_list[p][3][1] - line_list[p][2][1]) / (line_list[p][2][0] - line_list[p][3][0])
            y_int_1 = (line_list[p][2][0]*x_int_1) + line_list[p][2][1]

        intersect_list[p].append((x_int_1, y_int_1))

    y_int_pos_vals = []
    y_int_neg_vals = []

    # Sorts the y values of all intersection points into positive and negative categories
    for q in range(2):
        for r in range(num_dia):
            item = intersect_list[r][q]

            if isinstance(item, tuple):
                value_to_append = item[1]
            else:
                value_to_append = np.nan 

            if q == 0:
                y_int_pos_vals.append(value_to_append)
            elif q == 1:
                y_int_neg_vals.append(value_to_append)

    # Removes all nans from the arrays
    y_int_pos_vals_upd = [x for x in y_int_pos_vals if not (isinstance(x, float) and np.isnan(x))]
    y_int_neg_vals_upd = [x for x in y_int_neg_vals if not (isinstance(x, float) and np.isnan(x))]

    # Takes the average of the positive and negative intersection values
    y_int_pos_avg = np.average(np.array(y_int_pos_vals_upd))*1.5
    y_int_neg_avg = np.average(np.array(y_int_neg_vals_upd))*1.5

    # Finds which average is larger in magnitude and sets that symetrically across 0 for bounds
    if abs(y_int_pos_avg) > abs(y_int_neg_avg):
            bias_bounds = (-y_int_pos_avg, y_int_pos_avg)
    else:
            bias_bounds = (y_int_neg_avg, abs(y_int_neg_avg))

    return bias_bounds

def extract_coulomb_diamonds(x_data: np.array,
                                y_data: np.array,
                                current_data: np.array,
                                filepath: str,
                                filename: str,
                                smoothing: float = 1.5,
                                edge_threshold: float = 0.5,
                                contrast_percentile: float = 95.0,
                                peak_distance: int = 4,
                                zero_bias_exclusion: float = 0.02,
                                bias_window: float = 0.5,
                                x_window: tuple = None,
                                slope_range: tuple = None,
                                slope_steps: int = 900,
                                degeneracy_prominence: float = 0.25,
                                slope_tolerance: float = 0.5,
                                min_edge_points: int = 4,
                                slope_consistency: float = 0.35,
                                blockade_ratio: float = 1.8,
                                min_concentration: float = 1.5,
                                orientation_margin: float = 0.03,
                                alignment_gain: float = 0.20,
                                debug: bool = False
                                ):
    """
    Description
    -----------
    Locate the Coulomb diamonds in a 2D bias-vs-gate scan and outline them.

    It works from the numerical derivative of the current with respect to the gate axis,
    ``dI/dVg``. Because the ohmic/leakage background of such a scan depends
    almost entirely on the bias, differentiating along the gate axis removes it
    and leaves the diamond edges as ridges.

    Each diamond is a quadrilateral pinned to its two zero-bias vertices, with all
    four edges fitted independently to the ridge points that belong to them. The
    edges are not forced parallel, because forcing that makes the outline miss the
    real diamond whenever the two sides differ even slightly. They are instead only
    loosely tied together: each edge slope is held within ``slope_consistency`` of
    the corresponding global value, so the four stay similar without being equal.

    The global pair is refined iteratively. A first estimate comes from the whole
    scan, the per-edge fits are then re-derived from it, and the global pair is
    recomputed from those fits until it settles. Without this the outlines stay
    locked to a poor starting estimate.

    Algorithm
    -----------
    1. Grid the scan onto its unique gate/bias axes and lightly smooth it.
    2. Take ``dI/dVg`` and normalise the magnitude row by row, so weak-signal
       bias rows still contribute.
    3. Pick out ridge points as the per-row peaks of that edge map, skipping the
       narrow band around zero bias where all edges converge.
    4. Recover the two edge slopes with a Radon-style accumulator: for a trial
       slope ``m`` every ridge point is projected to the zero-bias intercept
       ``b = x - y/m``. The correct slope is the one that makes those intercepts
       pile up, because every diamond edge extrapolates to a charge-degeneracy
       point on the zero-bias axis. The concentration score is normalised
       against a uniform spread so the search is not biased toward steep slopes.
    5. Read the charge-degeneracy points off the combined accumulator. Each
       neighbouring pair of degeneracy points brackets one diamond.
    6. Fit each of the four edges of each diamond to the ridge points nearest its
       predicted position, iterating the global pair until it settles, then
       intersect the upper pair for the apex and the lower pair for the base.
    7. Discard any candidate that runs off the sides of the scan, is taller than
       the measured bias range, or fails a Coulomb-blockade check on its interior
       (inside a real diamond the current is flat, so its gate derivative is
       quiet compared with the rest of the scan).

    Parameters
    -----------
    x_data : np.array
        Gate (plunger) voltage. Either the flat per-point list, or the 1D axis
        when ``current_data`` is supplied as a 2D grid.
    y_data : np.array
        Source-drain bias voltage, in the same layout as ``x_data``.
    current_data : np.array
        Measured current. Either the flat per-point list matching ``x_data`` and
        ``y_data``, or a 2D grid shaped ``(len(y_data), len(x_data))``.
    filepath : str
        name of directory to save coulomb diamonds plot in
    filename : str
        name of file to save the coulomb diamonds plot under
    smoothing : float
        Standard deviation, in pixels, of the Gaussian applied before
        differentiating. Raise it for noisier scans, lower it to resolve
        closely spaced diamonds.
    edge_threshold : float
        Minimum row-normalised edge strength for a ridge point, where 1.0 is the
        ``contrast_percentile`` of the edge map in that bias row.
    contrast_percentile : float
        Percentile of each bias row's edge map that is treated as full scale, the
        algorithmic equivalent of narrowing a colour-bar range. The default of 95
        keys the scale to the strongest features in the row. Lowering it (75, or
        60) compresses the bright features and lets faint edges compete, which
        matters when the diamond signal is only a per-cent-level modulation on a
        large ohmic background. Note that it can change which line family the
        slope search settles on, so check the result against the plots.
    peak_distance : int
        Minimum separation, in gate pixels, between two ridge points in the same
        bias row.
    zero_bias_exclusion : float
        Fraction of the bias half-range around zero bias that is ignored. The
        four edges meeting at a degeneracy point cannot be resolved there.
    bias_window : float
        Fraction of the bias half-range searched for edges, measured out from
        zero bias. Diamonds occupy only the low-bias part of a wide scan, so the
        default keeps the outer half of the scan from adding noise.
    x_window : tuple(float, float) or None
        Optional ``(x_min, x_max)`` restriction of the gate range. Useful for
        excluding a turn-on region or another strong non-diamond feature. When
        None the full gate range is used.
    slope_range : tuple(float, float) or None
        Smallest and largest edge-slope magnitude ``|dV/dVg|`` considered by the
        slope search. When None both bounds are derived from the scan itself using
        the observed zero-bias peak spacing: the lower bound is the shallowest
        slope whose apex still clears the zero-bias axis by a few bias pixels, and
        the upper bound keeps the apex inside the searched bias window. Leaving
        this at None is recommended.
    slope_steps : int
        Number of trial slopes tested per sign. Higher is finer but slower.
    degeneracy_prominence : float
        A peak in the degeneracy accumulator must reach this fraction of the
        tallest peak to be accepted as a charge-degeneracy point.
    slope_tolerance : float
        Fractional band around the global slope within which a ridge point is
        accepted as belonging to a given diamond edge. 0.5 means +/-50%.
    min_edge_points : int
        Fewest ridge points needed to refine a slope for one diamond. Below this
        the global slope is used for that diamond instead.
    slope_consistency : float
        How far a single diamond's fitted slope may depart from the global value,
        as a fraction. Every diamond in a scan is produced by the same charging
        physics, so their edge slopes should be near a common pair; 0.25 allows
        +/-25% of genuine variation while stopping one noisy fit from tilting a
        shape away from its neighbours. Set to 0 to force every diamond onto the
        global pair, or raise it to let each diamond be fitted freely.
    blockade_ratio : float
        A candidate diamond is kept only if the mean edge strength in its
        interior is below this multiple of the mean edge strength over the
        searched region. Inside a diamond the dot is Coulomb blockaded, so the
        current is flat and its gate derivative small. This is a guard against
        spurious shapes rather than a fit criterion, so the default is deliberately
        permissive; lower it to demand a cleaner blockade region, raise it if
        diamonds you can see are being dropped.
    min_concentration : float
        How much more tightly the winning edge family must concentrate its
        zero-bias intercepts than an even spread would, before the scan is
        accepted as containing diamonds at all. 1.0 is "no better than random".
        This is what makes a featureless or pure-noise scan raise instead of
        returning meaningless shapes.
    orientation_margin : float
        Fractional improvement in blockade depth required before the two slope
        magnitudes are swapped between the edge pairs. The swap mirrors every
        diamond (apex moving from one side to the other), so this guards against
        flipping them on a difference that is really just noise. Raise it to make
        the choice stickier, set it to 0 to always take the better of the two.
    alignment_gain : float
        Fractional reduction in the conductance enclosed by the outlines that a
        rigid shift of the whole degeneracy set must achieve before it is applied.
        The ridge accumulator can place the degeneracy points a fraction of a period
        off, which slides every outline off its diamond; shifting to the blockade
        fixes that. The threshold is high on purpose, because a small gain is not
        evidence of a real phase error and acting on it moves correctly placed
        outlines. Set to 0 to always take the best shift, or to 1 to disable.
    debug : bool
        Also plot the derivative map with the detected ridge points and the
        degeneracy accumulator, and print the per-diamond table.

    Returns
    -----------
    diamond_slopes : dict
        One entry per accepted diamond, numbered left to right along the gate
        axis, each holding the slope (dV/dVg) of its four edges::

            {
              'diamond_1': {'top_left': 2.22,  'top_right': -3.89,
                            'bottom_left': -3.61, 'bottom_right': 2.54},
              'diamond_2': {...},
            }

        All four are fitted separately, so they differ from one another. They are
        constrained only to stay within ``slope_consistency`` of the global pair,
        which keeps the two positive edges similar to each other and likewise the
        two negative ones.
    diamond_properties : dict
        Scan-wide averages taken over the accepted diamonds::

            {
              'average_positive_slope': 2.28,     # dV/dVg, mean of the two
                                                  #   positive edges of every diamond
                                                  #   (top-left and bottom-right)
              'average_negative_slope': -3.51,    # mean of the two negative edges
                                                  #   (top-right and bottom-left)
              'average_diamond_height': 0.1067,   # apex-to-base separation, in volts
                                                  #   of SD bias
              'average_diamond_width': 0.0393,    # gate-voltage separation of the two
                                                  #   charge-degeneracy points that
                                                  #   bracket the diamond
            }

    Raises
    -----------
    ValueError
        If the scan cannot be gridded, if the slope search finds no consistent
        edge family, or if no complete (not cut off) diamond is found.
    """

    # --- inputs ---
    # Coerce to numpy arrays even when they already are
    x1 = np.array(x_data, dtype=float)
    y1 = np.array(y_data, dtype=float)
    curr = np.array(current_data, dtype=float)

    # Accept either a 2D grid on 1D axes, or flat per-point lists
    if curr.ndim == 2 and x1.ndim == 1 and y1.ndim == 1 and curr.shape == (y1.size, x1.size):
        unique_x = np.unique(x1)
        unique_y = np.unique(y1)
        Z_matrix = curr.copy()
    else:
        xf, yf, cf = x1.ravel(), y1.ravel(), curr.ravel()
        if not (xf.size == yf.size == cf.size):
            raise ValueError("x_data, y_data and current_data must describe the same "
                             "number of points, or current_data must be a 2D grid "
                             "shaped (len(y_data), len(x_data)).")
        unique_x = np.unique(xf)
        unique_y = np.unique(yf)
        Z_matrix = np.full((unique_y.size, unique_x.size), np.nan)
        Z_matrix[np.searchsorted(unique_y, yf), np.searchsorted(unique_x, xf)] = cf

    if unique_x.size < 8 or unique_y.size < 8:
        raise ValueError(f"Scan is too small to contain diamonds: got a "
                         f"{unique_y.size} x {unique_x.size} grid.")

    fig_raw, ax_raw = plt.subplots(figsize=(8, 6))
    mesh = ax_raw.pcolormesh(unique_x, unique_y, Z_matrix, shading='auto', cmap='viridis')
    fig_raw.colorbar(mesh, ax=ax_raw, label='I (nA)')
    ax_raw.set_xlabel('Gate voltage (V)')
    ax_raw.set_ylabel('SD bias (V)')
    ax_raw.set_title('Raw 2D scan')

    filepath_raw_data = os.path.join(filepath, "raw_data_" + filename)
    fig_raw.savefig(filepath_raw_data, dpi = 'figure', bbox_inches='tight')

    payload = {
                'stage': 'Bootstrapping',
                'step_name': 'Raw Coulomb Diamonds Plot',
                'figure_object': fig_raw,
                'results': {},
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig_raw)

    # Fill any pixel the scan never visited by interpolating along the gate axis
    if np.isnan(Z_matrix).any():
        for j in range(Z_matrix.shape[0]):
            row = Z_matrix[j]
            missing = np.isnan(row)
            if missing.all():
                row[:] = 0.0
            elif missing.any():
                row[missing] = np.interp(np.flatnonzero(missing),
                                         np.flatnonzero(~missing),
                                         row[~missing])

    # --- derivative / edge map ---
    # Differentiating along the gate axis cancels the bias-only ohmic background
    Z_smooth = gaussian_filter(Z_matrix, sigma=smoothing, mode='nearest')
    dI_dx = np.gradient(Z_smooth, unique_x, axis=1)
    
    # Normalise each bias row so low-signal rows still contribute ridge points.
    # Rows whose derivative never rises above floating-point rounding are zeroed
    # first: without that floor, normalising would amplify the rounding noise of a
    # perfectly flat scan into apparent edges.
    edge_map = np.abs(dI_dx)
    gate_step = float(np.median(np.diff(unique_x)))
    # Use the larger of the current's range and its magnitude, so a scan holding a
    # constant non-zero current still gets a floor well above rounding noise
    current_scale = max(float(np.ptp(Z_matrix)), float(np.max(np.abs(Z_matrix))))
    derivative_floor = 1e-4 * current_scale / max(gate_step, 1e-12)
    row_scale = np.percentile(edge_map, contrast_percentile, axis=1, keepdims=True)
    edge_norm = np.where(row_scale > derivative_floor,
                         edge_map / np.where(row_scale > 0, row_scale, 1.0),
                         0.0)

    # --- ridge points ---
    bias_half = np.max(np.abs(unique_y))
    y_inner = zero_bias_exclusion * bias_half
    y_outer = bias_window * bias_half

    # A Coulomb diamond is by definition a bias-dependent feature, so if the scan's
    # structure is the same at every bias there is nothing to find. This catches a
    # scan that varies only along the gate axis, whose constant gate derivative can
    # otherwise survive the later tests.
    structure = Z_smooth - np.median(Z_smooth, axis=1, keepdims=True)
    varies_with_bias = float(np.mean(np.std(structure, axis=0)))
    varies_with_gate = float(np.mean(np.std(structure, axis=1)))
    if varies_with_gate > 0 and varies_with_bias < 0.02 * varies_with_gate:
        raise ValueError(
            f"The scan's structure does not depend on the bias "
            f"(variation across bias is {varies_with_bias / varies_with_gate:.1%} of "
            f"the variation across the gate), so it cannot contain Coulomb diamonds.")

    if x_window is None:
        x_lo_lim, x_hi_lim = unique_x.min(), unique_x.max()
    else:
        x_lo_lim, x_hi_lim = float(min(x_window)), float(max(x_window))

    in_x = (unique_x >= x_lo_lim) & (unique_x <= x_hi_lim)

    ridge_x, ridge_y, ridge_w = [], [], []
    for j, y_val in enumerate(unique_y):
        if not (y_inner <= abs(y_val) <= y_outer):
            continue
        row = np.where(in_x, edge_norm[j], 0.0)
        # Prominence as well as height: a diamond edge is a genuine ridge standing
        # clear of its surroundings, whereas a scan with a constant gate derivative
        # gives a flat row whose rounding wiggles would otherwise clear a pure
        # height threshold and masquerade as edges.
        found, _ = signal.find_peaks(row, height=edge_threshold,
                                     prominence=0.15 * edge_threshold,
                                     distance=peak_distance)
        for i in found:
            ridge_x.append(unique_x[i])
            ridge_y.append(y_val)
            ridge_w.append(row[i])

    ridge_x = np.array(ridge_x)
    ridge_y = np.array(ridge_y)
    ridge_w = np.array(ridge_w)

    if ridge_x.size < 2 * min_edge_points:
        raise ValueError(f"Found only {ridge_x.size} edge points in the derivative "
                         f"map, which is too few to define any diamond. Try lowering "
                         f"edge_threshold or smoothing.")

    # --- slope search ---
    # Every diamond edge extrapolates to a charge-degeneracy point on the zero
    # bias axis, so the correct slope is the one whose intercepts pile up.
    bin_width = gate_step
    # The accumulator spans exactly the searched gate range, because a degeneracy
    # point can only lie inside it. Holding that span fixed also gives the
    # concentration score the same denominator for every trial slope, which removes
    # the bias that would otherwise favour extreme slopes.
    hist_lo, hist_hi = x_lo_lim, x_hi_lim
    n_bins = max(16, int(round((hist_hi - hist_lo) / bin_width)))

    def _intercepts(m):
        """Zero-bias intercept of the line of slope m through each ridge point."""
        return ridge_x - ridge_y / m

    # Charge-degeneracy spacing, measured from the period of the zero-bias Coulomb
    # oscillation. This sets the scale for the slope search, for how much the
    # accumulator is smoothed, and for how close two degeneracy points may sit, so
    # getting it wrong by a factor of two halves every diamond.
    #
    # The oscillation must be measured on the SIGNED residual, not on the magnitude
    # of the derivative: taking the magnitude of an oscillation doubles its apparent
    # frequency, which would report half the true spacing. The residual is
    # phase-aligned by the sign of the bias first, because the current itself
    # reverses with bias.
    spacing = None
    period_measured = False
    osc_band = ((np.abs(unique_y) >= y_inner)
                & (np.abs(unique_y) <= max(2.0 * y_inner, 0.25 * bias_half)))
    if osc_band.sum() >= 3 and in_x.sum() >= 8:
        residual_signed = Z_smooth - np.median(Z_smooth, axis=1, keepdims=True)
        profile = (residual_signed[osc_band][:, in_x]
                   * np.sign(unique_y[osc_band])[:, None]).mean(axis=0)
        profile = profile - profile.mean()
        if np.any(profile):
            osc = np.correlate(profile, profile, mode='full')[profile.size - 1:]
            if osc[0] > 0:
                osc = osc / osc[0]
                osc_peaks, _ = signal.find_peaks(osc[:max(4, profile.size // 2)],
                                                 height=0.10)
                osc_peaks = osc_peaks[osc_peaks >= 2]
                if osc_peaks.size:
                    # first autocorrelation peak is the fundamental period
                    spacing = float(osc_peaks[0]) * bin_width
                    period_measured = True

    if spacing is None:
        # Fall back to counting features on the edge map. Note this counts two per
        # oscillation, so it can under-estimate the spacing.
        near_zero = np.abs(unique_y) <= max(y_inner, 2.0 * bin_width)
        if near_zero.any():
            zero_profile = edge_norm[near_zero].mean(axis=0)
        else:
            zero_profile = edge_norm[np.argmin(np.abs(unique_y))]
        zero_profile = np.where(in_x, zero_profile, 0.0)
        zero_peaks, _ = signal.find_peaks(zero_profile, distance=peak_distance)
        if zero_peaks.size >= 2:
            spacing = float(np.median(np.diff(unique_x[zero_peaks])))
        else:
            spacing = 4.0 * bin_width

    spacing = max(spacing, 2.0 * bin_width)

    def _smoothing_for(spacing_estimate):
        """Accumulator smoothing scaled to the expected degeneracy spacing.

        Too little and one degeneracy point breaks into several fragments; too
        much and neighbouring points merge.
        """
        return float(np.clip(0.12 * spacing_estimate / bin_width, 1.0, 12.0))

    def _accumulator(m, sigma):
        """Weighted histogram of those intercepts, lightly smoothed."""
        hist, edges = np.histogram(_intercepts(m), bins=n_bins,
                                   range=(hist_lo, hist_hi), weights=ridge_w)
        return gaussian_filter1d(hist, sigma), edges

    if slope_range is None:
        # Cap the search at the steepest slope that still keeps the apex of a
        # diamond of that width inside the searched bias window: a symmetric
        # diamond of width w has its apex at w*m/2. Without a cap the accumulator
        # drifts toward near-vertical features, which are common in these scans and
        # are not diamond edges. The floor is the shallowest slope whose apex still
        # stands a few bias pixels clear of the zero-bias axis.
        bias_step = float(np.median(np.diff(unique_y)))
        lo_m = max(6.0 * bias_step / spacing, 0.02)
        # Generous per-slope bound only. The real constraint couples the two slopes,
        # so it is applied to each candidate pair below rather than to either slope
        # alone: a strongly asymmetric diamond can have one very steep edge and still
        # keep its apex inside the scan.
        hi_m = float(np.clip(8.0 * bias_half / spacing, 4.0 * lo_m, 400.0))
    else:
        lo_m, hi_m = float(min(np.abs(slope_range))), float(max(np.abs(slope_range)))

    # Search the two slopes jointly rather than one at a time. The two edge families
    # of a diamond must terminate at the same charge-degeneracy points, so a pair is
    # scored by how well the intercepts of both families agree. Searching them
    # independently can settle on a pair that is individually plausible but mutually
    # inconsistent, which is exactly the failure this avoids.
    n_grid = max(24, int(np.sqrt(slope_steps)) * 3)
    trial = np.linspace(lo_m, hi_m, n_grid)
    acc_sigma = _smoothing_for(spacing)

    # One accumulator per candidate slope, computed once and then reused for every
    # pair, so the joint search costs 2*n_grid histograms rather than n_grid squared
    acc_plus = np.clip(np.array([_accumulator(+m, acc_sigma)[0] for m in trial]), 0.0, None)
    acc_minus = np.clip(np.array([_accumulator(-m, acc_sigma)[0] for m in trial]), 0.0, None)

    pair_score = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        combined = np.sqrt(acc_plus[i][None, :] * acc_minus)
        total = combined.sum(axis=1)
        ok = total > 0
        pair_score[i, ok] = ((combined[ok] ** 2).sum(axis=1) / total[ok] ** 2) * n_bins

    # A diamond of width w bounded by slopes a and b has its apex at w*a*b/(a+b),
    # the harmonic-mean combination. Requiring that apex to lie inside the measured
    # bias range is the exact physical constraint, and it couples the two slopes: one
    # edge may be very steep provided the other is shallow enough. Applying it to the
    # pair also removes the degenerate very-steep corner of the grid, where every
    # intercept collapses onto the point's own gate voltage and any clustering in
    # gate voltage alone would inflate the score.
    mag_pos, mag_neg = np.meshgrid(trial, trial, indexing='ij')
    apex_height = spacing * (mag_pos * mag_neg) / (mag_pos + mag_neg)
    pair_score[apex_height > bias_half] = 0.0

    if not np.any(np.isfinite(pair_score)) or pair_score.max() <= 0:
        raise ValueError("The slope search found no consistent family of diamond "
                         "edges. Try widening slope_range or adjusting smoothing.")

    # Best-scoring valid pair. No preference for shallow slopes is needed here: the
    # apex constraint above already excludes the degenerate steep corner, and adding
    # one on top biases strongly asymmetric diamonds toward too-shallow edges.
    chosen = np.unravel_index(int(np.argmax(pair_score)), pair_score.shape)
    m_pos_global = float(trial[chosen[0]])
    m_neg_global = float(-trial[chosen[1]])
    pair_concentration = float(pair_score[chosen[0], chosen[1]])

    # A featureless scan (pure background, or pure noise) still yields ridge points
    # once the edge map is row-normalised, but their zero-bias intercepts do not
    # pile up. A concentration near 1.0 means the intercepts are spread as evenly
    # as random points would be, so there is no diamond structure to find.
    weakest = pair_concentration
    if weakest < min_concentration:
        raise ValueError(
            f"No clear Coulomb diamond structure was found: the best edge family "
            f"concentrates its zero-bias intercepts only {weakest:.2f}x more than an "
            f"even spread would, below min_concentration = {min_concentration}. Pure "
            f"noise scores about 1.1-1.3, so a score in that region cannot be "
            f"distinguished from noise automatically.\n"
            f"  If you can see diamonds in the scan, the search is most likely being "
            f"pulled off them by another feature. Things to try, in order:\n"
            f"   1. x_window=(lo, hi) to exclude turn-on or other non-diamond regions\n"
            f"   2. bias_window smaller (e.g. 0.3) so only the low-bias region is used\n"
            f"   3. slope_range=(lo, hi) to pin the search to the edge slope you can "
            f"see, which is the strongest lever when a scan holds more than one line "
            f"family (for example near-vertical charge-sensor transitions)\n"
            f"   4. min_concentration lower to force a result, accepting that the "
            f"output may not be meaningful\n"
            f"  Best slope pair found so far: m_pos = {m_pos_global:+.3f}, "
            f"m_neg = {m_neg_global:+.3f}. Run with debug=True to see the edge map, "
            f"the detected edge points and the intercept accumulator.")

    # --- charge-degeneracy points ---
    # Both edge families intercept the zero-bias axis at degeneracy points, combined
    # here with a geometric mean rather than a sum. A real degeneracy point
    # terminates one edge of each family, so it appears in both accumulators,
    # whereas the smear left when one family is projected with the other family's
    # slope appears in only one. The geometric mean keeps the former and suppresses
    # the latter.
    def _combined(sigma):
        pos, edges = _accumulator(m_pos_global, sigma)
        neg, _ = _accumulator(m_neg_global, sigma)
        return np.sqrt(np.clip(pos, 0.0, None) * np.clip(neg, 0.0, None)), edges

    accumulator, edges = _combined(_smoothing_for(spacing))
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Degeneracy points must lie inside the region actually searched
    accumulator = np.where((centres >= x_lo_lim) & (centres <= x_hi_lim), accumulator, 0.0)

    if accumulator.max() <= 0:
        raise ValueError("No charge-degeneracy points were found on the zero-bias axis.")

    # Two degeneracy points cannot sit much closer together than the oscillation
    # period. When the period was measured from the oscillation itself it is
    # trustworthy, so require most of it: a gap of only half a period means one
    # diamond has been split in two, which draws a half-width shape sitting inside
    # the real diamond. When only the fallback estimate is available, stay
    # permissive rather than risk merging genuinely unevenly spaced diamonds.
    separation_factor = 0.75 if period_measured else 0.45
    min_separation = max(2, int(round(separation_factor * spacing / bin_width)))
    found, _ = signal.find_peaks(accumulator,
                                 height=degeneracy_prominence * accumulator.max(),
                                 distance=min_separation)

    # Projecting one edge family with the other family's slope leaves a weak
    # artefact peak at the diamond apex, exactly midway between two real
    # degeneracy points. Drop any peak that sits near such a midpoint and is
    # clearly weaker than both of its neighbours. Real neighbouring degeneracy
    # points have comparable weight, so they survive this test.
    trimmed = True
    while trimmed and found.size >= 3:
        trimmed = False
        heights = accumulator[found]
        for k in range(1, found.size - 1):
            gap = found[k + 1] - found[k - 1]
            midpoint = 0.5 * (found[k + 1] + found[k - 1])
            near_middle = gap > 0 and abs(found[k] - midpoint) <= 0.25 * gap
            much_weaker = heights[k] < 0.7 * min(heights[k - 1], heights[k + 1])
            if near_middle and much_weaker:
                found = np.delete(found, k)
                trimmed = True
                break

    degeneracy = np.sort(centres[found])

    if degeneracy.size < 2:
        raise ValueError(f"Found {degeneracy.size} charge-degeneracy point(s) on the "
                         f"zero-bias axis; at least 2 are needed to bracket a diamond. "
                         f"Try lowering degeneracy_prominence or edge_threshold.")

    # --- per-diamond refinement ---
    upper = ridge_y > 0
    lower = ridge_y < 0

    # Group the ridge points by bias row once, so each diamond can look up the edge
    # points in a row without rescanning the whole cloud
    rows_of_ridges = {}
    for xr_, yr_, wr_ in zip(ridge_x, ridge_y, ridge_w):
        rows_of_ridges.setdefault(yr_, []).append(xr_)
    for yr_ in rows_of_ridges:
        rows_of_ridges[yr_] = np.sort(np.array(rows_of_ridges[yr_]))

    def _fit_edges(x_left, x_right, m_pos_guess, m_neg_guess):
        """Fit the four edges of one diamond independently.

        For each bias row the expected position of each edge is predicted from the
        global slope, and the nearest ridge point within a narrow corridor of that
        prediction is taken as the measured edge point. Simply taking the outermost
        ridge in the row does not work: these scans often carry extra, much steeper
        lines running through the diamonds, and the outermost ridge is then one of
        those rather than the diamond boundary.

        Each edge is forced through its own zero-bias vertex, which keeps the
        diamond closed, and its slope is the median of the slopes implied by its
        points. The four results are only loosely tied together: each is held within
        ``slope_consistency`` of the corresponding global slope, so the edges stay
        similar without being forced equal.
        """
        width = x_right - x_left
        corridor = 0.3 * width
        left_upper, left_lower, right_upper, right_lower = [], [], [], []
        for y_val, xs_row in rows_of_ridges.items():
            for anchor, guess, bucket in (
                    (x_left, m_pos_guess if y_val > 0 else m_neg_guess,
                     left_upper if y_val > 0 else left_lower),
                    (x_right, m_neg_guess if y_val > 0 else m_pos_guess,
                     right_upper if y_val > 0 else right_lower)):
                predicted = anchor + y_val / guess
                if not (x_left - corridor <= predicted <= x_right + corridor):
                    continue
                near = xs_row[np.abs(xs_row - predicted) <= corridor]
                if near.size == 0:
                    continue
                measured = float(near[np.argmin(np.abs(near - predicted))])
                if abs(measured - anchor) > 1e-12:
                    bucket.append(y_val / (measured - anchor))

        def _median_slope(values, guess, want_positive):
            vals = np.array([v for v in values if np.isfinite(v)
                             and (v > 0 if want_positive else v < 0)])
            if vals.size < min_edge_points:
                return guess
            estimate = float(np.median(vals))
            lo = guess * (1.0 - slope_consistency)
            hi = guess * (1.0 + slope_consistency)
            return float(np.clip(estimate, min(lo, hi), max(lo, hi)))

        return (_median_slope(left_upper, m_pos_guess, True),    # top-left
                _median_slope(right_upper, m_neg_guess, False),  # top-right
                _median_slope(left_lower, m_neg_guess, False),   # bottom-left
                _median_slope(right_lower, m_pos_guess, True))   # bottom-right



    diamond_slopes = {}
    outlines = []
    rejected = {'geometry': 0, 'off_the_sides': 0, 'taller_than_scan': 0,
                'too_few_interior_pixels': 0, 'no_blockade': 0}
    x_min, x_max = unique_x.min(), unique_x.max()

    # Coordinates of every pixel, plus the reference edge strength that the
    # blockade test for each candidate diamond is compared against
    grid_x, grid_y = np.meshgrid(unique_x, unique_y)
    searched = np.zeros_like(edge_norm, dtype=bool)
    searched[np.ix_((np.abs(unique_y) >= y_inner) & (np.abs(unique_y) <= y_outer), in_x)] = True
    region_mean = float(edge_norm[searched].mean()) if searched.any() else float(edge_norm.mean())

    # --- orientation ---
    # The two slope magnitudes could be assigned either way round: putting the
    # steeper one on the top-left/bottom-right pair leans the parallelogram one way,
    # putting it on the top-right/bottom-left pair leans it the other. Both
    # assignments give exactly the same zero-bias intercepts, so the intercept
    # agreement score cannot choose between them and a wrong choice mirrors every
    # diamond. The Coulomb blockade decides it: only the correct assignment puts the
    # current-suppressed region inside the outlined shape. Chord conductance I/V is
    # the direct measure, being small inside a diamond and large outside.
    with np.errstate(divide='ignore', invalid='ignore'):
        chord = np.abs(Z_matrix / grid_y)
    usable = (np.abs(unique_y) >= max(y_inner, 2.0 * abs(float(np.median(np.diff(unique_y))))))
    chord_norm = np.full_like(chord, np.nan)
    for j in np.flatnonzero(usable):
        row = chord[j]
        finite = np.isfinite(row)
        if finite.any():
            scale = np.median(row[finite])
            if scale > 0:
                chord_norm[j] = row / scale

    def _corners(m_tl, m_tr, m_bl, m_br, x_left, x_right):
        """The four corners of a diamond from its two vertices and four edge slopes.

        The apex is where the two upper edges meet and the base where the two lower
        edges meet. Returns None when the edges do not close into a sensible shape.
        """
        if m_tl <= 0 or m_br <= 0 or m_tr >= 0 or m_bl >= 0:
            return None
        if np.isclose(m_tl, m_tr) or np.isclose(m_bl, m_br):
            return None
        x_top = (m_tl * x_left - m_tr * x_right) / (m_tl - m_tr)
        y_top = m_tl * (x_top - x_left)
        x_bot = (m_bl * x_left - m_br * x_right) / (m_bl - m_br)
        y_bot = m_bl * (x_bot - x_left)
        if not (y_top > 0 > y_bot):
            return None
        if not (x_left <= x_top <= x_right and x_left <= x_bot <= x_right):
            return None
        return ((x_left, 0.0), (x_top, y_top), (x_right, 0.0), (x_bot, y_bot))

    def _interior_mask(corners, shrink=0.35):
        """Pixels well inside the quadrilateral, shrunk toward its centre.

        The four edges are fitted independently, so the shape is a general convex
        quadrilateral rather than a parallelogram and the containment test has to be
        general too. Shrinking toward the centroid keeps the sample clear of the
        edges themselves.
        """
        pts = np.array(corners, dtype=float)
        centre = pts.mean(axis=0)
        pts = centre + (1.0 - shrink) * (pts - centre)

        # Normalise the winding direction from the signed area, so the side test is
        # correct whichever way round the corners happen to be ordered
        signed_area = 0.0
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            signed_area += x0 * y1 - x1 * y0
        winding = 1.0 if signed_area > 0 else -1.0

        inside = np.ones(grid_x.shape, dtype=bool)
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            # the sign of the cross product says which side of the edge a point is on
            cross = (x1 - x0) * (grid_y - y0) - (y1 - y0) * (grid_x - x0)
            inside &= winding * cross >= 0
        return inside

    def _blockade_depth(mp, mn, points=None):
        """Mean normalised chord conductance inside every candidate diamond.

        Lower means the outlined shapes really do sit on blockaded regions, so this
        is the direct measure of whether the outline encloses the diamond. Uses the
        symmetric two-slope form, which is enough for comparing candidate placements.
        """
        pts = degeneracy if points is None else points
        samples = []
        for k in range(len(pts) - 1):
            corners = _corners(mp, mn, mn, mp, float(pts[k]), float(pts[k + 1]))
            if corners is None:
                continue
            vals = chord_norm[_interior_mask(corners)]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                samples.append(vals)
        if not samples:
            return np.inf
        return float(np.concatenate(samples).mean())

    a, b = abs(m_pos_global), abs(m_neg_global)
    if not np.isclose(a, b):
        depth_direct = _blockade_depth(a, -b)
        depth_swapped = _blockade_depth(b, -a)
        # Require a real improvement before mirroring every diamond. When the two
        # depths are all but equal the blockade genuinely does not favour either
        # orientation, and flipping on that would just be following noise.
        if depth_swapped < depth_direct * (1.0 - orientation_margin):
            m_pos_global, m_neg_global = b, -a
            if debug:
                print(f"orientation: swapped to m_pos={m_pos_global:+.3f}, "
                      f"m_neg={m_neg_global:+.3f} (blockade {depth_swapped:.4f} vs "
                      f"{depth_direct:.4f})")
        elif debug:
            print(f"orientation: kept m_pos={m_pos_global:+.3f}, "
                  f"m_neg={m_neg_global:+.3f} (blockade {depth_direct:.4f} vs "
                  f"{depth_swapped:.4f})")

    # --- alignment ---
    # The accumulator locates the degeneracy points from the edge ridges, and those
    # can sit systematically off by a fraction of a period, which slides every
    # outline off its diamond. A diamond is ultimately defined by the blockaded
    # region between two degeneracy points, so shift the whole set and rescale the
    # slope pair to minimise the conductance enclosed by the outlines. This aligns
    # the shapes with the diamonds themselves rather than with the ridge statistics.
    # Only a rigid shift is considered, and only a large, unambiguous improvement is
    # accepted. A small gain is not evidence of a real phase error and chasing it
    # drags well-placed outlines off correctly located diamonds.
    align_before = _blockade_depth(m_pos_global, m_neg_global)
    best_align = align_before

    shift_grid = np.linspace(-0.5, 0.5, 41) * spacing
    shift_scores = np.array([_blockade_depth(m_pos_global, m_neg_global, degeneracy + s)
                             for s in shift_grid])
    if np.any(np.isfinite(shift_scores)):
        candidate = float(shift_grid[int(np.nanargmin(shift_scores))])
        candidate_score = float(np.nanmin(shift_scores))
        if (np.isfinite(align_before) and align_before > 0
                and candidate_score < (1.0 - alignment_gain) * align_before):
            shifted = degeneracy + candidate
            shifted = shifted[(shifted >= x_lo_lim) & (shifted <= x_hi_lim)]
            if shifted.size >= 2:
                degeneracy = shifted
                best_align = candidate_score
                if debug:
                    print(f"alignment: shifted degeneracy by {candidate:+.5f} V "
                          f"({candidate / spacing:+.2f} period); enclosed conductance "
                          f"{align_before:.4f} -> {best_align:.4f}")
        elif debug:
            print(f"alignment: no shift applied (best available gain "
                  f"{100 * (align_before - candidate_score) / align_before:.0f}%, "
                  f"needs {100 * alignment_gain:.0f}%)")

    # Let the per-edge fits feed back into the global pair. The joint search gives a
    # starting point from the whole scan, but if every edge in every diamond is
    # pulled to one side of the allowed band then the starting point is wrong and
    # holding onto it would keep all the outlines the wrong shape. Re-deriving the
    # global pair from the fitted edges and repeating converges on the value the data
    # actually supports.
    for _ in range(4):
        fitted_pos, fitted_neg = [], []
        for k in range(degeneracy.size - 1):
            m_tl, m_tr, m_bl, m_br = _fit_edges(float(degeneracy[k]),
                                                float(degeneracy[k + 1]),
                                                m_pos_global, m_neg_global)
            fitted_pos.extend([m_tl, m_br])
            fitted_neg.extend([m_tr, m_bl])
        if not fitted_pos or not fitted_neg:
            break
        new_pos = float(np.median(fitted_pos))
        new_neg = float(np.median(fitted_neg))
        if new_pos <= 0 or new_neg >= 0:
            break
        # Only accept the update if it encloses the blockade better than the current
        # pair. The edge fits follow the ridges, which is the finer measurement, but
        # these scans also carry steeper non-diamond lines that pull the fits away,
        # so the blockade has the final say on placement.
        new_align = _blockade_depth(new_pos, new_neg)
        if new_align >= best_align:
            break
        converged = (abs(new_pos - m_pos_global) < 0.01 * abs(m_pos_global)
                     and abs(new_neg - m_neg_global) < 0.01 * abs(m_neg_global))
        m_pos_global, m_neg_global = new_pos, new_neg
        best_align = new_align
        if converged:
            break

    if debug:
        print(f"global slopes after edge refinement: m_pos = {m_pos_global:+.3f}, "
              f"m_neg = {m_neg_global:+.3f} "
              f"(enclosed conductance {_blockade_depth(m_pos_global, m_neg_global):.4f})")

    for k in range(degeneracy.size - 1):
        x_left = float(degeneracy[k])
        x_right = float(degeneracy[k + 1])

        # Each of the four edges is fitted to its own points
        m_tl, m_tr, m_bl, m_br = _fit_edges(x_left, x_right,
                                            m_pos_global, m_neg_global)



        corners = _corners(m_tl, m_tr, m_bl, m_br, x_left, x_right)
        if corners is None:
            rejected['geometry'] += 1
            continue

        (_, _), (x_top, y_top), (_, _), (x_bot, y_bot) = corners

        # Drop diamonds running off the sides of the scan
        if min(x_left, x_right, x_top, x_bot) < x_min or max(x_left, x_right, x_top, x_bot) > x_max:
            rejected['off_the_sides'] += 1
            continue

        # Drop diamonds much taller than the scan, since their apex or base was
        # never measured. A small overshoot is allowed: when the diamonds nearly fill
        # the bias range, ordinary fit uncertainty can put the apex just beyond the
        # last measured row and rejecting those would discard good diamonds.
        bias_tolerance = 1.15
        if (y_top > bias_tolerance * unique_y.max()
                or y_bot < bias_tolerance * unique_y.min()):
            rejected['taller_than_scan'] += 1
            continue

        # Inside a diamond the dot is blockaded, so the gate derivative of the
        # current should be quiet compared with the rest of the searched region.
        interior = _interior_mask(corners)
        if interior.sum() < min_edge_points:
            rejected['too_few_interior_pixels'] += 1
            continue
        if edge_norm[interior].mean() > blockade_ratio * region_mean:
            rejected['no_blockade'] += 1
            continue

        label = f"diamond_{len(diamond_slopes) + 1}"
        diamond_slopes[label] = {'top_left': m_tl,
                                 'top_right': m_tr,
                                 'bottom_left': m_bl,
                                 'bottom_right': m_br}
        outlines.append({'label': label,
                         'left': (x_left, 0.0),
                         'top': (x_top, y_top),
                         'right': (x_right, 0.0),
                         'bottom': (x_bot, y_bot)})

    if not diamond_slopes:
        breakdown = ", ".join(f"{reason}: {count}" for reason, count in rejected.items() if count)
        raise ValueError(f"No complete Coulomb diamonds were found: "
                         f"{degeneracy.size} degeneracy point(s) gave "
                         f"{degeneracy.size - 1} candidate diamond(s), all rejected "
                         f"({breakdown}).")

    # --- averaged properties ---
    # The two positive edges of every diamond (top-left and bottom-right) belong to
    # one slope family and the two negative edges (top-right and bottom-left) to the
    # other, so each family is averaged over every edge of every diamond.
    positive_slopes = [s[edge] for s in diamond_slopes.values()
                       for edge in ('top_left', 'bottom_right')]
    negative_slopes = [s[edge] for s in diamond_slopes.values()
                       for edge in ('top_right', 'bottom_left')]

    # Height is the apex-to-base separation along the bias axis, in volts of SD bias.
    # Width is the gate-voltage separation of the two charge-degeneracy points that
    # bracket the diamond.
    diamond_heights = [shape['top'][1] - shape['bottom'][1] for shape in outlines]
    diamond_widths = [shape['right'][0] - shape['left'][0] for shape in outlines]

    diamond_properties = {
        'average_positive_slope': float(np.mean(positive_slopes)),
        'average_negative_slope': float(np.mean(negative_slopes)),
        'average_diamond_height': float(np.mean(diamond_heights)),
        'average_diamond_width': float(np.mean(diamond_widths)),
        'Number of Coulomb Diamonds': len(diamond_slopes)
    }

    # --- reporting ---
    if debug:
        print(f"slope search range: |m| in [{lo_m:.2f}, {hi_m:.2f}]")
        print(f"global slopes (joint search): m_pos = {m_pos_global:+.3f}, "
              f"m_neg = {m_neg_global:+.3f}  (pair concentration "
              f"{pair_concentration:.2f}x)")
        print(f"degeneracy spacing estimate: {spacing:.5f} V")
        print(f"degeneracy points ({degeneracy.size}): {np.round(degeneracy, 5)}")
        print(f"accepted {len(diamond_slopes)} of {degeneracy.size - 1} candidate(s); "
              f"rejected " + ", ".join(f"{r}: {c}" for r, c in rejected.items() if c))
        print(f"{'diamond':<12}{'x_left':>10}{'x_right':>10}{'apex V':>9}{'base V':>9}"
              f"{'top_L':>8}{'top_R':>8}{'bot_L':>8}{'bot_R':>8}")
        for shape in outlines:
            s = diamond_slopes[shape['label']]
            print(f"{shape['label']:<12}{shape['left'][0]:>10.5f}{shape['right'][0]:>10.5f}"
                  f"{shape['top'][1]:>9.4f}{shape['bottom'][1]:>9.4f}"
                  f"{s['top_left']:>8.2f}{s['top_right']:>8.2f}"
                  f"{s['bottom_left']:>8.2f}{s['bottom_right']:>8.2f}")
        print(f"averages: positive slope {diamond_properties['average_positive_slope']:+.3f}, "
              f"negative slope {diamond_properties['average_negative_slope']:+.3f}, "
              f"height {diamond_properties['average_diamond_height']:.5f} V, "
              f"width {diamond_properties['average_diamond_width']:.5f} V")

    # --- plotting ---
    # The diamond signal can be only a per-cent-level modulation on the ohmic
    # background, which is invisible on a full-range colour scale. Subtracting each
    # bias row's median removes the background (which depends on bias only) and
    # percentile colour limits then put the remaining modulation across the full
    # colour range, so the diamonds can actually be seen and the fit judged.
    residual = Z_matrix - np.median(Z_matrix, axis=1, keepdims=True)
    r_lo, r_hi = np.percentile(residual, [2, 98])
    if r_hi <= r_lo:
        r_lo, r_hi = residual.min(), residual.max() + 1e-12

    def _draw_outlines(ax):
        for shape in outlines:
            loop = [shape['left'], shape['top'], shape['right'],
                    shape['bottom'], shape['left']]
            ax.plot([p[0] for p in loop], [p[1] for p in loop],
                    color='red', linewidth=1.4)
        ax.scatter([s['left'][0] for s in outlines] + [s['right'][0] for s in outlines],
                   np.zeros(2 * len(outlines)), color='black', s=10, zorder=3,
                   label='charge degeneracy')
        ax.set_xlim(unique_x.min(), unique_x.max())
        ax.set_ylim(unique_y.min(), unique_y.max())
        ax.set_xlabel('Gate voltage (V)')
        ax.set_ylabel('SD bias (V)')
        ax.legend(loc='upper right')

    fig_dia, (ax_dia, ax_res) = plt.subplots(1, 2, figsize=(15, 6))
    mesh2 = ax_dia.pcolormesh(unique_x, unique_y, Z_matrix, shading='auto', cmap='viridis')
    fig_dia.colorbar(mesh2, ax=ax_dia, label='I (nA)')
    if curr.min()*0.2 > curr.max()*0.2:
        v_lim = np.rint(abs(curr.max()*0.2)).astype(int)
    else:
        v_lim = np.rint(abs(curr.min()*0.2)).astype(int)
    mesh2.set_clim(vmin = -v_lim, vmax = v_lim)
    _draw_outlines(ax_dia)
    ax_dia.set_title(f'Coulomb diamonds ({len(diamond_slopes)} found)')

    mesh_r = ax_res.pcolormesh(unique_x, unique_y, residual, shading='auto',
                               cmap='viridis', vmin=r_lo, vmax=r_hi)
    fig_dia.colorbar(mesh_r, ax=ax_res, label='I - row median (nA)')
    _draw_outlines(ax_res)
    ax_res.set_title(f'Background removed, contrast {r_lo:+.2f} to {r_hi:+.2f} nA')
    fig_dia.tight_layout()

    filepath_analyzed = os.path.join(filepath, "analyzed_" + filename)
    fig_dia.savefig(filepath_analyzed, dpi = 'figure', bbox_inches='tight')

    payload = {
                'stage': 'Bootstrapping',
                'step_name': 'Analyzed Coulomb Diamonds',
                'figure_object': fig_dia,
                'results': diamond_properties,
            }
    tuning_bridge.plot_queue.put(payload)

    plt.close(fig_dia)

    if debug:
        fig_dbg, (ax_d1, ax_d2) = plt.subplots(2, 1, figsize=(8, 9))
        mesh3 = ax_d1.pcolormesh(unique_x, unique_y, edge_norm, shading='auto', cmap='inferno')
        fig_dbg.colorbar(mesh3, ax=ax_d1, label='|dI/dVg| (row normalised)')
        ax_d1.scatter(ridge_x, ridge_y, s=4, color='cyan', label='edge points')
        for shape in outlines:
            loop = [shape['left'], shape['top'], shape['right'], shape['bottom'], shape['left']]
            ax_d1.plot([p[0] for p in loop], [p[1] for p in loop], color='lime', linewidth=1.2)
        ax_d1.set_xlabel('Gate voltage (V)')
        ax_d1.set_ylabel('SD bias (V)')
        ax_d1.set_title('Derivative map with detected edge points')
        ax_d1.legend(loc='upper right')

        ax_d2.plot(centres, accumulator, color='black', linewidth=1)
        for xd in degeneracy:
            ax_d2.axvline(xd, color='red', linestyle='--', linewidth=1)
        ax_d2.set_xlim(unique_x.min(), unique_x.max())
        ax_d2.set_xlabel('Gate voltage (V)')
        ax_d2.set_ylabel('accumulated edge weight')
        ax_d2.set_title('Zero-bias intercept accumulator (dashed = degeneracy points)')
        fig_dbg.tight_layout()

        filepath_debugged = os.path.join(filepath, "debugged_" + filename)
        fig_dbg.savefig(filepath_debugged, dpi = 'figure', bbox_inches='tight')

        payload = {
                'stage': 'Bootstrapping',
                'step_name': 'Debugged Coulomb Diamonds',
                'figure_object': fig_dbg,
                'results': diamond_properties,
            }
        tuning_bridge.plot_queue.put(payload)

        plt.close(fig_dbg)

    # plt.show()

    return diamond_slopes, diamond_properties

def run_test_plot():
    """Queue three test panels: a 1D scan, a Matplotlib figure, and a 2D heatmap.

    The second panel passes a Matplotlib figure through ``figure_object`` rather
    than raw arrays. The GUI reads the data back out of that figure and redraws it
    with Plotly, so it ends up just as interactive as the raw-array panels.
    """

    # ---- 1D scan, raw arrays
    x = np.linspace(0, 5, 100)
    y = np.sin(x)
    tuning_bridge.plot_queue.put({
        'stage': 'Bootstrapping',
        'step_name': 'Turn-On (1D, raw arrays)',
        'plot_data': {
            'kind': 'line',
            'x': x,
            'y': y,
            'x_label': 'Gate Voltage (V)',
            'y_label': 'Current (nA)',
            'name': 'Sine Wave',
            'title': 'Test Plot: Sine Wave',
        },
        'results': {'Turn-On Voltage (V)': 2.5},
    })

    # ---- 1D scan, passed as a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label='Sine Wave')
    ax.plot(x, np.cos(x), label='Cosine Wave')
    ax.legend()
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Current (nA)')
    ax.set_title('Test Plot: passed as a Matplotlib figure')
    tuning_bridge.plot_queue.put({
        'stage': 'Bootstrapping',
        'step_name': 'Turn-On (1D, Matplotlib figure)',
        'figure_object': fig,
        'results': {'Traces': 2},
    })

    # ---- 2D heatmap, raw arrays
    gate = np.linspace(1.20, 1.50, 120)
    bias = np.linspace(-0.20, 0.20, 100)
    GX, GY = np.meshgrid(gate, bias)
    current = np.sin(2 * np.pi * (GX - 1.20) / 0.04) * np.exp(-(GY / 0.06) ** 2) + 5 * GY
    tuning_bridge.plot_queue.put({
        'stage': 'Bootstrapping',
        'step_name': 'Coulomb Diamonds (2D heatmap)',
        'plot_data': {
            'kind': 'heatmap',
            'x': gate,
            'y': bias,
            'z': current,
            'x_label': 'Plunger Gate (V)',
            'y_label': 'SD Bias (V)',
            'z_label': 'I (nA)',
            'title': 'Test Plot: 2D heatmap',
        },
        'results': {'Diamonds Found': 7},
    })

    plt.close(fig)

