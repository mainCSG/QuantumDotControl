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
from pathlib import Path
from typing import Callable, Dict, List

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
from scipy.ndimage import convolve, map_coordinates, gaussian_filter1d

import skimage
from skimage import filters, transform
from skimage.feature import canny
from skimage.filters import threshold_otsu, sato
from skimage.morphology import diamond, rectangle  # noqa
from skimage.transform import probabilistic_hough_line

import yaml
from colorlog import ColoredFormatter

import qcodes as qc
from qcodes.dataset import AbstractSweep, Measurement
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase

from nicegui import ui
from tunerlog import TunerLog

logger = TunerLog('Data Analysis')
  
def logarithmic(x, a, b, x0, y0):
    
    """
    Logarithmic model used for curve fitting.

    Parameters
    -----------
    x : np.array
        independent variable array
    a, b, x0, y0 : float
        fit parameters
    
    """
    return a * np.log(b*(x-x0)) + y0

def exponential(x, a, b, x0, y0):
    """Exponential model used for curve fitting.

    Parameters:
        x: independent variable array
        a, b, x0, y0: fit parameters
    """
    return a * np.exp(b * (x-x0)) + y0

def sigmoid(x, a, b, x0, y0):
    """Sigmoid model used for turn-on / pinch-off fitting.

    Parameters:
        x: independent variable array
        a, b, x0, y0: fit parameters
    """
    return a * expit(-b * (x - x0)) + y0

def linear(x, m, b):
    """Simple linear model for fitting straight-line behavior."""
    return m * x + b         

def relu(x, a, x0):
    """ReLU-style model that is zero below x0 and linear above it."""
    return np.maximum(0, a * (x - x0))

def fit_to_function(x_data, 
                    y_data, 
                    function: Callable,
                    p0: list[float] = None,
                    print_results: bool = True):
    """Fit a provided model function to x/y data using nonlinear least squares.

    Parameters:
        x_data: independent variable values
        y_data: dependent variable values
        function: callable model to fit (e.g. sigmoid)
        p0: optional initial guess for model parameters
        print_results: whether to print fitted parameter values

    Returns:
        params: model parameter names
        popt: optimized parameter values
        pcov: covariance matrix of parameter estimates
    """

    if p0 is None:
        popt, pcov = curve_fit(function, x_data, y_data)
        perr = np.sqrt(np.diag(pcov))
    else:
        popt, pcov = curve_fit(function, x_data, y_data, p0=p0)
        perr = np.sqrt(np.diag(pcov))
    
    params = list(inspect.signature(function).parameters.keys())[1:]

    if print_results:
        for name, val, err in zip(params, popt, perr):
            print(f"{name} = {val:.3f} ± {err:.3f}")

    return params, popt, pcov

def extract_turn_on_voltage(x_data: np.array,
                            y_data: np.array,
                            noisefloor: float,
                            filepath: str,
                            filename: str,
                            plot_results: bool = True):
    
    """Estimate the turn-on voltage from a gate-sweep current curve.

    This routine baseline-corrects the current, then finds the first
    voltage where the current rises above the provided threshold.
    """

    # --- Data definitions ---
    
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    if y1[-1] < 0:
        y1 = -y1

    # --- Finding Turn-On Voltage ---
    turnon_voltage = 0
    turnon_current = 0

    threshold = noisefloor * 10

    for val in y1:

        if val > abs(threshold):
            idx_turnon = np.where(y1 == val)[0][0] # get the index of the turn-on point
            
            logger.info(f"Index: {idx_turnon}")
            
            turnon_voltage = x1[idx_turnon - 1]
            
            logger.info(f"Turn_On Voltage: {turnon_voltage}")
            
            turnon_current = y1[idx_turnon - 1]
            break

    # --- Plot data ---
    
    if plot_results:

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(x1, y1, '-', color='C0', linewidth=2, label='I ($V_{gate}$)')
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

        filepath = os.path.join(filepath, filename)

        fig.savefig(filepath, dpi = 'figure', bbox_inches='tight')

        plt.close(fig)

    # --- Print summary ---
    #print(f"  Turn-on Voltage:  {turnon_voltage:.3f} V")

    return turnon_voltage, fig

def extract_pinch_off_curve_ranges(x_data: np.array,
                           y_data: np.array,
                           noisefloor: float,
                           filepath: str,
                           filename: str,
                           debug: bool = False,
                           plot_results: bool = True):
    """Identify pinch-off and saturation voltage ranges for a sweep.

    This function normalizes the sign of the current, selects the scan
    direction from the zero-voltage point, finds the pinch-off position
    using slope detection, and then locates the saturation region.
    """

    # --- Data definitions ---
    
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    if y1[0] < 0:
        y1 = -y1

    # --- Finding Pinch-off Voltage ---

    start_idx = int(np.argmin(np.abs(x1)))
    if start_idx == 0:
        step = 1
    elif start_idx == len(x1) - 1:
        step = -1
    else:
        left_abs = abs(x1[start_idx - 1])
        right_abs = abs(x1[start_idx + 1])
        step = 1 if right_abs >= left_abs else -1

    scan_indices = np.arange(start_idx, len(x1), step) if step > 0 else np.arange(start_idx, -1, -1)
    x_scan = x1[scan_indices]
    y_scan = y1[scan_indices]
    
    # 1. Use the first 5% of data points (closest to 0V) to characterize the noise floor
    baseline_window = max(5, int(0.05 * len(y_scan)))
    baseline_data = y_scan[:baseline_window]
    baseline_mean = np.mean(baseline_data)
    baseline_std = np.std(baseline_data)
    total_signal_range = np.max(y_scan) - np.min(y_scan)

    # If the start already exhibits heavy oscillations relative to the total range,
    # it means the device turn-on is active right from the initial gate voltage.
    if baseline_std > 0.02 * total_signal_range:
        pinch_off_pos = 0
    else:
        # Otherwise, find where it cleanly breaks away from a quiet noise floor
        departure_threshold = baseline_mean + max(3.0 * baseline_std, 0.008 * total_signal_range)
        pinch_off_pos = 0
        consecutive_points_needed = 3
        for i in range(len(y_scan) - consecutive_points_needed):
            if all(y_scan[i + j] > departure_threshold for j in range(consecutive_points_needed)):
                pinch_off_pos = i
                break

    early_rise_threshold = max(3, int(0.05 * len(x_scan)))
    if pinch_off_pos <= early_rise_threshold or pinch_off_pos >= len(x_scan) - 1:
        idx_pinch_off = int(scan_indices[0])
        pinch_off_pos = 0  # Sync position
    else:
        idx_pinch_off = int(scan_indices[pinch_off_pos])

    # Local peak adjustment fallback (kept from your original architecture)
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

    pinch_off_voltage = x1[idx_pinch_off]
    pinch_off_current = y1[idx_pinch_off]
    
    # --- Finding Saturation Voltage using Smooth Derivatives ---

    after_pinch = y_scan[pinch_off_pos:]
    if len(after_pinch) < 5:
        idx_sat = int(scan_indices[-1])
        sat_voltage = x1[idx_sat]
        sat_current = y1[idx_sat]
        sat_threshold = None
        sat_plateau_start = None
    else:
        dx = np.abs(np.diff(x_scan))
        dx_mean = np.mean(dx) if len(dx) > 0 else 1.0
        
        window_length = min(15, len(after_pinch) // 3)
        if window_length % 2 == 0:
            window_length = max(3, window_length - 1)
        window_length = max(3, window_length)

        # Compute smooth 1st derivative
        y_der = signal.savgol_filter(after_pinch, window_length=window_length, polyorder=2, deriv=1, delta=dx_mean)

        # Characterize terminal tail behavior
        tail_size = min(15, len(y_der) // 4)
        end_slopes = y_der[-tail_size:]
        mean_end_slope = np.mean(end_slopes)
        std_end_slope = np.std(end_slopes)

        # Use the 90th percentile of the derivative instead of the absolute maximum.
        # This completely filters out the impact of an isolated giant climbing spike.
        robust_max_slope = np.percentile(np.abs(y_der), 90)
        slope_threshold = max(mean_end_slope + 3.0 * std_end_slope, 0.08 * robust_max_slope)

        # Trace backward from the end point
        suffix_start = len(after_pinch) - 1
        while suffix_start > 0 and np.abs(y_der[suffix_start]) <= slope_threshold:
            suffix_start -= 1

        # Safeguard to prevent tracing back into the pinch-off region
        if suffix_start <= 2:
            suffix_start = len(after_pinch) - 1

        sat_idx_scan = pinch_off_pos + suffix_start
        idx_sat = int(scan_indices[sat_idx_scan])
        
        sat_voltage = x1[idx_sat]
        sat_current = y1[idx_sat]
        sat_threshold = y1[idx_sat]
        sat_plateau_start = sat_voltage

    # --- Fit sigmoids ---
    params, popt, pcov = fit_to_function(x1, y1, sigmoid, print_results=False)

    # --- Extract key points ---
   
    A, B, V0, dV = popt

    y_fit = sigmoid(x1, *popt)

    # --- Plot data ---

    if plot_results:

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(x1, y1, '-', color='C0', linewidth=2, label='I ($V_{gate}$)')
        ax.scatter(pinch_off_voltage, pinch_off_current, color='red', s=100, zorder=5, label='Pinch-off Point')
        ax.scatter(sat_voltage, sat_current, color='green', s=100, zorder=5, label='Saturation Point')
        ax.plot(x1, y_fit, '--', color='red', linewidth=2, label='Fitted Sigmoid')

        if debug == True:

            if sat_plateau_start is not None and sat_threshold is not None:
                ax.axhline(sat_threshold, color='tab:green', linestyle='--', linewidth=1.25, alpha=0.85, label='Saturation Threshold')
                ax.axvspan(sat_plateau_start, x1[scan_indices[-1]], color='tab:green', alpha=0.12)
                ax.scatter(sat_plateau_start, sat_threshold, color='tab:green', marker='x', s=80, zorder=6, label='Saturation Start')

        ax.legend(fontsize=20, frameon=False, loc='upper left')

        # --- Double-sided arrows showing full range (swapped positions) ---

        # Define arrow y-positions (swap positions)

        y_arrow1 = ax.get_ylim()[1] + 0.05  # Device 1 arrow ABOVE
        # y_arrow2 = ax.get_ylim()[0] - 0.01  # Device 2 arrow BELOW

        # Device 1 arrow (now above)
        
        ax.annotate(
            '', xy=(sat_voltage, y_arrow1), xytext=(pinch_off_voltage, y_arrow1),
            arrowprops=dict(arrowstyle='<->', color='C0', lw=3.0, shrinkA=0, shrinkB=0),
            annotation_clip=False
        )
        ax.text((sat_voltage + pinch_off_voltage)/2, y_arrow1 - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                s='', color='C0', ha='center', va='top', fontsize=20)

        # --- Characteristic vertical lines extending exactly to the data points ---

        y_pinch1 = y1[np.where(x1 == pinch_off_voltage)][0]
        y_sat1   = y1[np.where(x1 == sat_voltage)][0]

        for color, po, sat, label, y_arrow, direction, y_pinch, y_sat in [
            # Device 1 → arrow above, extend down to data
            ('C0', pinch_off_voltage, sat_voltage, 'Device 1', y_arrow1, 'down', y_pinch1, y_sat1)
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

        ax.set_xlabel(r'V$_{gate}$ (V)', fontsize=35)
        ax.set_ylabel('I (nA)', fontsize=35)

        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
        ax.tick_params(direction='in', length=5, width=1.2, labelsize=18, top=True, right=True)

        xticks_span = np.linspace(x1.min(), x1.max(), 5)

        ax.set_xticks(xticks_span)
        ax.set_xticklabels([f'{xticks_span[0]:.2f}', '', f'{xticks_span[2]:.2f}', '', f'{xticks_span[-1]:.2f}'], fontsize=25)

        yticks_span = np.linspace(y1.min(), y1.max(), 5)

        ax.set_yticks(yticks_span)
        ax.set_yticklabels([f'{yticks_span[0]:.2f}', '', '', '', f'{yticks_span[-1]:.3f}'], fontsize=25)

        # Extend y-limits slightly to make space for arrows

        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])

        plt.tight_layout()

        filepath = os.path.join(filepath, filename)
        fig.savefig(filepath, dpi = 'figure', bbox_inches='tight')

        plt.close(fig)

    # --- Print summary ---

    """print(f"  Saturation Voltage: {sat_voltage:.3f} V")
    print(f"  Midpoint Voltage:   {V0:.3f} V")
    print(f"  Pinch-off Voltage:  {pinch_off_voltage:.3f} V\n") """

    voltage_window = (pinch_off_voltage, sat_voltage)

    return voltage_window, fig

def extract_max_conductance_points(x_data: np.array,
                                   y_data: np.array,
                                   peak_height: list[float] = [None, None],
                                   peak_prominence: list[float] = [None, None],
                                   peak_width: list[float] = [None, None]
                                   ):
    """Analyze current data to identify the largest conductance features.

    This function plots the current and its derivative, then highlights
    the most extreme conductance peaks and valleys.
    """

    x1 = np.array(x_data)
    y1 = np.array(y_data)

    # Now, we calculate the derivative and replot

    dIdV = np.gradient(y1, x1)

    if peak_height == [None, None]:
        peak_height = [0.25 * np.max(dIdV), 0.25 * np.max(dIdV)]
    if peak_prominence == [None, None]:
        peak_prominence = [0.3 * np.max(dIdV), 0.3 * np.max(dIdV)]


    # --- Find two largest and two smallest conductance points (positive + negative extremes) ---

    peak_idx_pos, _ = signal.find_peaks(dIdV, height = peak_height[0], prominence = peak_prominence[0], width=peak_width[0])
    peak_idx_neg, test_props = signal.find_peaks(-dIdV, height = peak_height[1], prominence = peak_prominence[1], width=peak_width[1])

    # for idx, p in enumerate(peak_idx_neg):
    #     # If the coordinate is near your dip (around V_P = 1.35 V)
    #     if 1.30 < x1[p] < 1.39:
    #         print(f"Negative Dip found at V_P = {x1[p]:.3f} V")
    #         print(f" -> Measured Prominence: {test_props['prominences'][idx]:.2f}")
    #         print(f" -> Measured Width:      {test_props['widths'][idx]:.2f}")

    peak_idx = np.sort(np.concatenate([peak_idx_pos, peak_idx_neg]))

    # Extract the corresponding data points
    x_top = x1[peak_idx]
    I_top = y1[peak_idx]
    G_top = dIdV[peak_idx]

    max_idx = peak_idx[np.argmax(dIdV[peak_idx])]
    min_idx = peak_idx[np.argmin(dIdV[peak_idx])]
    
    I_max = y1[max_idx]
    I_min = y1[min_idx]
    G_max = dIdV[max_idx]
    G_min = dIdV[min_idx]

    best_sens_pts = [(x1[max_idx], I_max), (x1[min_idx], I_min)]

    # Create two subplots that share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6))

    # --- Top panel: Current ---
    ax1.plot(x1, y1, color='#2c5aa0', linewidth=1)
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

    # --- Bottom panel: Conductance ---
    ax2.plot(x1, dIdV, color='#2c5aa0', linewidth=1)
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

    # --- Create the connection line ---
    for i in range(len(x_top)):
        con = ConnectionPatch(
            xyA=(x_top[i], I_top[i]), coordsA=ax1.transData,
            xyB=(x_top[i], G_top[i]), coordsB=ax2.transData,
            color='#FF5500', linestyle='--', linewidth=0.7
        )
        fig.add_artist(con)

    # --- Create a custom legend entry (hollow circle) ---
    legend_marker = mlines.Line2D([], [], color='#FF5500', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='High Sensitivity Points')
    
    legend_marker_2 = mlines.Line2D([], [], color='#01FF05', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='Best Sensitivity Points')

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

    ax1.legend(handles=[legend_marker, legend_marker_2], loc='upper left', fontsize=16, frameon=False)

    # --- Adjust layout ---
    plt.subplots_adjust(hspace=0.40)
    plt.show()

    return best_sens_pts, G_top

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
                          debug: bool = False,
                          plot_results: bool = True):
    
    """
    Find working-point lines in a 2D barrier sweep image.

    This function converts raw barrier voltage and current data into an image,
    applies ridge detection and Hough transform filtering, and returns the
    extracted working-point lines that correspond to relevant device ridges.
    """

    # We start by ensuring our inputs are numpy arrays

    lb_data = np.array(lb_data)
    rb_data = np.array(rb_data)
    current_data = np.array(current_data)
    barrier_pinch_offs = np.array(barrier_pinch_offs)
    device_type = 'hole'

    # 2. Establish uniform coordinate grids
    # (Assumes original data represents a regular mesh grid)
    ux = np.unique(lb_data)
    uy = np.unique(rb_data)

    # Define your clipping thresholds here (adjust values as needed)
    lb_min, lb_max = barrier_pinch_offs[0], ux.max()  # Replace with specific limits if desired
    rb_min, rb_max = barrier_pinch_offs[1], uy.max()  # Replace with specific limits if desired

    # 3. Create a boolean mask matching the original 1D data structure
    clip_mask = (
        (lb_data >= lb_min) & (lb_data <= lb_max) & 
        (rb_data >= rb_min) & (rb_data <= rb_max)
    )

    # 4. Apply the clipping mask to all arrays
    lb_data = lb_data[clip_mask]
    rb_data = rb_data[clip_mask]
    current_data = current_data[clip_mask]

    # 5. Handle polarity and device typing
    if current_data[0] < 0:
        current_data = -current_data

    if np.average(lb_data) > 0 and np.average(rb_data) > 0:
        current_data = np.flip(current_data, axis=None)
        device_type = 'electron'

    # 6. Calculate new dimensions based on unique clipped values
    nx_new = len(np.unique(lb_data))
    ny_new = len(np.unique(rb_data))

    # 7. Reshape the 1D clipped data into a 2D grid
    if current_data.ndim == 1:
        # Verify the clipped size matches the expected 2D grid dimensions
        if current_data.size == nx_new * ny_new:
            current_data = current_data.reshape((ny_new, nx_new))
        else:
            raise ValueError("Clipped data size does not form a perfect rectangular grid.")

    ny, nx = current_data.shape

    # Here, we define the voltage ranges

    lb_voltages = np.linspace(lb_data.min(), lb_data.max(), nx)
    rb_voltages = np.linspace(rb_data.min(), rb_data.max(), ny)
    
    logger.info("calculation starting...")

    # ---------- Gradient Calculation and Ridge Detection ----------
    

    # Now, compute the gradient and the log of the gradient

    Gx, Gy = np.gradient(current_data)
    G = (1.0 / np.sqrt(2.0)) * np.sqrt(Gx**2 + Gy**2)

    g_lo, g_hi = np.percentile(G, [2, 98])
    G_clipped = np.clip(G, g_lo, g_hi)
    G_scaled = (G_clipped - g_lo) / (g_hi - g_lo)

    G_uint = (255 * G_scaled).astype(np.uint8)

    low = int(0.10 * 255)    # discard noise
    high = int(0.35 * 255)  # discard strongest boundaries

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
    
    # pixel index arrays (needed for interpolation)
    x_index_arr = np.arange(nx)
    y_index_arr = np.arange(ny)

    # Selects mid point of x and y axes
    x_idx_mid = np.interp((lb_voltages.min() + lb_voltages.max()) / 2, lb_voltages, x_index_arr)
    y_idx_mid = np.interp((rb_voltages.min() + rb_voltages.max()) / 2, rb_voltages, y_index_arr)
    x_idx_mid = int(np.clip(x_idx_mid, 0, nx - 1))
    y_idx_mid = int(np.clip(y_idx_mid, 0, ny - 1))
    
    ridge_masked = np.zeros_like(ridge_filtered)
    
    if device_type == 'electron':
        # Electron: analyze bottom-left
        ridge_masked[:y_idx_mid, :x_idx_mid] = ridge_filtered[:y_idx_mid, :x_idx_mid]
    else:  # hole
        # Hole: analyze top-right
        ridge_masked[y_idx_mid:, x_idx_mid:] = ridge_filtered[y_idx_mid:, x_idx_mid:]

    # From these edges, we detect lines using a probabilistic hough transform.
    # Use a slightly lower threshold and tune the minimum required segment length so long bottom-left lines are prioritized.
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

    # roi = None
    # roi_offset = (0, 0)
    # if device_type == 'electron' and x_idx_mid > 5 and y_idx_mid > 5:
    #     roi = ridge_masked[y_idx_mid:, :x_idx_mid]
    #     roi_offset = (0, y_idx_mid)
    # elif device_type == 'hole' and x_idx_mid < nx - 5 and y_idx_mid < ny - 5:
    #     roi = ridge_masked[:y_idx_mid, x_idx_mid:]
    #     roi_offset = (x_idx_mid, 0)

    # if roi is not None and roi.size > 0:
    #     extra_lines = transform.probabilistic_hough_line(
    #         roi,
    #         threshold=max(5, hough_threshold - 3),
    #         line_length=max(8, int(minLineLength * 0.12)),
    #         line_gap=max(1, int(maxLineGap * 0.05))
    #     )
    #     for p0, p1 in extra_lines:
    #         lines.append((
    #             (p0[0] + roi_offset[0], p0[1] + roi_offset[1]),
    #             (p1[0] + roi_offset[0], p1[1] + roi_offset[1])
    #         ))

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

    # if not filtered_lines and roi is not None and roi.size > 0:
    #     for alt_img in [band_passed, G_uint, ridge_norm]:
    #         alt_roi = alt_img[:y_idx_mid, :x_idx_mid] if device_type == 'electron' else alt_img[y_idx_mid:, x_idx_mid:]
    #         extra_lines = transform.probabilistic_hough_line(
    #             alt_roi,
    #             threshold=max(4, hough_threshold - 4),
    #             line_length=max(8, int(minLineLength * 0.12)),
    #             line_gap=max(1, int(maxLineGap * 0.05))
    #         )
    #         for p0, p1 in extra_lines:
    #             lines.append((
    #                 (p0[0] + roi_offset[0], p0[1] + roi_offset[1]),
    #                 (p1[0] + roi_offset[0], p1[1] + roi_offset[1])
    #             ))

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

    # Previously, we limited analysis to the bottom left quadrant, here we're defining the voltage range for that quadrant
    # Using the pinch-off voltages from barrier_pinch_offs parameter

    # enlarge region by adding 0.1 V to pinch-off values
    lb_mid_volt = (lb_voltages.min() + lb_voltages.max()) / 2  # First value: x-axis (left/bottom gate)
    rb_mid_volt = (rb_voltages.min() + rb_voltages.max()) / 2  # Second value: y-axis (right/bottom gate)

    perp_candidates = []
    perp_traces_for_plot = []

    perp_length_pixels = max(40, int(min(nx, ny) * 0.5))
    perp_samples = 400
    smooth_sigma = 2.0

    # Now, for each filtered line, we define a line perpendicular to it, then find the peaks in current along them

    logger.info("finding traces...")

    for x1, y1, x2, y2 in filtered_lines:
        
        # midpoints
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        # distances
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
            continue

        # perpendicular direction
        pxu, pyu = -dy / L, dx / L

        # length along the perpendicular lines in pixel space
        t = np.linspace(-perp_length_pixels / 2,
                        perp_length_pixels / 2,
                        perp_samples)

        # Limiting the values of the array to the bottom left quadrant
        samp_x = np.clip(mx + pxu * t, 0, nx - 1)
        samp_y = np.clip(my + pyu * t, 0, ny - 1)
        trace_id = len(perp_traces_for_plot)

        # defining current
        trace = map_coordinates(
            current_data,
            [samp_y, samp_x],
            order=3,
            mode="reflect"
        )

        # smooth the trace
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

        # find local maxima of current
        noise_sigma = 1.4826 * np.median(np.abs(conductance - np.median(conductance)))
        prominence_thresh = 4.0 * noise_sigma
        peaks, _ = signal.find_peaks(conductance,
                                     prominence=prominence_thresh,
                                     distance=15)

        if len(peaks) == 0:
            continue

        # defining the the maxima in voltage space from pixel space
        peak_idx = peaks
        px = samp_x[peak_idx]
        py = samp_y[peak_idx]

        vx = np.interp(px, x_index_arr, lb_voltages)
        vy = np.interp(py, y_index_arr, rb_voltages)

        # restrict to red zone based on device type
        if device_type == 'electron':
            valid = (vx < lb_mid_volt) & (vy < rb_mid_volt)
        else:  # hole
            valid = (vx > lb_mid_volt) & (vy > rb_mid_volt)
        peak_idx = peak_idx[valid]
        px = px[valid]
        py = py[valid]
        vx = vx[valid]
        vy = vy[valid]

        if len(peak_idx) == 0:
            continue

        # compiling data into trace info
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

    # First, we sort the points in order of increasing current
    
    logger.info("selecting bias points...")

    logger.info(f"filtered_lines: {len(filtered_lines)}")
    logger.info(f"perp_candidates: {len(perp_candidates)}")

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

    # Compute selected working points. For Triple Dot, shift each point 0.1 V in the
    # opposite yellow trace direction instead of perpendicular to it.
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
                    shift_vec = -unit_dir * 0.1
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

    # Now, we overlay perpendicular traces (strictly clipped to BL quadrant)
    
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

    if plot_results:

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

        # Set axis limits
        ax.set_xlim(lb_data.min(), lb_data.max())
        ax.set_ylim(rb_data.min(), rb_data.max())

        # Round ticks
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

        # Create horizontal colorbar above the axes
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
        
        else:  # hole
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
            # compute voltage coordinates
            v1x = np.interp(x1, x_index_arr, lb_voltages)
            v1y = np.interp(y1, y_index_arr, rb_voltages)
            v2x = np.interp(x2, x_index_arr, lb_voltages)
            v2y = np.interp(y2, y_index_arr, rb_voltages)
            
            # Uncomment below to see the detected Hough lines
            # ax.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2) 

        # Perpendicular traces and peaks
        dot_tuning_shift = 0.1 if str(DotTuning).strip().lower() == 'triple dot' else 0.0

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

                        # compute a local tangent along the yellow trace and shift along it
                        if 1 <= i < (len(vx) - 1):
                            ddx = float(vx[i + 1]) - float(vx[i - 1])
                            ddy = float(vy[i + 1]) - float(vy[i - 1])
                        else:
                            # fallback to using the chosen block endpoints
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
                                # draw a cyan debug line showing the shift direction
                                ax.plot([vx_p, sx], [vy_p, sy], c='cyan', lw=1.5, alpha=0.9, zorder=9)
                            except Exception:
                                pass

                        # shifted star (exactly dot_tuning_shift along the yellow perp)
                        ax.scatter(sx, sy, s=200, c='white', marker='*', edgecolors='black', zorder=10)
                        shifted_points.append((sx, sy))

                        if vx_p == closest_candidate[0] and vy_p == closest_candidate[1]:
                            best_shifted_point = (sx, sy)
                            green_circle = True

                        if green_circle:
                            # hollow green  circle at original peak position for best candidate
                            ax.scatter(vx_p, vy_p, s=80, c='none', edgecolors='white', linewidths=1.5, zorder=11)
                            green_circle = False
                        else:
                            # hollow red circle at original peak position
                            ax.scatter(vx_p, vy_p, s=80, c='none', edgecolors='red', linewidths=1.5, zorder=11)

                        # arrow from original to shifted star
                        ax.annotate('', xy=(sx, sy), xytext=(vx_p, vy_p),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=1.0), zorder=12)
                except Exception:
                    pass

        ax.set_box_aspect(0.775)

        # If any shifted points lie outside current axis limits, expand limits slightly
        # if len(shifted_points) > 0:
        #     sx_vals = [p[0] for p in shifted_points]
        #     sy_vals = [p[1] for p in shifted_points]
        #     xmin, xmax = ax.get_xlim()
        #     ymin, ymax = ax.get_ylim()
        #     pad_x = 0.02 * (xmax - xmin) if (xmax - xmin) != 0 else 0.01
        #     pad_y = 0.02 * (ymax - ymin) if (ymax - ymin) != 0 else 0.01
        #     new_xmin = min(xmin, min(sx_vals) - pad_x)
        #     new_xmax = max(xmax, max(sx_vals) + pad_x)
        #     new_ymin = min(ymin, min(sy_vals) - pad_y)
        #     new_ymax = max(ymax, max(sy_vals) + pad_y)
        #     ax.set_xlim(new_xmin, new_xmax)
        #     ax.set_ylim(new_ymin, new_ymax)

        filepath = os.path.join(filepath, filename)
        fig.savefig(filepath, dpi = 'figure', bbox_inches='tight')

        logger.info("Figure saved!")

        plt.close(fig)

        # These are 1D perpendicular trace plots

        if debug:
            for tr in perp_traces_for_plot:
                s = tr["s"]
                I = tr["trace"]                 # smoothed current
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

        # Now, we detect lines from the normalized Gradient

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

        # Now, we detect lines from the Normalized Ridges

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
        return best_shifted_point, shifted_points, perp_traces_for_plot, fig
    elif DotTuning == 'SET':
        return best_shifted_point, perp_bias_points, perp_traces_for_plot, fig

def extract_tunnel_barrier_latching(dp_data: np.array,
                                    tb_data: np.array,
                                    current_data: np.array,
                                    peak_height: list[float] = [None, None],
                                    peak_prominence: list[float] = [None, None],
                                    peak_width: list[float] = [None, None]
                                    ):
    """Analyze current data to identify if latching is occuring during dot-lead tuning

    This function plots the derivative of the current and analyzes the peaks to see if latching is occuring.
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
    # fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=False)
    # axes = axes.flatten()  # Flatten grid into a 1D list for easy looping

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

        # # 1. Compute the derivative (dI/d_dp) using the spatial coordinate grid spacing
        # derivative_vals = np.gradient(z_vals, x_vals)
        
        # # 2. Plot the derivative line to the corresponding grid cell
        # axes[i].plot(x_vals, derivative_vals, color='tab:purple', linewidth=1.5)
        
        # # Place LaTeX formatted text overlay relative to the subplot viewport bounds
        # axes[i].text(
        #     0.05, 0.93, 
        #     rf"$V_B$ = {trace_y_values[i]:.3f}", 
        #     transform=axes[i].transAxes, 
        #     fontsize=10, 
        #     verticalalignment='top',
        #     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        # )

    # # Clean up empty subplots if any
    # for j in range(num_traces, len(axes)):
    #     fig.delaxes(axes[j])

    # # Add global canvas labels and updated main title
    # fig.supxlabel(r"$V_P$ (V)", fontsize=12)
    # fig.supylabel("G (nS)", fontsize=12)
    # fig.suptitle("Conductance Traces Extracted Along the Tunnel Barrier", fontsize=14, fontweight='bold')
    
    # plt.tight_layout()
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

def extract_lever_arms(data: pd.DataFrame,
                       plot_process: bool = False) -> dict:
    """Estimate lever arms from a 2D transconductance map.

    This function pivots the input dataframe to a grid, computes the
    gradient in the current data, applies filtering, and optionally plots
    the intermediate transconductance results.
    """
    
    # Load in data and separate 
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

