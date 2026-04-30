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
import signal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import ConnectionPatch

from scipy.optimize import curve_fit
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
  
def logarithmic(x, a, b, x0, y0):
    return a * np.log(b*(x-x0)) + y0

def exponential(x, a, b, x0, y0):
    return a * np.exp(b * (x-x0)) + y0

def sigmoid(x, a, b, x0, y0):
    return a/(1+np.exp(b * (x-x0))) + y0

def linear(x, m, b):
    return m * x + b         

def relu(x, a, x0, b):
    return np.maximum(0, a * (x - x0) + b)

def fit_to_function(x_data, 
                    y_data, 
                    function: Callable):
    
    popt, pcov = curve_fit(function, x_data, y_data)
    perr = np.sqrt(np.diag(pcov))

    params = list(inspect.signature(function).parameters.keys())[1:]

    for name, val, err in zip(params, popt, perr):
        print(f"{name} = {val:.3f} ± {err:.3f}")

    return params, popt, pcov

def pinch_off_curve_ranges(x_data, y_data):

    # --- Data definitions ---
    
    x1 = np.array(x_data)
    y1 = np.array(y_data)

    # --- Fit sigmoids ---
    
    p0_1 = [min(y1), max(y1), np.median(x1), 0.05]
    params, popt, pcov = fit_to_function(x1, y1, sigmoid)

    # --- Extract key points ---
   
    A, B, V0, dV = popt

    # Define the characteristic voltage range as (V0 ± √8 * dV)
   
    range_factor = np.sqrt(8)

    pinch_off = V0 - range_factor * dV
    sat       = V0 + range_factor * dV
    
    # --- Plot data ---
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x1, y1, '-', color='C0', linewidth=2, label='I ($V_{B1}$)')
    ax.legend(fontsize=24, frameon=False, loc='upper right')

    # --- Double-sided arrows showing full range (swapped positions) ---

    # Define arrow y-positions (swap positions)

    y_arrow1 = ax.get_ylim()[1] + 0.05  # Device 1 arrow ABOVE
    y_arrow2 = ax.get_ylim()[0] - 0.01  # Device 2 arrow BELOW

    # Device 1 arrow (now above)
    
    ax.annotate(
        '', xy=(sat, y_arrow1), xytext=(pinch_off, y_arrow1),
        arrowprops=dict(arrowstyle='<->', color='C0', lw=3.0, shrinkA=0, shrinkB=0),
        annotation_clip=False
    )
    ax.text((sat + pinch_off)/2, y_arrow1 - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            s='', color='C0', ha='center', va='top', fontsize=20)

    # --- Characteristic vertical lines extending exactly to the data points ---

    # Compute corresponding y-values from the *fitted sigmoid* (smooth, reliable)

    y_pinch1 = sigmoid(pinch_off, *popt)
    y_sat1   = sigmoid(sat, *popt)

    for color, po, sat, label, y_arrow, direction, y_pinch, y_sat in [
        # Device 1 → arrow above, extend down to data
        ('C0', pinch_off, sat, 'Device 1', y_arrow1, 'down', y_pinch1, y_sat1)
    ]:
        if direction == 'up':
            # Extend upward from arrow to the y-values of the fitted curve
            ax.vlines(po, ymin=y_arrow, ymax=y_pinch - 0.01, colors=color, linestyles='--', alpha=0.6)
            ax.vlines(sat, ymin=y_arrow, ymax=y_sat - 0.025, colors=color, linestyles='--', alpha=0.6)
        else:
            # Extend downward from arrow to the y-values of the fitted curve
            ax.vlines(po, ymin=y_pinch + 0.02, ymax=y_arrow, colors=color, linestyles='--', alpha=0.6)
            ax.vlines(sat, ymin=y_sat - 0.015, ymax=y_arrow, colors=color, linestyles='--', alpha=0.6)

    # --- Overlay fitted sigmoid curves ---

    V_fit = np.linspace(x1.min(), x1.max(), 500)

    # Fitted curves for each device

    y_fit1 = sigmoid(V_fit, *popt)
    y_fit2 = sigmoid(V_fit, *popt)

    # --- Labels and formatting ---

    ax.set_xlabel(r'V$_{B1}$, V$_{B2}$ (V)', fontsize=45)
    ax.set_ylabel('I (nA)', fontsize=55)

    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
    ax.tick_params(direction='in', length=5, width=1.2, labelsize=18, top=True, right=True)

    ax.set_xticks([-2.5, -2.0, -1.5, -1.0, -0.5])
    ax.set_xticklabels(['-2.5', '', '', '', '-0.5'], fontsize=45)

    ax.set_yticks([0.0, 0.4, 0.8, 1.2])
    ax.set_yticklabels(['0.0', '', '', '1.2'], fontsize=45)

    # Extend y-limits slightly to make space for arrows

    ax.set_ylim(-0.1, ax.get_ylim()[1])

    plt.tight_layout()
    plt.show()

    # --- Print summary ---

    print(f"  Saturation Voltage: {sat:.3f} V")
    print(f"  Midpoint Voltage:   {V0:.3f} V")
    print(f"  Pinch-off Voltage:  {pinch_off:.3f} V\n")

    pass

def extract_max_conductance_points(self, x_data, y_data):

    x1 = np.array(x_data)
    y1 = np.array(y_data)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(x1, y1)
    plt.xlabel('V_P (V)')
    plt.ylabel('Current (nA)')
    plt.title('Coulomb Blockade For P-Type Device')

    plt.show()

    # Now, we calculate the derivative and replot

    dIdV = np.gradient(y1, x1)

    posdIdV = abs(dIdV)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(x1, posdIdV)
    plt.xlabel('V_P (mV)')
    plt.ylabel('Conductance (nS)')
    plt.title('Conductance Peaks for P-Type Device')

    plt.show()

    # --- Find two largest and two smallest conductance points (positive + negative extremes) ---

    # Get indices of top 2 positive conductance values
    top_idx_pos = np.argsort(dIdV)[-2:]

    # Get indices of bottom 2 negative conductance values
    top_idx_neg = np.argsort(dIdV)[:2]

    # Combine them and sort by x-position for consistent plotting
    top_idx = np.sort(np.concatenate([top_idx_pos, top_idx_neg]))

    # Extract the corresponding data points
    x_top = x1.iloc[top_idx]
    I_top = y1.iloc[top_idx]
    G_top = dIdV[top_idx]

    # Create two subplots that share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 6))

    # --- Top panel: Current ---
    ax1.plot(x1, y1, color='#2c5aa0', linewidth=1)
    ax1.scatter(x_top, I_top, facecolors='none', edgecolors='#FF5500', s=100, linewidths=2, zorder=5, label='High Sensitivity Points')
    ax1.set_ylabel('I (nA)', fontsize=45)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(min(x1) - 0.05, max(x1) + 0.05)
    ax1.tick_params(labelbottom=True)

    # --- Bottom panel: Conductance ---
    ax2.plot(x1, posdIdV, color='#2c5aa0', linewidth=1)
    ax2.scatter(x_top, G_top, facecolors='none', edgecolors='#FF5500', s=100, linewidths=2, zorder=5, label='Max G')
    ax2.set_xlabel(r'$V_P$ (V)', fontsize=45)
    ax2.set_ylabel('G (nS)', fontsize=45)
    ax2.set_xlim(min(x1) - 0.05, max(x1) + 0.05)

    # --- Create the connection line ---
    con = ConnectionPatch(
        xyA=(x_top, I_top), coordsA=ax1.transData,
        xyB=(x_top, G_top), coordsB=ax2.transData,
        color='#FF5500', linestyle='--', linewidth=0.7
    )
    fig.add_artist(con)

    # --- Create a custom legend entry (hollow circle) ---
    legend_marker = mlines.Line2D([], [], color='#FF5500', marker='o',
                                markerfacecolor='none', markersize=10,
                                linewidth=0, label='High Sensitivity Points')

    # --- Custom tick labels: only min and max shown ---

    # Get existing ticks (so tick marks stay)
    for ax in [ax1, ax2]:

        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, top=True, right=True)
        ax.tick_params(direction='in', length=5, width=1.2, labelsize=20, top=True, right=True)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        
    ax1.set_xticks([-0.4, 0.0, 0.4])
    ax1.set_xticklabels(['-0.4', '0.0', '0.4'], fontsize=25)

    ax1.set_yticks([0.0, 0.15])
    ax1.set_yticklabels(['0.0', '0.15'], fontsize=25)

    ax2.set_xticks([-0.4, 0.0, 0.4])
    ax2.set_xticklabels(['-0.4', '0.0', '0.4'], fontsize=25)

    ax2.set_yticks([0.0, 10])
    ax2.set_yticklabels(['0', '10'], fontsize=25)

    ax1.legend(handles=[legend_marker], loc='upper left', fontsize=16, frameon=False)

    # --- Adjust layout ---
    plt.subplots_adjust(hspace=0.40)
    plt.show()

def extract_bias_point(
        lb_data: np.array,
        rb_data: np.array,
        current_data: np.array,
        minAngleDeg: float = -55,
        maxAngleDeg: float = -35,
        minLineLength: int = 50,
        maxLineGap: int = 250,
        debug: bool = False,
        plot_results: bool = True) -> list[tuple]:

    # We start by ensuring our inputs are numpy arrays

    lb_data = np.array(lb_data)
    rb_data = np.array(rb_data)
    current_data = np.array(current_data)

    # Now, we reshape the data into an array

    if current_data.ndim == 1:
        nx = len(np.unique(lb_data))
        ny = len(np.unique(rb_data))
        current_data = current_data.reshape((ny, nx))

    ny, nx = current_data.shape

    # Here, we define the voltage ranges

    lb_voltages = np.linspace(lb_data.min(), lb_data.max(), nx)
    rb_voltages = np.linspace(rb_data.min(), rb_data.max(), ny)

    
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
    ridge_filtered = ridge_norm > 0.20

    # Now, we limit our analysis to the bottom left-quadrant

    ridge_masked = np.zeros_like(ridge_filtered)
    ridge_masked[:ny // 2, :nx // 2] = ridge_filtered[:ny // 2, :nx // 2]

    # From these edges, we detect lines using a probabilistic hough transform 

    lines = transform.probabilistic_hough_line(
        ridge_masked,
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

    # Previously, we limited analysis to the bottom left quadrant, here we're defining the voltage range for that quadrant

    lb_mid_volt = 0.5 * (lb_data.min() + lb_data.max())
    rb_mid_volt = 0.5 * (rb_data.min() + rb_data.max())

    perp_candidates = []
    perp_traces_for_plot = []

    perp_length_pixels = max(40, int(min(nx, ny) * 0.5))
    perp_samples = 400
    smooth_sigma = 2.0

    x_index_arr = np.arange(nx)
    y_index_arr = np.arange(ny)

    # Now, for each filtered line, we define a line perpendicular to it, then find the peaks in current along them

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
        noise_sigma = 1.4826 * np.median(np.abs(trace_smooth - np.median(trace_smooth)))
        prominence_thresh = 4.0 * noise_sigma
        peaks, _ = signal.find_peaks(trace_smooth,
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

        # bottom-left quadrant restriction
        valid = (vx < lb_mid_volt) & (vy < rb_mid_volt)
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

    if not perp_candidates:
        return []


    # ---------- Selecting Final Bias Points ----------


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

    perp_traces_for_plot = [
        tr for tr in perp_traces_for_plot
        if tr["trace_id"] in selected_trace_ids
    ]

    # Now, we overlay perpendicular traces (strictly clipped to BL quadrant)
    
    for tr in perp_traces_for_plot:
        vx = np.interp(tr["px"], x_index_arr, lb_voltages)
        vy = np.interp(tr["py"], y_index_arr, rb_voltages)

        in_quad = (vx < lb_mid_volt) & (vy < rb_mid_volt)
        if not np.any(in_quad):
            continue

        idx = np.where(in_quad)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        blocks = np.split(idx, splits + 1)

        peak_idx = tr.get("peak_idx", None)
        chosen_block = None
        if peak_idx is not None:
            for b in blocks:
                if np.intersect1d(peak_idx, b).size > 0:
                    chosen_block = b
                    break
        if chosen_block is None:
            chosen_block = blocks[0]

        tr["chosen_block"] = chosen_block


    # ---------- Final Plotting ----------


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
        
        ax.set_xticks([0.65, 0.85])
        ax.set_yticks([0.60, 0.70])
        ax.set_xticklabels(["0.65", "0.85"], fontsize=40)
        ax.set_yticklabels(["0.60", "0.70"], fontsize=40)

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
        ax.set_xlabel(r'V$_{B2}$ (V)', fontsize=45, labelpad = -35)
        ax.set_ylabel(r'V$_{B1}$ (V)', fontsize=45)

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
        cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
        cbar.set_ticklabels(['0', '', '', '', '1'])
        cbar.ax.tick_params(labelsize=30, direction="in", length=6)

        cbar.ax.minorticks_on()

        cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # Style minor ticks
        cbar.ax.tick_params(
            which="minor",
            direction="in",
            length=4,
            width=1.0
        )

        # Block Boundary

        # Top side: horizontal line from left-mid to right-mid
        ax.plot(
            [lb_mid_volt, lb_data.min()],  # x: left → right
            [rb_mid_volt, rb_mid_volt],    # y constant at top
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        # Right side: vertical line from bottom-mid to top-mid
        ax.plot(
            [lb_mid_volt, lb_mid_volt],    # x constant at right
            [rb_data.min(), rb_mid_volt],  # y: bottom → top
            linestyle='--',
            color='red',
            linewidth=1.2,
            alpha=0.9
        )

        rect = Rectangle(
        (lb_data.min(), rb_data.min()),                 # bottom-left corner
        lb_mid_volt - lb_data.min(),                   # width
        rb_mid_volt - rb_data.min(),                   # height
        facecolor='red',
        alpha=0.2,                          # set opacity here (1.0 = fully opaque)
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
            ax.plot([v1x, v2x], [v1y, v2y], c='black', lw=1.2) 

        # Perpendicular traces and peaks
        for tr in perp_traces_for_plot:
            idx = tr.get("chosen_block", None)
            if idx is None or len(idx) == 0: continue
            vx = np.interp(tr["px"], np.arange(nx), lb_voltages)
            vy = np.interp(tr["py"], np.arange(ny), rb_voltages)
            ax.plot(vx[idx], vy[idx], c='yellow', lw=1.5, alpha=0.9)
            valid_peaks = np.intersect1d(tr.get("peak_idx", []), idx)
            ax.scatter(vx[valid_peaks], vy[valid_peaks], s=150, c='white',
                    marker='*', edgecolors='black', zorder=10)

        ax.set_box_aspect(0.775)

        plt.show()

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

                # Mark current peaks
                
                if len(peak_idx) > 0:
                    axs[0].scatter(
                        s[peak_idx],
                        I[peak_idx],
                        c="red",
                        s=40,
                        zorder=5,
                        label="Current peaks"
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
        plt.xlabel(r'V$_{B2}$ (V)', fontsize=45)
        plt.ylabel(r'V$_{B1}$ (V)', fontsize=45)
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

    return perp_bias_points, perp_traces_for_plot

def extract_lever_arms(data: pd.DataFrame,
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

