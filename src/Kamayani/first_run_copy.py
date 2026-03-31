# The following gates are assumed to be connected to the following channels in sim900:
# Channel 1: Top gate
# Channel 2: QPC bottom
# Channel 3: QPC top
# Channel 4: Exit gate
# Channel 5: Entrance gate
# Channel 6: Source-drain bias (currently set to zero)

########## VARIABLES - TO BE PUT BY THE USER ##########

# Enter the directory where you want the data to be saved. 
base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'

# Enter the three values of the qpc voltage. 
qpc_gate_voltages = [0.100]

# Enter the sensitivity of agilent.
agilent_sensitivity = 1e-7

# Enter the current value until which the V_ent and V_exit gates are symmetrically increased.
current_first_point = 0.9e-9

# Enter the current by manually checking the offset of the pre-amplifier. 
min_current_DC = 30.0e-13

# Enter the maximum voltage that cen be set up for Vent and Vexit. 
limiting_voltage = 1.1 

# agilent_sensitivity_RF = 1e-9

qpc_gate_voltages_RF = [0.100]

user_power_dBm = [0, 1]
######## I need to work starting from here.``
# Enter the step_size
step_size = 0.01

# Enter the step size for top gate.
step_top_gate = 0.1

# top_gate = 
# source_drain_bias = 

import pyvisa
from qcodes.instrument.visa import VisaInstrument
import qcodes as qc
from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
import time
import os
import sys
from datetime import datetime
from typing import List, Tuple
from typing import Sequence
from typing import Any
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

########## FUNCTIONS ###########

def reinitialize_instrument(name: str, driver, address: str, *args, **kwargs):
    """
    Safely reinitializes a QCoDeS instrument.

    Args:
        name (str): The name assigned to the instrument instance (e.g., "sim900").
        driver: The QCoDeS instrument driver class (e.g., SIM928).
        address (str): The VISA address (e.g., "GPIB0::3::INSTR").
        *args, **kwargs: Additional positional or keyword arguments to pass to the driver.
    
    Returns:
        instrument: The newly created instrument instance.
    """
    if name in locals():
        try:
            locals()[name].close()
        except Exception as e:
            print(f"Warning: Failed to close {name}: {e}")
    elif name in globals():
        try:
            globals()[name].close()
        except Exception as e:
            print(f"Warning: Failed to close {name}: {e}")

    if name in qc.Instrument._all_instruments:
        try:
            qc.Instrument.find_instrument(name).close()
        except Exception as e:
            print(f"Warning: Failed to close existing instrument handle {name}: {e}")

    # Reinitialize
    instrument = driver(name, address, *args, **kwargs)
    globals()[name] = instrument  # Add to global scope
    return instrument

def save_arrays_to_csv(
    filename: str,
    sim,
    **arrays: Sequence[float]
) -> None:
    """
    Assemble named 1D arrays into a table and write to CSV.

    Args:
        filename:  Path to the output CSV file.
        sim:       Your instrument controller with .get_voltage(slot).
        **arrays:  One or more keyword arguments where the key is the
                desired column name and the value is a sequence
                (e.g. list or numpy array) of numbers. All sequences
                must have the same length.

    Raises:
        ValueError: If the provided sequences do not all have the same length.
    """
    # Verify equal lengths
    lengths = [len(v) for v in arrays.values()]
    if not lengths:
        raise ValueError("No arrays provided to save.")
    if any(length != lengths[0] for length in lengths):
        raise ValueError(
            f"All arrays must have the same length; got lengths = {lengths}"
        )
    
    qpc_voltage = sim.get_voltage(3)

    # Build DataFrame and write CSV
    df = pd.DataFrame({name: list(values) for name, values in arrays.items()})
    filename_1 = f'{qpc_voltage:.3f}_{filename}_{timestamp}.csv'
    directory = os.path.join(base_dir, filename_1)
    df.to_csv(directory, index=False,float_format='%.3e')

    # 3) Scatter plot of the first two columns
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_label = df.columns[0]
    y_label = df.columns[1]

    # Read QPC gate voltage from channel 3
    legend_label = f"V_qpc = {qpc_voltage:.3f} V"

    plt.figure()
    plt.scatter(x, y, label=legend_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_2 = f'{qpc_voltage:.3f}_{filename}_{timestamp}_scatter.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()

def ground_device_sim(sim, slot: str, step: float = 0.1, delay: float = 0.5):
    V0 = sim.get_voltage(slot)
    if abs(V0) < 1e-12:
        return
    
    direction = -np.sign(V0)
    setpoints = np.arange(V0, 0, direction * step)

    for V in setpoints:
        sim.set_voltage(slot, float(V))
        time.sleep(delay)

    sim.set_voltage(slot, 0.0)
    time.sleep(delay)

def set_voltage_sim(sim, slot: str, target: float, step: float = 0.1, delay: float = 0.5):
    current = sim.get_voltage(slot)
    if abs(current - target) < 1e-12:
        return
    
    direction = -np.sign(current - target)
    setpoints = np.arange(current, target, direction * step)

    for V in setpoints:
        sim.set_voltage(slot, float(V))
        time.sleep(delay)

    sim.set_voltage(slot, target)
    time.sleep(delay)

    print(f'The voltage of the channel {slot} of sim900 is set to {target: .6f} V.')

def simultaneous_set_voltage_sim(sim, slot1: str, target1: float, slot2: str, target2: float, step: float = 0.01, delay: float = 0.5):

    V1_0 = sim.get_voltage(slot1)
    V2_0 = sim.get_voltage(slot2)

    if abs(V1_0 - target1) < 1e-12 and abs(V2_0 - target2) < 1e-12:
        return
    
    dir1 = np.sign(target1 - V1_0)
    dir2 = np.sign(target2 - V2_0)

    sp1 = np.arange(V1_0, target1, dir1 * step)
    sp2 = np.arange(V2_0, target2, dir2 * step)

    if sp1.size == 0 or sp1[-1] != target1:
        sp1 = np.append(sp1, target1)
    if sp2.size == 0 or sp2[-1] != target2:
        sp2 = np.append(sp2, target2)

    for v1, v2 in zip(sp1,sp2):
        sim.set_voltage(slot1, float(v1))
        sim.set_voltage(slot2, float(v2))
        time.sleep(delay)

    sim.set_voltage(slot1, target1)
    sim.set_voltage(slot2, target2)
    time.sleep(delay)

    print(f'The voltage of the channel {slot1} and {slot2} of sim900 is set to {target1: .6f} V and {target2: .6f} V.')

def set_voltage_until_current(sim, agilent, slot1: str, slot2: str, step: float, delay: float, current_target: float):
    """
    Increases voltage on two SIM slots until the Agilent current reading reaches `current_target`.
    Returns the voltage values (V1, V2) at which the current target is achieved or the cap (1.1 V) is hit.
    Logs full voltage and current history to a CSV.
    """
    qpc_voltage = sim.get_voltage(3)

    V1 = sim.get_voltage(slot1)
    V2 = sim.get_voltage(slot2)
    voltage = abs(agilent.volt())
    current = voltage * agilent_sensitivity 

    v1_history: List[float] = [V1]
    v2_history: List[float] = [V2]
    current_history: List[float] = [current]

    while current < current_target:
        # Check if voltage cap is reached
        if V1 >= limiting_voltage or V2 >= limiting_voltage:
            V1 = min(V1, limiting_voltage)
            V2 = min(V2, limiting_voltage)
            break

        V1 += step
        V2 += step
        sim.set_voltage(slot1, V1)
        sim.set_voltage(slot2, V2)
        time.sleep(delay)

        voltage = abs(agilent.volt())
        current = voltage * agilent_sensitivity

        v1_history.append(V1)
        v2_history.append(V2)
        current_history.append(current)

    # Save the voltage-current history
    data_first_point = pd.DataFrame({
        'V_slot1': v1_history,
        'V_slot2': v2_history,
        'I (A)': current_history
    })

    filename = f'{qpc_voltage:.3f}_data_first_point_{timestamp}.csv'
    directory = os.path.join(base_dir, filename)
    data_first_point.to_csv(directory, index=False, float_format='%.3e')

    # Final voltages when condition met
    V1_final = sim.get_voltage(slot1)
    V2_final = sim.get_voltage(slot2)

    return [V1_final, V2_final]

def sweep_gate_until_min_current(sim,
                                agilent,
                                constant_slot: str,
                                sweep_slot: str,
                                step: float,
                                delay: float,
                                min_current: float,
                                num_points: int
                                ) -> Tuple[List[float], List[float], float]:
    """
    Sweep the voltage on `sweep_slot` from its current value down in decrements of `step`
    until the measured current falls to or below `min_current`. After that,
    continue sweeping for `num_points` additional steps. The voltage on
    `constant_slot` remains fixed throughout.

    Returns the full voltage and current histories, plus the average of the
    voltages recorded during the final `num_points` sweep.

    Args:
        sim:             Instrument with .get_voltage(slot) and .set_voltage(slot, V).
        agilent:         DMM instance where agilent.volt() returns the measured voltage (V).
        constant_slot:   Channel index to hold constant.
        sweep_slot:      Channel index to sweep.
        step:            Voltage decrement per iteration (V).
        delay:           Seconds to wait after setting voltage before reading back.
        min_current:     Current threshold (A) to trigger the extra sweep.
        num_points:      Number of additional sweep points after threshold reached.

    Returns:
        vs_history:      List of all voltages read on `sweep_slot`.
        current_history: List of all corresponding currents (in A) at each step.
        avg_extra_vs:    Float — the average of the last `num_points` voltages.
    """
    vs_history: List[float] = []
    current_history: List[float] = []
    extra_vs: List[float] = []

    # Initial readings
    vs = sim.get_voltage(sweep_slot)
    voltage = abs(agilent.volt())
    current = voltage * agilent_sensitivity
    vs_history.append(vs)
    current_history.append(current)

    # 1) Sweep down until current ≤ min_current (or vs reaches 0)
    while current > min_current and vs > 0:
        vs = max(vs - step, 0.0)
        sim.set_voltage(sweep_slot, vs)
        time.sleep(delay)

        vs = sim.get_voltage(sweep_slot)
        voltage = abs(agilent.volt())
        current = voltage * agilent_sensitivity

        vs_history.append(vs)
        current_history.append(current)

    # 2) Continue sweeping for num_points more steps, record voltages in extra_vs
    for _ in range(num_points):
        if vs > 0:
            vs = max(vs - step, 0.0)
            sim.set_voltage(sweep_slot, vs)

        time.sleep(delay)
        vs = sim.get_voltage(sweep_slot)
        voltage = abs(agilent.volt())
        current = voltage * agilent_sensitivity

        vs_history.append(vs)
        current_history.append(current)
        extra_vs.append(vs)

    # Compute average of the extra sweep voltages
    avg_extra_vs = sum(extra_vs) / len(extra_vs) if extra_vs else 0.0

    return vs_history, current_history, avg_extra_vs

def ramp_two_channels(sim: Any,
                        agilent: Any,
                        slot_exit: str,
                        slot_ent: str,
                        exit_target: float,
                        ent_target: float,
                        #min_current: float,
                        step_exit: float,
                        step_ent: float,
                        delay: float = 0.5,
                        ) -> Tuple[List[float], List[float], List[float]]:
    """
    Simultaneously ramp two gates from their current voltages up to the
    specified targets, recording voltages and current.

    Args:
        sim:         Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
        agilent:     DMM instance where agilent.volt() returns the measured voltage (V).
        slot_exit:   Channel index for the “exit” gate.
        slot_ent:    Channel index for the “ent” gate.
        exit_target: Final voltage (V) for slot_exit.
        ent_target:  Final voltage (V) for slot_ent.
        step_exit:   Voltage increment per iteration for slot_exit.
        step_ent:    Voltage increment per iteration for slot_ent.
        delay:       Seconds to wait after setting voltages.

    Returns:
        (exit_vals, ent_vals, current_vals):
        - exit_vals:  list of voltages read from slot_exit at each step
        - ent_vals:   list of voltages read from slot_ent at each step
        - current_vals: list of currents (in A) measured at each step
    """

    Vent_voltage = sim.get_voltage(slot_exit)
    Vexit_voltage = sim.get_voltage(slot_ent)
    qpc_voltage = sim.get_voltage(3)

    exit_vals: List[float] = []
    ent_vals: List[float] = []
    current_vals: List[float] = []

    # read starting voltages
    v_exit = sim.get_voltage(slot_exit)
    v_ent  = sim.get_voltage(slot_ent)

    # record initial point
    exit_vals.append(v_exit)
    ent_vals.append(v_ent)
    voltage = abs(agilent.volt())
    current = voltage * agilent_sensitivity 
    current_vals.append(current)

    max_iters = 10000
    iteration = 0
    # use 'and' if you want to stop as soon as either reaches its target:
    while v_exit < exit_target and v_ent < ent_target and iteration < max_iters:
        iteration += 1
        # optionally print progress:
        #print(f"[{iteration}] Vexit={v_exit:.3f}, Vent={v_ent:.3f}")

        # ramp logic, trusting local variables:
        if v_exit < exit_target:
            v_exit = min(v_exit + step_exit, exit_target)
            sim.set_voltage(slot_exit, v_exit)
        if v_ent < ent_target:
            v_ent = min(v_ent + step_ent, ent_target)
            sim.set_voltage(slot_ent, v_ent)

        time.sleep(delay)

        # read back current only (skip re‐reading voltages if unreliable)
        voltage = abs(agilent.volt())
        current = voltage * agilent_sensitivity 
        exit_vals.append(v_exit)
        ent_vals.append(v_ent)
        current_vals.append(current)

    if iteration >= max_iters:
        raise RuntimeError("Ramp did not complete within the maximum iteration count")

    df = pd.DataFrame({'V_exit (V)': exit_vals, 'V_ent (V)': ent_vals, 'Current (A)': current_vals})
    filename_1 = f'{qpc_voltage:.3f}_sweep_Vent_Vexit_{timestamp}.csv'
    directory = os.path.join(base_dir, filename_1)
    df.to_csv(directory, index=False,float_format='%.3e')

    # 3) Scatter plot of the first two columns
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    z = df.iloc[:, 2]
    x_label = df.columns[0]
    y_label = df.columns[1]
    z_label = df.columns[2]

    # Read QPC gate voltage from channel 3
    plt.figure()
    legend_label_Vexit = f"V_qpc = {qpc_voltage:.3f}; V_ent = [{Vent_voltage:.3f} V, {ent_target:.3f} V]"

    plt.plot(x, z, label=legend_label_Vexit)
    plt.xlabel(x_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} vs {x_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_2 = f'{qpc_voltage:.3f}_Vexit_Coulomb_{timestamp}.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()

    plt.figure()
    legend_label_Vent = f"V_qpc = {qpc_voltage:.3f}; V_exit = [{Vexit_voltage:.3f} V, {exit_target:.3f} V]"
    plt.plot(y, z, label=legend_label_Vent)
    plt.xlabel(y_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} vs {y_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_3 = f'{qpc_voltage:.3f}_Vent_Coulomb_{timestamp}.png'
    plot_path_1 = os.path.join(base_dir, filename_3)
    plt.savefig(plot_path_1)
    plt.close()

    return directory

def fft_from_df(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided FFT amplitude spectrum of df[y_col] sampled at df[x_col].

    Args:
        df:      pandas DataFrame containing the data.
        x_col:   Name of the column for the independent variable (must be uniformly spaced).
        y_col:   Name of the column for the dependent variable (signal).

    Returns:
        freqs:   1D array of non-negative frequency bins (cycles per unit of x_col).
        amps:    1D array of corresponding single-sided FFT amplitudes.
    """
    # Extract data as NumPy arrays
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Verify uniform spacing in x
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx, atol=1e-9):
        raise ValueError(f"Values in '{x_col}' must be uniformly spaced")

    N = len(y)

    # Compute the FFT and normalize
    Y = np.fft.fft(y)
    Y_mag = np.abs(Y) / N

    # Generate the frequency axis
    freqs = np.fft.fftfreq(N, d=dx)

    # Keep only non-negative frequencies (single-sided spectrum)
    mask = freqs >= 0
    return freqs[mask], 2 * Y_mag[mask]

def sweep_map(sim: Any,
            agilent: Any,
            slot_exit: int,
            slot_ent: int,
            Vexit_avg: float,
            Vent_avg: float,
            Vexit_span: float,
            Vent_span: float,
            step_exit: float = 0.1,
            step_ent: float = 0.1,
            delay: float = 0.5,
            output_csv: str = 'map_data.csv',
            output_fig: str = 'map_colormap.png') -> None:
    """
    For each Vent in [Vent_avg, Vent_avg + Vent_span] (step_ent):
    1) sweep Vexit from Vexit_avg → Vexit_avg+Vexit_span (step_exit),
    2) record (Vent, Vexit, current),
    3) then ramp Vexit back down from max → Vexit_avg (step_exit),
    save all to CSV and draw a colormap.

    Args:
    sim:       Instrument controller (with .set_voltage / .get_voltage).
    agilent:   DMM instance (agilent.volt() → measured voltage).
    slot_exit: Channel index for Vexit.
    slot_ent:  Channel index for Vent.
    Vexit_avg: Base Vexit.
    Vent_avg:  Base Vent.
    Vexit_span: ΔVexit to sweep.
    Vent_span:  ΔVent to sweep.
    step_exit: Vexit increment/decrement per point.
    step_ent:  Vent increment per line.
    delay:     Settle time after each set_voltage.
    output_csv: Path for CSV output.
    output_fig: Path for colormap PNG.
    """
    # build the sweep axes
    Vexit_vals = np.arange(Vexit_avg,
                        Vexit_avg + Vexit_span + 1e-9,
                        step_exit)
    Vent_vals = np.arange(Vent_avg,
                        Vent_avg + Vent_span + 1e-9,
                        step_ent)

    records = []
    current_map = np.zeros((len(Vent_vals), len(Vexit_vals)))

    for i, Vent in enumerate(Vent_vals):
        # set Vent and wait
        sim.set_voltage(slot_ent, Vent)
        time.sleep(delay)

        # forward sweep of Vexit
        for j, Vexit in enumerate(Vexit_vals):
            sim.set_voltage(slot_exit, Vexit)
            time.sleep(delay)

            voltage = abs(agilent.volt())
            current = voltage * agilent_sensitivity 

            current_map[i, j] = current
            records.append({
                'Vent (V)': Vent,
                'Vexit (V)': Vexit,
                'Current (A)': current
            })

        # now ramp Vexit back down to Vexit_avg
        # skip the first element since it's the starting point
        for Vexit_down in reversed(Vexit_vals[:-1]):
            sim.set_voltage(slot_exit, Vexit_down)
            time.sleep(delay)

    qpc_voltage = sim.get_voltage('ch3')
    # 4) save CSV
    df = pd.DataFrame.from_records(records)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
    filename_1 = f'{qpc_voltage}_{output_csv}_{timestamp}.csv'
    directory = os.path.join(base_dir, filename_1)
    df.to_csv(directory, index=False,float_format='%.3e')

    # 5) plot and save colormap

    qpc_voltage = sim.get_voltage(3)
    legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

    plt.figure()
    X, Y = np.meshgrid(Vexit_vals, Vent_vals)
    plt.pcolormesh(X, Y, current_map, shading='auto')
    plt.xlabel('Vexit (V)')
    plt.ylabel('Vent (V)')
    plt.title('Current Map')
    cbar = plt.colorbar()
    cbar.set_label('Current (A)')

    # 4) Save the plot next to the CSV
    filename_2 = f'{qpc_voltage}_{output_fig}_{timestamp}.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()


######## AUTO-TUNING INITIALIZATION PROTOCOL #########

### Connecting to SIM900 and Agilent ###

# Reinitialize SIM928
sim = reinitialize_instrument(
    name="sim900",
    driver=SIM928,
    address="GPIB0::3::INSTR",
    slot_names={i: f'ch{i}' for i in range(1, 9)}
)

# Reinitialize Agilent 34401A
agilent = reinitialize_instrument(
    name="agilent",
    driver=Agilent34401A,
    address="GPIB0::21::INSTR"
)

### Defining varibales which doesn't need to put by the user ###
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

maximum_qpc_voltage = []

results_avg_voltage_val = []

qpc_targets = [(v, v) for v in qpc_gate_voltages]

for v2_target, v3_target in qpc_targets:

    for slot in reversed(range(2,9)):
        ground_device_sim(sim, f'ch{slot}', step= 0.01, delay = 0.2)

    ground_device_sim(sim, 'ch1', step= 0.1, delay = 0.2)

    print('The voltage of all channels in sim900 is set to 0V.')
    print('Step 1: Grounding the devices: completed.')

# Step 2: Setting up the voltage of the top gate (channel 1 in sim900)

    set_voltage_sim(sim, slot = 'ch1', target = 5.0, step = 0.1, delay = 0.5)

    print('Step 2: Setting up the voltage of the top gate (channel 1 of sim900): completed.')

# Step 3: Setting up the voltage of both of the QPC gates (channel 2 and channel 3 in sim900)

    simultaneous_set_voltage_sim(sim, 'ch2', v2_target, 'ch3', v3_target, step = 0.01, delay = 0.5)

    print('Step 3: Setting up the voltage of the QPC gates (channels 2 and 3 of sim900): completed.')

# Step 4: Setting up the voltage of channel 6 of sim900 (which replaces setting the voltage of SR830)

# We are using a 1 M ohm 100 k ohm resistor that will give us 100 uV input. 
    
    set_voltage_sim(sim, slot = 'ch6', target = 0.001, step = 0.0001, delay = 0.5)

    print('Step 4: Setting up the voltage of channel 6 of sim900: completed.')

# Step 5: Sweeping the entrance and the exit gates to find the region of operation. 

# Step (5i): Bringing the exit and the entrance gates symmetrically until current = 5nA. 

    first_point = set_voltage_until_current(sim,agilent, 'ch4', 'ch5', 0.003, 0.2, current_first_point)

    print(f'first point: {first_point}')

# Step (5ii): Stay at the exit gate value found in Step (5i) - (first_point[0]) and bring the entrance gate down until
# minimum current is reached and then sweep for a given number of points.

    set_voltage_sim(sim, slot = 'ch4', target = first_point[0], step = 0.001, delay = 0.2)

    sweep_1_Vent, sweep_1_Vent_current, Vent_avg = sweep_gate_until_min_current(sim,agilent,'ch4','ch5',0.001,0.2,min_current_DC,40)

    save_arrays_to_csv('sweep_1_Vent',sim,                    
                    **{
            'V_ent (V)': sweep_1_Vent,
            'Current (A)': sweep_1_Vent_current
        })
    print(f'Vent : {Vent_avg}')

# Step (5iii): Go back at the values of exit and entrance gates found in Step (5i) (first_point).

    set_voltage_sim(sim, slot = 'ch4', target = first_point[0], step = 0.001, delay = 0.2)
    set_voltage_sim(sim, slot = 'ch5', target = first_point[1], step = 0.001, delay = 0.2)

# Step (5iv): Stay at the entrance gate value set in Step (5iii) - (first_point[1]) and bring the exit gate down until
# minimum current is reached and then sweep for a given number of points.

    set_voltage_sim(sim, slot = 'ch5', target = first_point[1], step = 0.001, delay = 0.2)

    sweep_2_Vexit, sweep_2_Vexit_current, Vexit_avg = sweep_gate_until_min_current(sim,agilent,'ch5','ch4',0.001,0.2,min_current_DC,40)

    save_arrays_to_csv('sweep_2_Vexit',sim, 
                    **{
            'V_exit (V)': sweep_2_Vexit,
            'Current (A)': sweep_2_Vexit_current
        })
    
    print(f'Vexit : {Vexit_avg}')

    results_avg_voltage_val.append({
        'Vent_avg (V)': Vent_avg,
        'Vexit_avg (V)': Vexit_avg,
        'V2_target (V)': v2_target,
        'V3_target (V)': v3_target
    })

# Step (5v): Set the values of exit and entrance gates found as Vexit_avg and Vent_avg.

    set_voltage_sim(sim, slot = 'ch4', target = Vexit_avg, step = 0.001, delay = 0.5)
    set_voltage_sim(sim, slot = 'ch5', target = Vent_avg, step = 0.001, delay = 0.5)

    diagonal_sweep = ramp_two_channels(
            sim=sim,
            agilent=agilent,
            slot_exit='ch4',
            slot_ent='ch5',
            exit_target=Vexit_avg+0.100,
            ent_target=Vent_avg+0.100,
            step_exit=0.001,
            step_ent=0.001,
            delay=0.2
        )

    # Raw data has three columns 'V_exit (V)', 'V_ent (V)', 'Current (A)'
    raw_data = pd.read_csv(diagonal_sweep)
    raw_data = raw_data.drop(raw_data.index[-1]).reset_index(drop=True)

    ## finding the V_ent_max, this will be the maximum entrance value in our RF part 

    # number of points in a row above threshold
    window_DC = 5  

# pull out the two series as plain lists
    current_vals = raw_data['Current (A)'].tolist()
    ent_vals     = raw_data['V_ent (V)'].tolist()

    V_ent_max = None
    for i in range(len(current_vals) - window_DC + 1):
        block_I = current_vals[i:i + window_DC]
    # check for 5 consecutive readings above your DC threshold
        if all(I > min_current_DC for I in block_I):
        # average the matching V_ent points
            V_ent_max = sum(ent_vals[i:i + window_DC]) / window_DC
            break

    # Adding the moving average column and creating a new dataset.
    window = 7

    # 3) Compute the centered moving average of the current
    raw_data['Current_MA'] = raw_data['Current (A)'] \
        .rolling(window=window, center=True, min_periods=1) \
        .mean()

    # 4) Create the delta column: current minus its moving average
    raw_data['Current_minus_MA'] = raw_data['Current (A)'] - raw_data['Current_MA']

    out_file = f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\output_{timestamp}.csv'

    # 2) write raw_data (with new cols) to CSV
    raw_data.to_csv(out_file, index=False, float_format='%.3e')

    # 3) if you really need to re‑load it, pass the filename
    data_moving_avg = pd.read_csv(out_file)

    print(data_moving_avg)

    fig1, ax1 = plt.subplots()
    data_moving_avg.plot(x='V_ent (V)',
                        y='Current (A)',
                        label='I',
                        ax=ax1)
    data_moving_avg.plot(x='V_ent (V)',
                        y='Current_minus_MA',
                        label='I - I_MA',
                        ax=ax1)
    data_moving_avg.plot(x='V_ent (V)',
                        y='Current_MA',
                        label='I_MA',
                        ax=ax1)
    ax1.set_xlabel('V_ent (V)')
    ax1.set_ylabel('Current (A)')
    fig1.tight_layout()
    fig1.savefig(f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\combined_{timestamp}.png')

    fig2, ax2 = plt.subplots()
    data_moving_avg.plot(x='V_ent (V)',
                        y='Current_minus_MA',
                        label='I - I_MA vs V_ent',
                        ax=ax2,
                        legend=False)   # or legend=True if you like
    ax2.set_xlabel('V_ent (V)')
    ax2.set_ylabel('I - I_MA (A)')
    plt.tight_layout()
    fig2.savefig(f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\plot_I_minus_MA_{timestamp}.png')
    # 1) Extract only the V_ent and delta‑current columns
    new_dataset = data_moving_avg[['V_ent (V)', 'Current_minus_MA']].copy()

    # 2) (Optional) Write it out to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file_subset = f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\subset_{timestamp}.csv'
    new_dataset.to_csv(out_file_subset, index=False, float_format='%.3e')

    # 3) Inspect
    print(new_dataset)

    # 1) Assume `new_dataset` already exists:
    #    new_dataset = data_moving_avg[['V_ent (V)', 'Current_minus_MA']].copy()

    # 2) Determine the uniform Vent step
    vent_step = new_dataset['V_ent (V)'].diff().mode()[0]

    # 3) Define how far to pad at top and bottom (±0.08 V)
    pad_range = 0.08

    # 4) Compute original Vent min/max
    vmin = new_dataset['V_ent (V)'].min()
    vmax = new_dataset['V_ent (V)'].max()

    # 5) Generate padded Vent values below and above the existing range
    pad_low  = np.arange(vmin - pad_range, vmin, vent_step)
    pad_high = np.arange(vmax + vent_step, vmax + pad_range + vent_step/2, vent_step)

    # 6) Build DataFrames of zeros for the padded points
    df_low  = pd.DataFrame({
        'V_ent (V)': pad_low,
        'Current_minus_MA': 0.0
    })
    df_high = pd.DataFrame({
        'V_ent (V)': pad_high,
        'Current_minus_MA': 0.0
    })

    # 7) Concatenate and sort into the final padded dataset
    padded_data = pd.concat([df_low, new_dataset, df_high], ignore_index=True)
    padded_data = padded_data.sort_values('V_ent (V)').reset_index(drop=True)

    # 8) Inspect or save
    print(padded_data)
    # Optional: padded_data.to_csv('padded_data.csv', index=False, float_format='%.3e')

    fig3, ax3 = plt.subplots()
    padded_data.plot(x='V_ent (V)',
                        y='Current_minus_MA',
                        label='I - I_MA vs V_ent',
                        ax=ax3,
                        legend=False)   # or legend=True if you like
    ax3.set_xlabel('V_ent (V)')
    ax3.set_ylabel('I - I_MA (A)')
    fig3.tight_layout()
    fig3.savefig(f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\padded_data_{timestamp}.png')

    freqs, amps = fft_from_df(new_dataset, 'V_ent (V)', 'Current_minus_MA')

    df_fft = pd.DataFrame({
        'Frequency (Hz)': freqs,
        'Amplitude': amps
    })

    df_fft = df_fft[df_fft['Frequency (Hz)'] >= 5]

    print(df_fft)
    print(f'freq: {freqs}, amp: {amps}')

    fig4, ax4 = plt.subplots()
    df_fft.plot(x='Frequency (Hz)',
                        y='Amplitude',
                        label='FFT',
                        ax=ax4,
                        legend=False)   # or legend=True if you like
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Amplitude')
    fig4.tight_layout()
    fig4.savefig(f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\FFT_{timestamp}.png')

    # 1) Define a small tolerance for “zero” frequency
    tol = 1e-12

    # 2) Build a mask that excludes the DC bin (and any tiny residual around 0)
    mask = np.abs(freqs) > tol

    # 3) Within the masked amplitudes, find the index of the absolute maximum
    #    Note: amps[mask] is a smaller array, so we need to map back to the full index.
    relative_max_idx = np.argmax(amps[mask])
    absolute_max_idx = np.nonzero(mask)[0][relative_max_idx]

    # 4) Extract the corresponding frequency and amplitude
    peak_freq = freqs[absolute_max_idx]
    peak_amp  = amps[absolute_max_idx]

    print(f"Absolute maximum peak at {peak_freq:.3f} Hz with amplitude {peak_amp:.3e} A")

    qpc_voltage = v3_target   # or sim.get_voltage(3) if you want the actual readback
    maximum_qpc_voltage.append((qpc_voltage, peak_freq))

qpc_vs, peak_fs = zip(*maximum_qpc_voltage)

# Compute the periods (1 / frequency)
peak_periods = [1 / f if f != 0 else np.nan for f in peak_fs]  # Avoid division by zero

V_qpc_vs_time_period = f'{qpc_voltage:.3f}_V_qpc_vs_time_period_{timestamp}.png'
directory_V_qpc_vs_time_period = os.path.join(base_dir, V_qpc_vs_time_period)
# Plot the scatter plot
fig5, ax5 = plt.subplots()
ax5.scatter(qpc_vs, peak_periods)
ax5.set_xlabel('V_qpc (V)')
ax5.set_ylabel('ΔV (V)')
fig5.tight_layout()
fig5.savefig(directory_V_qpc_vs_time_period)

results_avg_voltage_val.append({
         'Vent_avg (V)': Vent_avg,
         'Vexit_avg (V)': Vexit_avg,
         'V_ent_max (V)':   V_ent_max,
         'V2_target (V)': v2_target,
         'V3_target (V)': v3_target
})

df_results = pd.DataFrame(results_avg_voltage_val)
avg_voltage_dataname = f'results_avg_voltage_val_{timestamp}.csv'
directory_avg_voltage = os.path.join(base_dir, avg_voltage_dataname)
df_results.to_csv(directory_avg_voltage, index=False)

######## RF Part of the auto-tuning #########

######## Driver for the RF source ################
class AgilentE4432B(VisaInstrument):
    def __init__(self, name: str, address: str, **kwargs):
        super().__init__(name, address, **kwargs)
        self.connect_message()

    def get_idn(self):
        idn_str = self.ask('*IDN?')
        parts = idn_str.strip().split(',')
        return {
            'vendor': parts[0] if len(parts) > 0 else '',
            'model': parts[1] if len(parts) > 1 else '',
            'serial': parts[2] if len(parts) > 2 else '',
            'firmware': parts[3] if len(parts) > 3 else ''
        }

    def set_frequency(self, freq_hz: float):
        self.write(f'FREQ {freq_hz}HZ')

    def get_frequency(self) -> float:
        return float(self.ask('FREQ?'))

    def set_power(self, power_dbm: float):
        self.write(f'POW {power_dbm}DBM')

    def get_power(self) -> float:
        return float(self.ask('POW?'))

    def rf_on(self):
        self.write('OUTP ON')

    def rf_off(self):
        self.write('OUTP OFF')

P_dBm = [-30, -29, -28, -27, -26, -25, -24, -23, -22, -21,
         -20, -19, -18, -17, -16, -15, -14, -13, -12, -11,
         -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30]

Vp = [
   0.010, 0.011, 0.013, 0.014, 0.016, 0.018, 0.020, 0.022, 0.025, 0.028,
   0.032, 0.035, 0.040, 0.045, 0.050, 0.056, 0.063, 0.071, 0.079, 0.089,  
   0.100, 0.112, 0.126, 0.141, 0.158, 0.178, 0.200, 0.224, 0.251, 0.282,  
   0.316, 0.355, 0.398, 0.447, 0.501, 0.562, 0.631, 0.708, 0.794, 0.891,  
   1.000, 1.122, 1.259, 1.413, 1.585, 1.778, 1.995, 2.239, 2.512, 2.818,  
   3.162, 3.548, 3.981, 4.467, 5.012, 5.623, 6.310, 7.079, 7.943, 8.913, 10.000 ]
 
power_to_voltage = pd.DataFrame({'Power (dBm)': P_dBm,'Voltage (Vp)': Vp})

for slot in reversed(range(2,9)):
    ground_device_sim(sim, f'ch{slot}', step= 0.01, delay = 0.5)

print('The voltage of channels 2 - 8 in sim900 is set to 0V.')

# Ask the user to enter the Agilent RF sensitivity instead of hard-coding it
while True:
    try:
        agilent_sensitivity_RF = float(
            input('Enter the Agilent RF sensitivity (A/V), e.g. 1e-9: ').strip()
        )
        break
    except ValueError:
        print('Invalid number—please enter a floating-point value (e.g. 1e-9).')

response = input('Did you change the sensitivity? (y/n): ').strip().lower()
if response != 'y':
    print('Please change the sensitivity before proceeding. Exiting script.')
    exit()

# Connect and read values
rf = AgilentE4432B('rf_gen', 'GPIB0::19::INSTR')
print('Connected to:', rf.get_idn())

freq = rf.get_frequency()
power = rf.get_power()

print(f'Frequency: {freq} Hz')
print(f'Power: {power} dBm')

# Target values
target_freq = 100_000_000.0  # 100 MHz
target_power = -30.0         # -30 dBm

# --- Parameters (customize if you like) ---
power_step = 0.1    # dBm increment per step
delay      = 0.2    # seconds to wait after each set

# --- Read current RF power ---
current_power = rf.get_power()
print(f'Current RF power: {current_power:.1f} dBm')

# --- Prompt user for target power ---
target_power = power

# --- Determine step direction ---
if target_power > current_power:
    step = power_step
elif target_power < current_power:
    step = -power_step
else:
    print('Current power already at target; no ramping needed.')
    step = 0

# --- Ramp in steps ---
if step != 0:
    for p in np.arange(current_power, target_power, step):
        rf.set_power(p)
        print(f'  → RF power set to {p:.1f} dBm')
        time.sleep(delay)
    
    # Ensure final exact target
    rf.set_power(target_power)
    print(f'  → RF power set to target {target_power:.1f} dBm')

# Check if settings are correct
if freq == target_freq and power == target_power:
    print('Frequency and power are correctly set.')
    print('Turning RF output ON.')
    rf.rf_on()
else:
    print('ERROR: RF source not set to required frequency or power.')
    print(f'Expected Frequency: {target_freq} Hz, Measured: {freq} Hz')
    print(f'Expected Power: {target_power} dBm, Measured: {power} dBm')

    # Prompt user
    while True:
        response = input('Did you fix the error? (y/n): ').strip().lower()
        if response == 'y':
            print('Proceeding with the script.')
            break
        elif response == 'n':
            print('Exiting. Please set the correct frequency and power before rerunning.')
            exit()
        else:
            print('Invalid input. Please enter "y" or "n".')

# sensitivity_RF = 1e-9

# qpc_gate_voltages_RF = [0.00,0.100]

# user_power_dBm = [1, 2, 3]

qpc_targets_RF = [(v, v) for v in qpc_gate_voltages_RF]

for v2_target, v3_target in qpc_targets_RF:

# Step 3: Setting up the voltage of both of the QPC gates (channel 2 and channel 3 in sim900)
    simultaneous_set_voltage_sim(sim, 'ch2', v2_target, 'ch3', v3_target, step = 0.01, delay = 0.5)

    print('Step 3: Setting up the voltage of the QPC gates (channels 2 and 3 of sim900): completed.')
            
##Next steps

#
# 1. In the diagonal sweep, for the coulomb oscillations, find the V_ent value when the current> I_min. I would suggest taking 5 consecutive points where 
 #  this condition is satisfied and save it in the same csv folder where we save the V_avg values. You can call it V_ent_max.
#something like the following:
'''
# number of points in a row above threshold
window_DC = 5  

# pull out the two series as plain lists
current_vals = raw_data['Current (A)'].tolist()
ent_vals     = raw_data['V_ent (V)'].tolist()

V_ent_max = None
for i in range(len(current_vals) - window_DC + 1):
    block_I = current_vals[i:i + window_DC]
    # check for 5 consecutive readings above your DC threshold
    if all(I > min_current_DC for I in block_I):
        # average the matching V_ent points
        V_ent_max = sum(ent_vals[i:i + window_DC]) / window_DC
        break
#'''

 # 2. Everything else is smae as what we did before, just replace the V_avg ent with new V_max
 # 3. Moving forward to plateau detection, we will sweep the exit gate untill it reaches the I_max=150 pA, or V_max=V_exit_Avg
 #5. look for quantization that means the derivative = 0 for a range of voltages (atleast for 10 consecutive voltages values), also we know the 
 #quantization occurs at I=nef
 #def find_flat_plateau(
   # Vs: np.ndarray,
   # Is: np.ndarray,
   # tol: float=1e-12,
   # window: int=10
#) -> Optional[Tuple[float,float]]:
   # dI = np.abs(np.diff(Is)/np.diff(Vs))
    #for i in range(len(dI) - window + 1):
    #    if np.all(dI[i:i+window] < tol):
    #        return Vs[i], Vs[i+window]
    #return None

 #6. If there is quatization do the decay cascade model (you need atleast 2 plateaus for that), obtain the delta value, record the delta and corresponding v_entrance (rf AND DC), v_qpc and range of exit value
 #7. once we find quantization move to the next power (make sure we ground everything before that). In the third stage if we dont find quantization move to next poer and try.
 #8. Once we finish the sweep , we need to scan through all the delta value and take a pump map for the one with high delta value. 
 # How to find the exit and entrance scan range: go to the maximum point in the pump 1D trace, (i.e. when it was  I_max=150 pA),
 #bring the entrance down untill I<= I_min_RF (same procedure as we did in the dc), find the point by averaging. Do the same for exit. That will be our range for pump map, just what we did for the dc map


# --- model definition ---
# def cascade_model(x, a, v, d):
#     e = 1.602176634e-19
#     f = rf.get_frequency()
#     t1 = np.exp(-np.exp(-a * (x - v)))
#     t2 = np.exp(-np.exp(-a * (x - v) + d))
#     return e * f * (t1 + t2) * 1e9  # in nA·Hz·C units

def cascade_model(x, a, v, d):
    e = 1.602176634e-19
    f = rf.get_frequency()
    z = -a * (x - v)
    # clip to avoid overflow:
    z1 = np.clip(z, -50, +50)
    z2 = np.clip(z + d, -50, +50)

    t1 = np.exp(-np.exp(z1))
    t2 = np.exp(-np.exp(z2))
    return e * f * (t1 + t2) * 1e9

# thresholds and steps
threshold_current              = 150e-12    # stops individual sweep
minimum_threshold_current_RF   = 0.5e-12    # decides whether to plot

summary_results = []

for v2_target, v3_target in qpc_targets_RF:

    simultaneous_set_voltage_sim(sim,
                                'ch2', v2_target,
                                'ch3', v3_target,
                                step=0.01, delay=0.5)
    row = df_results[np.isclose(df_results['V2_target (V)'],
                                v2_target, atol=1e-6)]
    if row.empty:
        continue

    vent_avg_RF  = row.iloc[0]['Vent_avg (V)']
    vexit_avg_RF = row.iloc[0]['Vexit_avg (V)']

    for power in user_power_dBm:

        # --- Parameters (customize if you like) ---
        power_step = 0.1    # dBm increment per step
        delay      = 0.2    # seconds to wait after each set

        # --- Read current RF power ---
        current_power = rf.get_power()
        print(f'Current RF power: {current_power:.1f} dBm')

        # --- Prompt user for target power ---
        target_power = power

        # --- Determine step direction ---
        if target_power > current_power:
            step = power_step
        elif target_power < current_power:
            step = -power_step
        else:
            print('Current power already at target; no ramping needed.')
            step = 0

        # --- Ramp in steps ---
        if step != 0:
            for p in np.arange(current_power, target_power, step):
                rf.set_power(p)
                print(f'  → RF power set to {p:.1f} dBm')
                time.sleep(delay)
            
            # Ensure final exact target
            rf.set_power(target_power)
            print(f'  → RF power set to target {target_power:.1f} dBm')

        vp_row = power_to_voltage[
            power_to_voltage['Power (dBm)'] == power
        ]
        if vp_row.empty:
            continue
        vp = vp_row['Voltage (Vp)'].values[0]

        vent_min = 0
        vent_max = V_ent_max - vp

        cascade_found_for_power = False


        vent_step           = 0.003   # 1 mV Vent increments
        vent_sweep_interval = 0.05    # trigger Vexit sweep every 50 mV
        vexit_step          = 0.003   # 1 mV exit increments
        next_sweep_threshold = vent_min + vent_sweep_interval

        # How many Vent steps are in one sweep‐interval?
        steps_per_sweep = int(round(vent_sweep_interval / vent_step))

        # Generate your Vent values:
        vent_values = np.arange(vent_min,
                                vent_max + vent_step/2,
                                vent_step)

        for idx, vent in enumerate(vent_values):
            # 1) ramp Vent
            set_voltage_sim(sim, slot='ch5', target=vent,
                            step=vent_step, delay=0.1)

            # 2) every Nth step (i.e. every 50 mV), do a Vexit sweep:
            if idx % steps_per_sweep == 0:
                vexit_vals = []
                currents   = []
                for vexit in np.arange(0.0,
                                    vexit_avg_RF + vexit_step/2,
                                    vexit_step):
                    set_voltage_sim(sim, slot='ch4', target=vexit,
                                    step=vexit_step, delay=0.1)
                    I = abs(agilent.volt()) * agilent_sensitivity_RF
                    vexit_vals.append(vexit)
                    currents.append(I)
                    if I >= threshold_current:
                        break

                # save / analyze here…
                # e.g. df = pd.DataFrame({ … })
                #      df.to_csv(…)
                print(f'→ Vexit sweep at Vent={vent:.3f} V')

                # save the mini-sweep
                df = pd.DataFrame({
                    'Vent (V)':    [vent] * len(vexit_vals),
                    'Vexit (V)':   vexit_vals,
                    'Current (A)': currents
                })
                fname = f'vent_{vent:.3f}_exit_sweep_{timestamp}.csv'
                df.to_csv(os.path.join(base_dir, fname), index=False)
                print(f'→ Performed exit sweep at Vent={vent:.3f} V, saved to {fname}')

                # schedule next trigger
                next_sweep_threshold += vent_sweep_interval

                # now that currents is defined, apply your skip and peak logic
                if not any(I > minimum_threshold_current_RF for I in currents):
                    # nothing above 0.3 pA, move on
                    continue

                derivative = np.gradient(currents, vexit_vals)
                peaks, _   = find_peaks(derivative)
                num_peaks  = len(peaks)
                print(f'Found {num_peaks} peaks in dI/dVexit at Vent={vent:.3f} V, P={power} dBm')

                # else: no sweep, so nothing to skip or analyze at this vent

                if num_peaks > 2:
                    # fit cascade_model to the full derivative curve
                    xdata = np.array(vexit_vals)
                    ydata = np.array(currents)

                    # sensible parameter bounds and initial guess
                    bounds = ([0.0,           0.0,          -10.0],
                            [10.0,          vexit_max+0.1, 10.0])
                    p0     = [1.0, xdata[np.argmax(ydata)], 1.0]

                    popt, pcov = curve_fit(
                        cascade_model,
                        xdata,
                        ydata,
                        p0=p0,
                        bounds=bounds,
                        maxfev=10_000
                    )

                    a_fit, v_fit, d_fit = popt
                    print(f"cascade_model fit parameters: a={a_fit:.3g}, v0={v_fit:.3f} V, d={d_fit:.3g}")

                    # plot derivative + fit overlay
                    fig, ax = plt.subplots()
                    ax.plot(xdata, ydata,     'o',  label='Measured I')
                    ax.plot(xdata,
                            cascade_model(xdata, *popt),
                            '--', label='Cascade fit')
                    ax.set_xlabel('Vexit (V)')
                    ax.set_ylabel('Current (A)')
                    ax.set_title(f'Pump Current vs Vexit, Vent={vent:.3f} V, P={power} dBm')
                    ax.legend()
                    plt.show()

                    Vent_max = V_ent_max
                    Vexit_const = vexit_avg_RF
                    n_extra = 5

                    # Horizontal sweep: Vexit fixed, sweep Vent until I < threshold, +5 extra
                    horizontal_data = []
                    extra_count = 0
                    vent = Vent_max

                    while True:
                        # set Vent & constant Vexit
                        set_voltage_sim(sim, 'ch4', vent, step=vent_step, delay=0.1)
                        set_voltage_sim(sim, 'ch5', Vexit_const, step=vexit_step, delay=0.1)

                        voltage = abs(agilent.volt())
                        I = voltage * agilent_sensitivity_RF
                        horizontal_data.append({'Vent (V)': vent, 'I (A)': I})

                        if I < threshold_current:
                            extra_count += 1
                            if extra_count > n_extra:
                                break

                        vent += vent_step

                    # Vertical sweep: Vent fixed, sweep Vexit until I < threshold, +5 extra
                    vertical_data = []
                    extra_count = 0
                    vexit = Vexit_const

                    while True:
                        # set constant Vent & Vexit
                        set_voltage_sim(sim, 'ch4', Vent_max, step=vent_step, delay=0.1)
                        set_voltage_sim(sim, 'ch5', vexit,   step=vexit_step, delay=0.1)

                        voltage = abs(agilent.volt())
                        I = voltage * agilent_sensitivity_RF
                        vertical_data.append({'Vexit (V)': vexit, 'I (A)': I})

                        if I < threshold_current:
                            extra_count += 1
                            if extra_count > n_extra:
                                break

                        vexit += vexit_step

                    # Save horizontal sweep
                    df_h = pd.DataFrame(horizontal_data)
                    hfile = (f'horz_sweep_V2_{v2_target:.3f}_'
                            f'P_{power}dBm_'
                            f'Ventmax_{Vent_max:.3f}V_'
                            f'{timestamp}.csv')
                    df_h.to_csv(os.path.join(base_dir, hfile), index=False)

                    # Save vertical sweep
                    df_v = pd.DataFrame(vertical_data)
                    vfile = (f'vert_sweep_V2_{v2_target:.3f}_'
                            f'P_{power}dBm_'
                            f'Ventmax_{Vent_max:.3f}V_'
                            f'{timestamp}.csv')
                    df_v.to_csv(os.path.join(base_dir, vfile), index=False)

                    print(f'→ Horizontal sweep saved to {hfile}')
                    print(f'→ Vertical   sweep saved to {vfile}')

                    # Take the last five entries from each mini‐sweep
                    last5_h = horizontal_data[-5:]
                    last5_v = vertical_data[-5:]

                    # Compute their averages
                    Vent_min_RF  = np.mean([pt['Vent (V)']    for pt in last5_h])
                    vexit_min_RF = np.mean([pt['Vexit (V)']   for pt in last5_v])

                    print(f'Vent_min_RF  (avg of last 5 points) = {Vent_min_RF:.4f} V')
                    print(f'vexit_min_RF (avg of last 5 points) = {vexit_min_RF:.4f} V')

                        # record this combination
                    summary_results.append({
                        'V2_target (V)':    v2_target,
                        'Power (dBm)':      power,
                        'Vent_max (V)':       vent_max,
                        'Vexit_avg_RF (V)':   vexit_avg_RF,
                        'Vent_min_RF (V)':  Vent_min_RF,
                        'Vexit_min_RF (V)': vexit_min_RF,
                        'd (fit)':          d_fit
                    })
                    
                    # signal to skip remaining Vent for this power
                    cascade_found_for_power = True
                    break

            if cascade_found_for_power:
                # immediately continue with next power
                break

# at this point, each power is processed until its first cascade fit, 
# then we move on to the next power/QPC setting

df_summary = pd.DataFrame(summary_results)
df_summary.to_csv(os.path.join(base_dir,
                 f'cascade_summary_{timestamp}.csv'),
                 index=False)

# 2. Select the row with the highest d-value
best_idx  = df_summary['d (fit)'].idxmax()
best_row  = df_summary.loc[best_idx]

# 3. Extract the needed parameters
v2_best      = best_row['V2_target (V)']
power_best   = best_row['Power (dBm)']
Vent_min     = best_row['Vent_min_RF (V)']
Vexit_min    = best_row['Vexit_min_RF (V)']
Vent_max     = best_row['Vent_max (V)']
Vexit_max    = best_row['Vexit_avg_RF (V)']

# 4. Build sensible file names
csv_name  = f'map_QPC_{v2_best:.3f}V_P_{power_best}dBm.csv'
fig_name  = f'map_QPC_{v2_best:.3f}V_P_{power_best}dBm.png'

sweep_map(
    sim=sim,
    agilent=agilent,
    slot_exit=5,            # channel index for Vexit (e.g. ch5)
    slot_ent=4,             # channel index for Vent  (e.g. ch4)
    Vexit_avg=Vexit_min,
    Vent_avg=Vent_min,
    Vexit_span=(Vexit_max - Vexit_min),
    Vent_span=(Vent_max  - Vent_min),
    step_exit=0.001,
    step_ent=0.001,
    delay=0.5,
    output_csv=csv_name,
    output_fig=fig_name
)

print(f'Colormap generated for QPC={v2_best:.3f} V, P={power_best} dBm:')
print(f'  • Vent ∈ [{Vent_min:.4f}, {Vent_max:.4f}] V')
print(f'  • Vexit∈ [{Vexit_min:.4f}, {Vexit_max:.4f}] V')
print(f'  • CSV → {csv_name}')
print(f'  • Figure → {fig_name}')













































































































































































































































































































































































































































































































































































































































































































# # Threshold to stop the Vexit sweep early
# threshold_current = 150e-12           # 150 pA

# # Minimum current threshold to decide whether to plot/save
# minimum_threshold_current_RF = 0.3e-12  # 0.3 pA

# vexit_step = 0.001   # 1 mV steps for Vexit
# vent_step  = 0.01    # 10 mV steps for Vent

# for v2_target, v3_target in qpc_targets_RF:

#     # 1) set QPC gates
#     simultaneous_set_voltage_sim(sim,
#                                 'ch2', v2_target,
#                                 'ch3', v3_target,
#                                 step=0.01, delay=0.5)

#     row = df_results[np.isclose(df_results['V2_target (V)'],
#                                 v2_target, atol=1e-6)]
#     if row.empty:
#         continue

#     vent_avg_RF  = row.iloc[0]['Vent_avg (V)']
#     vexit_avg_RF = row.iloc[0]['Vexit_avg (V)']

#     for power in user_power_dBm:
#         # set RF power
#         rf.set_power(power)

#         vp_row = power_to_voltage[
#             power_to_voltage['Power (dBm)'] == power
#         ]
#         if vp_row.empty:
#             continue
#         vp = vp_row['Voltage (Vp)'].values[0]

#         # Vent sweep range
#         vent_min = V_ent_max - vp
#         vent_max = V_ent_max

#         for vent in np.arange(vent_min, vent_max + 1e-12, vent_step):
#             # set Vent
#             set_voltage_sim(sim,
#                             slot='ch4',
#                             target=vent,
#                             step=0.001,
#                             delay=0.5)

#             # sweep Vexit
#             vexit_vals = []
#             currents   = []
#             for vexit in np.arange(0.0,
#                                    vexit_avg_RF + vexit_step/2,
#                                    vexit_step):

#                 set_voltage_sim(sim,
#                                 slot='ch5',
#                                 target=vexit,
#                                 step=vexit_step,
#                                 delay=0.1)

#                 I = agilent.read_current()
#                 vexit_vals.append(vexit)
#                 currents.append(I)

#                 # stop this sweep if current ≥ 150 pA
#                 if I >= threshold_current:
#                     break

#             # if no point even exceeds 0.3 pA, skip
#             if not any(I > minimum_threshold_current_RF for I in currents):
#                 continue

#             # save CSV
#             df_sweep = pd.DataFrame({
#                 'Vent (V)':    [vent] * len(vexit_vals),
#                 'Vexit (V)':   vexit_vals,
#                 'Current (A)': currents
#             })
#             fname = (f'QPC_V2_{v2_target:.3f}_'
#                      f'P_{power}dBm_'
#                      f'Vent_{vent:.3f}V_'
#                      f'{timestamp}.csv')
#             df_sweep.to_csv(os.path.join(base_dir, fname),
#                             index=False)

#             # plot Ipump vs Vexit
#             fig, ax = plt.subplots()
#             ax.plot(vexit_vals, currents, marker='o', linestyle='-')
#             ax.set_xlabel('Vexit (V)')
#             ax.set_ylabel('Ipump (A)')
#             ax.set_title(f'QPC V2={v2_target:.3f} V, P={power} dBm, Vent={vent:.3f} V')
#             plt.show()

#             print(f'Plotted & saved sweep for Vent={vent:.3f} V, power={power} dBm')

#             # Compute numerical derivative dI/dVexit
#             derivative = np.gradient(currents, vexit_vals)

#             # Find peaks in the derivative
#             peaks, _ = find_peaks(derivative)
#             num_peaks = len(peaks)
#             print(f'Number of peaks in dI/dVexit: {num_peaks}')

#             # Plot the derivative and mark peaks
#             fig2, ax2 = plt.subplots()
#             ax2.plot(vexit_vals, derivative, label='dI/dVexit')
#             ax2.plot(np.array(vexit_vals)[peaks],
#                     derivative[peaks],
#                     'rx',
#                     label='peaks')
#             ax2.set_xlabel('Vexit (V)')
#             ax2.set_ylabel('dI/dVexit (A/V)')
#             ax2.set_title(f'dI/dVexit for Vent={vent:.3f} V, P={power} dBm')
#             ax2.legend()
#             plt.show()