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
qpc_gate_voltages = [0.000, 0.200]

# Enter the sensitivity of agilent.
agilent_sensitivity = 1e-7

# Enter the current value until which the V_ent and V_exit gates are symmetrically increased.
current_first_point = 0.9e-9

# Enter the current by manually checking the offset of the pre-amplifier. 
min_current_DC = 30.0e-13

# Enter the maximum voltage that cen be set up for Vent and Vexit. 
limiting_voltage = 1.1 

######## I need to work starting from here.``
# Enter the step_size
step_size = 0.01

# Enter the step size for top gate.
step_top_gate = 0.1

# top_gate = 

# source_drain_bias = 

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
                        step_exit: float,
                        step_ent: float,
                        delay: float = 0.5
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

qpc_targets = [(v, v) for v in qpc_gate_voltages]

for v2_target, v3_target in qpc_targets:

    for slot in reversed(range(2,9)):
        ground_device_sim(sim, f'ch{slot}', step= 0.01, delay = 0.5)

    ground_device_sim(sim, 'ch1', step= 0.1, delay = 0.5)

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

# Step (5v): Set the values of exit and entrance gates found as Vexit_avg and Vent_avg.

    set_voltage_sim(sim, slot = 'ch4', target = Vexit_avg, step = 0.001, delay = 0.5)
    set_voltage_sim(sim, slot = 'ch5', target = Vent_avg, step = 0.001, delay = 0.5)


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

    sweep_map(sim,agilent,4,5,Vexit_avg,Vent_avg,0.100,0.100,0.001,0.001,0.1,output_csv=f'qpc_map_{timestamp}',output_fig=f'qpc_map__{timestamp}')

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


