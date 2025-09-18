from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
import time
import os
from datetime import datetime
from typing import List, Tuple
from typing import Sequence
from typing import Any
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # # The following gates are assumed to be connected to the following channels in sim900:
# # # Channel 1: Top gate
# # # Channel 2: QPC bottom
# # # Channel 3: QPC top
# # # Channel 4: Exit gate
# # # Channel 5: Entrance gate
# # # Channel 6: Source-drain bias (currently set to zero)

# # Connecting to SIM900 and Agilent

sim = SIM928("sim900", "GPIB0::3::INSTR", slot_names={i: f'ch{i}' for i in range (1,9)})
agilent = Agilent34401A("agilent", "GPIB0::21::INSTR")

# # # Step 1: Grounding the device 

def ground_device_sim(sim, slot: int, step: float = 0.1, delay: float = 0.5):
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

for slot in reversed(range(2,9)):
    ground_device_sim(sim, slot, step= 0.01, delay = 0.5)

ground_device_sim(sim, 1, step= 0.1, delay = 0.5)

print('The voltage of all channels in sim900 is set to 0V.')
print('Step 1: Grounding the devices: completed.')

# # Step 2: Setting up the voltage of the top gate (channel 1 in sim900)

def set_voltage_sim(sim, slot: int, target: float, step: float = 0.1, delay: float = 0.5):
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

set_voltage_sim(sim, slot = 1, target = 5.0, step = 0.1, delay = 0.5)

print('Step 2: Setting up the voltage of the top gate (channel 1 of sim900): completed.')

# Step 3: Setting up the voltage of both of the QPC gates (channel 2 and channel 3 in sim900)

def simultaneous_set_voltage_sim(sim, slot1: int, target1: float, slot2: int, target2: float, step: float = 0.01, delay: float = 0.5):

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

simultaneous_set_voltage_sim(sim, 2, 0.2, 3, 0.2, step = 0.01, delay = 0.5)

print('Step 3: Setting up the voltage of the QPC gates (channels 2 and 3 of sim900): completed.')

# Step 4: Setting up the voltage of channel 6 of sim900 (which replaces setting the voltage of SR830)

# We are using a 1 M ohm 100 k ohm resistor that will give us 100 uV input. 
set_voltage_sim(sim, slot = 6, target = 0.001, step = 0.0001, delay = 0.5)

print('Step 4: Setting up the voltage of channel 6 of sim900: completed.')

# Step 5: Sweeping the entrance and the exit gates to find the region of operation. 

# Step (5i): Bringing the exit and the entrance gates symmetrically until current = 5nA. 

def set_voltage_until_current(sim, agilent,slot1: int, slot2: int, step: float, delay: float, current_target: float):

    V1 = sim.get_voltage(slot1)
    V2 = sim.get_voltage(slot2)
    voltage = agilent.volt()
    current = -1*voltage*1e-7

    v1_history: List[float] = []
    v2_history: List[float] = []
    current_history: List[float] = []

    v1_history.append(V1)
    v2_history.append(V2)
    current_history.append(current)

    while current < current_target:

        if V1 >= 0.95 and V2 >= 0.95:
            raise RuntimeError(
                f"Failed to reach {current_target} A: "
                f"V1 and V2 have both reached 1.1 V (current = {current:.3e} A)."
            )
        
        V1 += step
        V2 += step 
        sim.set_voltage(slot1, V1)
        sim.set_voltage(slot2, V2)
        time.sleep(delay)
        voltage = agilent.volt()
        current = -1*voltage*1e-7

        v1_history.append(V1)
        v2_history.append(V2)
        current_history.append(current)

    data_first_point = pd.DataFrame({'V_slot1': v1_history,'V_slot2': v2_history, 'I (A)': current_history})

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
    filename = f'data_first_point_{timestamp}.csv'
    directory = os.path.join(base_dir, filename)

    data_first_point.to_csv(directory, index=False, float_format='%.3e')
    
    V1_a = sim.get_voltage(slot1)
    V2_b = sim.get_voltage(slot2)

    first_point = [V1_a, V2_b]
    return first_point

first_point = set_voltage_until_current(sim,agilent,4, 5, 0.003, 0.2, 0.9e-9)

print(f'first point: {first_point}')

# Step (5ii): Stay at the exit gate value found in Step (5i) - (first_point[0]) and bring the entrance gate down until
# minimum current is reached and then sweep for a given number of points.

set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.001, delay = 0.2)

def sweep_gate_until_min_current(sim,
                                 agilent,
                                 constant_slot: int,
                                 sweep_slot: int,
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
    voltage = agilent.volt()
    current = -1 * voltage * 1e-7
    vs_history.append(vs)
    current_history.append(current)

    # 1) Sweep down until current ≤ min_current (or vs reaches 0)
    while current > min_current and vs > 0:
        vs = max(vs - step, 0.0)
        sim.set_voltage(sweep_slot, vs)
        time.sleep(delay)

        vs = sim.get_voltage(sweep_slot)
        voltage = agilent.volt()
        current = -1 * voltage * 1e-7

        vs_history.append(vs)
        current_history.append(current)

    # 2) Continue sweeping for num_points more steps, record voltages in extra_vs
    for _ in range(num_points):
        if vs > 0:
            vs = max(vs - step, 0.0)
            sim.set_voltage(sweep_slot, vs)

        time.sleep(delay)
        vs = sim.get_voltage(sweep_slot)
        voltage = agilent.volt()
        current = -1 * voltage * 1e-7

        vs_history.append(vs)
        current_history.append(current)
        extra_vs.append(vs)

    # Compute average of the extra sweep voltages
    avg_extra_vs = sum(extra_vs) / len(extra_vs) if extra_vs else 0.0

    return vs_history, current_history, avg_extra_vs

sweep_1_Vent, sweep_1_Vent_current, Vent_avg = sweep_gate_until_min_current(sim,agilent,4,5,0.001,0.2,9.0e-13,40)

save_arrays_to_csv('sweep_1_Vent',sim,                    
                   **{
        'V_ent (V)': sweep_1_Vent,
        'Current (A)': sweep_1_Vent_current
    })
print(f'Vent : {Vent_avg}')

# Step (5iii): Go back at the values of exit and entrance gates found in Step (5i) (first_point).

set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.001, delay = 0.2)
set_voltage_sim(sim, slot = 5, target = first_point[1], step = 0.001, delay = 0.2)

# Step (5iv): Stay at the entrance gate value set in Step (5iii) - (first_point[1]) and bring the exit gate down until
# minimum current is reached and then sweep for a given number of points.

set_voltage_sim(sim, slot = 5, target = first_point[1], step = 0.001, delay = 0.2)

sweep_2_Vexit, sweep_2_Vexit_current, Vexit_avg = sweep_gate_until_min_current(sim,agilent,5,4,0.001,0.2,9.0e-13,40)

save_arrays_to_csv('sweep_2_Vexit',sim, 
                   **{
        'V_exit (V)': sweep_2_Vexit,
        'Current (A)': sweep_2_Vexit_current
    })
print(f'Vexit : {Vexit_avg}')

# Step (5v): Set the values of exit and entrance gates found as Vexit_avg and Vent_avg.

set_voltage_sim(sim, slot = 4, target = Vexit_avg, step = 0.1, delay = 0.5)
set_voltage_sim(sim, slot = 5, target = Vent_avg, step = 0.1, delay = 0.5)

# Step (5vi): Sweep the values V_exit for each V_ent and get a colormap.

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

            voltage = agilent.volt()
            current = voltage * 1e-6

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

    # 4) save CSV
    df = pd.DataFrame.from_records(records)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
    filename_1 = f'{output_csv}_{timestamp}.csv'
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
    filename_2 = f'{output_fig}_{timestamp}.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()

#sweep_map(sim,agilent,4,5,Vexit_avg,Vent_avg,0.020,0.020,0.001,0.001,0.1,output_csv='qpc_map',output_fig='qpc_map')


set_voltage_sim(sim, slot = 4, target = Vexit_avg, step = 0.1, delay = 0.5)
set_voltage_sim(sim, slot = 5, target = Vent_avg, step = 0.1, delay = 0.5)

def ramp_two_channels(sim: Any,
                      agilent: Any,
                      slot_exit: int,
                      slot_ent: int,
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

    Vent_voltage = sim.get_voltage(5)

    exit_vals: List[float] = []
    ent_vals: List[float] = []
    current_vals: List[float] = []

    # read starting voltages
    v_exit = sim.get_voltage(slot_exit)
    v_ent  = sim.get_voltage(slot_ent)

    # record initial point
    exit_vals.append(v_exit)
    ent_vals.append(v_ent)
    voltage = agilent.volt()
    current = -1 * voltage * 1e-7
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
        voltage = agilent.volt()
        current = -1 * voltage * 1e-7
        exit_vals.append(v_exit)
        ent_vals.append(v_ent)
        current_vals.append(current)

    if iteration >= max_iters:
        raise RuntimeError("Ramp did not complete within the maximum iteration count")

    df = pd.DataFrame({'V_exit (V)': exit_vals, 'V_ent (V)': ent_vals, 'Current (A)': current_vals})
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
    filename_1 = f'sweep_Vent_Vexit_{timestamp}.csv'
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
    qpc_voltage = sim.get_voltage(3)
    legend_label = f"V_qpc = {qpc_voltage:.3f}; V_ent = {Vent_voltage:.3f} V"

    plt.figure()
    plt.plot(x, z, label=legend_label)
    plt.xlabel(x_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} vs {x_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_2 = f'{Vent_voltage:.3f}_Vexit_Coulomb_{timestamp}.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()

    plt.figure()
    plt.plot(y, z, label=legend_label)
    plt.xlabel(y_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} vs {y_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_3 = f'{Vent_voltage:.3f}_Vent_Coulomb_{timestamp}.png'
    plot_path_1 = os.path.join(base_dir, filename_3)
    plt.savefig(plot_path_1)
    plt.close()

    return filename_1

diagonal_sweep = ramp_two_channels(
        sim=sim,
        agilent=agilent,
        slot_exit=4,
        slot_ent=5,
        exit_target=Vexit_avg,
        ent_target=Vent_avg,
        step_exit=0.001,
        step_ent=0.001,
        delay=0.2
    )

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

# Raw data has three columns 'V_exit (V)', 'V_ent (V)', 'Current (A)'.

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
raw_data = pd.read_csv(diagonal_sweep)

print(raw_data.to_string())

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
                     label='I vs V_ent',
                     ax=ax1)
data_moving_avg.plot(x='V_ent (V)',
                     y='Current_minus_MA',
                     label='I - I_MA vs V_ent',
                     ax=ax1)
data_moving_avg.plot(x='V_ent (V)',
                     y='Current_MA',
                     label='I_MA vs V_ent',
                     ax=ax1)
ax1.set_xlabel('V_ent (V)')
ax1.set_ylabel('Current (A)')
plt.tight_layout()
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
plt.tight_layout()
fig3.savefig(f'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\padded_data_{timestamp}.png')

freqs, amps = fft_from_df(new_dataset, 'V_ent (V)', 'Current_minus_MA')

df_fft = pd.DataFrame({
    'Frequency (1/V)': freqs,
    'Amplitude (A)': amps
})

df_fft = df_fft[df_fft['Frequency (1/V)'] >= 5]

print(df_fft)
print(f'freq: {freqs}, amp: {amps}')

fig4, ax4 = plt.subplots()
df_fft.plot(x='Frequency (1/V)',
                     y='Amplitude (A)',
                     label='FFT',
                     ax=ax4,
                     legend=False)   # or legend=True if you like
ax4.set_xlabel('Frequency (1/V)')
ax4.set_ylabel('Amplitude (A)')
plt.tight_layout()
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

print(f"Absolute maximum peak at {peak_freq:.3f} 1/V with amplitude {peak_amp:.3e} A")




































































# # 2) Run three sweeps, collecting their data
# data_runs = []
# offsets = [0.0, +0.01, -0.01]   # the three Vent offsets you used
# for off in offsets:
#     # position the gates as you did before
#     set_voltage_sim(sim, slot=4, target=Vexit_avg, step=0.1, delay=0.5)
#     set_voltage_sim(sim, slot=5, target=Vent_avg + off, step=0.1, delay=0.5)


#     label = f"Vent = {Vent_avg+off:.3f} V"
#     data_runs.append((exit_vals, currents, label))



# 3) One combined plot of exit vs current for all three:

# plt.figure()
# for exit_vals, currents, label in data_runs:
#     plt.plot(exit_vals, currents, label=label)

# base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
# plt.xlabel("Vexit (V)")
# plt.ylabel("Current (A)")
# plt.title("Exit‑voltage sweeps at three Vent offsets")
# plt.legend(loc="best")
# plt.grid(True)
# plt.tight_layout()
# plot_name_1 = "combined_exit_vs_current.png"
# path_to_save_1 = os.path.join(base_dir, plot_name_1)
# plt.savefig(path_to_save_1)
# plt.close()

# # 3) Combined plot: Vent vs current
# plt.figure()
# for ent_vals, currents, label in data_runs:
#     plt.plot(ent_vals, currents, label=label)
# base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
# plt.xlabel("Vent (V)")
# plt.ylabel("Current (A)")
# plt.title("Entry‑voltage sweeps at three Vent offsets")
# plt.legend(loc="best")
# plt.grid(True)
# plt.tight_layout()
# plot_name_2 = "combined_ent_vs_current.png"
# path_to_save_2 = os.path.join(base_dir, plot_name_2)
# plt.savefig(path_to_save_2)
# plt.close()



































































# # set_voltage_sim(sim, slot = 4, target = Vexit_avg, step = 0.1, delay = 0.5)
# # set_voltage_sim(sim, slot = 5, target = Vent_avg, step = 0.1, delay = 0.5)

# # ramp_two_channels(
# #     sim=sim,
# #     agilent=agilent,
# #     slot_exit=4,
# #     slot_ent=5,
# #     exit_target= Vexit_avg + 0.100,
# #     ent_target= Vent_avg + 0.100,
# #     step_exit=0.001,
# #     step_ent=0.001,
# #     delay=0.2
# # )

# # set_voltage_sim(sim, slot = 4, target = Vexit_avg, step = 0.1, delay = 0.5)
# # set_voltage_sim(sim, slot = 5, target = Vent_avg +0.02, step = 0.1, delay = 0.5)

# # ramp_two_channels(
# #     sim=sim,
# #     agilent=agilent,
# #     slot_exit=4,
# #     slot_ent=5,
# #     exit_target= Vexit_avg + 0.100,
# #     ent_target= Vent_avg + 0.100,
# #     step_exit=0.001,
# #     step_ent=0.001,
# #     delay=0.2
# # )

# # set_voltage_sim(sim, slot = 4, target = Vexit_avg, step = 0.1, delay = 0.5)
# # set_voltage_sim(sim, slot = 5, target = Vent_avg - 0.02, step = 0.1, delay = 0.5)

# # ramp_two_channels(
# #     sim=sim,
# #     agilent=agilent,
# #     slot_exit=4,
# #     slot_ent=5,
# #     exit_target= Vexit_avg + 0.100,
# #     ent_target= Vent_avg + 0.100,
# #     step_exit=0.001,
# #     step_ent=0.001,
# #     delay=0.2
# # )






























































# # # # def set_voltage_until_current(sim,
# # # #                               sr,
# # # #                               slot1: int,
# # # #                               slot2: int,
# # # #                               step: float,
# # # #                               delay: float,
# # # #                               current_target: float
# # # #                               ) -> Tuple[List[float], List[float], List[float]]:
# # # #     """
# # # #     Ramp the voltages on sim slots `slot1` and `slot2` in increments of `step`
# # # #     until the lock‑in amplifier’s R‑reading reaches or exceeds `current_target`.
# # # #     At each step, record V1, V2, and the current into lists.

# # # #     Returns:
# # # #         V1_list, V2_list, current_list
# # # #     """
# # # #     # ensure we’re reading the SR830’s internal reference
# # # #     sr.reference_source.set('internal')

# # # #     # initialize lists to store the ramp history
# # # #     V1_list: List[float] = []
# # # #     V2_list: List[float] = []
# # # #     current_list: List[float] = []

# # # #     # initial readings
# # # #     V1 = sim.get_voltage(slot1)
# # # #     V2 = sim.get_voltage(slot2)
# # # #     current = sr.R.get()

# # # #     V1_list.append(V1)
# # # #     V2_list.append(V2)
# # # #     current_list.append(current)

# # # #     # ramp until target current reached
# # # #     while current < current_target:
# # # #         # increment desired voltages
# # # #         V1 += step
# # # #         V2 += step

# # # #         # apply them to the instrument
# # # #         sim.set_voltage(slot1, V1)
# # # #         sim.set_voltage(slot2, V2)

# # # #         # read back actual voltages
# # # #         V1 = sim.get_voltage(slot1)
# # # #         V2 = sim.get_voltage(slot2)

# # # #         # wait and then read current
# # # #         time.sleep(delay)
# # # #         current = sr.R.get()

# # # #         # store this step’s readings
# # # #         V1_list.append(V1)
# # # #         V2_list.append(V2)
# # # #         current_list.append(current)

# # # #     return V1_list, V2_list, current_list


# # # # # usage example:
# # # # V1_vals, V2_vals, I_vals = set_voltage_until_current(sim, sr,
# # # #                                                      slot1=4,
# # # #                                                      slot2=5,
# # # #                                                      step=0.01,
# # # #                                                      delay=0.5,
# # # #                                                      current_target=5.0)

# # # # print('V1 ramp history:', V1_vals)
# # # # print('V2 ramp history:', V2_vals)
# # # # print('current history:', I_vals)

# # # # def ramp_voltages(sim,
# # # #                   slot1: int,
# # # #                   slot2: int,
# # # #                   step: float,
# # # #                   target_v1: float,
# # # #                   target_v2: float,
# # # #                   delay: float = 0.0
# # # #                  ) -> Tuple[List[float], List[float]]:
# # # #     """
# # # #     Simultaneously ramp voltages on sim slots `slot1` and `slot2` up to
# # # #     `target_v1` and `target_v2` in increments of `step`, recording each reading.

# # # #     Args:
# # # #         sim:           Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
# # # #         slot1:         Channel index for V1.
# # # #         slot2:         Channel index for V2.
# # # #         step:          Voltage increment (V) per iteration.
# # # #         target_v1:     Final voltage (V) to reach on slot1.
# # # #         target_v2:     Final voltage (V) to reach on slot2.
# # # #         delay:         Seconds to wait after setting voltages before reading back (default 0.0).

# # # #     Returns:
# # # #         v1_history:   List of all V1 readings (including initial and final).
# # # #         v2_history:   List of all V2 readings.
# # # #     """
# # # #     v1_history: List[float] = []
# # # #     v2_history: List[float] = []

# # # #     # Read initial voltages
# # # #     v1 = sim.get_voltage(slot1)
# # # #     v2 = sim.get_voltage(slot2)
# # # #     v1_history.append(v1)
# # # #     v2_history.append(v2)

# # # #     # Ramp until both channels have reached their targets
# # # #     while v1 < target_v1 or v2 < target_v2:
# # # #         if v1 < target_v1:
# # # #             v1 = min(v1 + step, target_v1)
# # # #             sim.set_voltage(slot1, v1)
# # # #         if v2 < target_v2:
# # # #             v2 = min(v2 + step, target_v2)
# # # #             sim.set_voltage(slot2, v2)

# # # #         if delay > 0:
# # # #             time.sleep(delay)

# # # #         # Read back actual voltages
# # # #         v1 = sim.get_voltage(slot1)
# # # #         v2 = sim.get_voltage(slot2)
# # # #         v1_history.append(v1)
# # # #         v2_history.append(v2)

# # # #     return v1_history, v2_history

# # # # # Ramp both channels from their current voltages up to 2.0 V and 3.5 V
# # # # # in 0.1 V steps, pausing 0.2 s between each step.
# # # # v1_vals, v2_vals = ramp_voltages(sim,
# # # #                                  slot1=4,
# # # #                                  slot2=5,
# # # #                                  step=0.1,
# # # #                                  target_v1=5.0,
# # # #                                  target_v2=3.0,
# # # #                                  delay=0.2)

# # # # print('V1 history:', v1_vals)
# # # # print('V2 history:', v2_vals)

# # # # import pandas as pd

# # # # # assume v1_vals and v2_vals are your two lists of equal length
# # # # df = pd.DataFrame({
# # # #     'v1_vals': v1_vals,
# # # #     'v2_vals': v2_vals
# # # # })

# # # # print(df)

# # # # df.plot(kind = 'scatter',x = 'v1_vals', y = 'v2_vals' )

# # # # directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\voltage_history.csv'
# # # # # write to CSV (no index column)
# # # # df.to_csv(directory, index=False)

# # ################ Unused lines of code #######################
# # # # Step (5ii): Stay at the exit gate value found in Step (5i) - (first_point[0]) and bring the entrance gate down to zero.
# # # set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.1, delay = 0.5)

# # # def sweep_gate_to_zero(sim,
# # #                        agilent,
# # #                        constant_slot: int,
# # #                        sweep_slot: int,
# # #                        step: float,
# # #                        delay: float
# # #                       ) -> Tuple[List[float], List[float]]:
# # #     """
# # #     Sweep the voltage on `sweep_slot` from its current value down to 0 V in decrements of `step`,
# # #     while holding the voltage on `constant_slot` constant. Record every sweep voltage and
# # #     the corresponding lock‑in current into two lists.

# # #     Args:
# # #         sim:             Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
# # #         constant_slot:   The channel index whose voltage remains unchanged.
# # #         sweep_slot:      The channel index to sweep down to zero.
# # #         step:            The decrement (V) per iteration.
# # #         delay:           Seconds to wait after setting voltage before reading back.

# # #     Returns:
# # #         vs_history:      List of all voltages read on `sweep_slot` (from start down to 0 V).
# # #         current_history: List of all corresponding SR830 R‑readings (in same order).
# # #     """

# # #     # storage for histories
# # #     vs_history: List[float] = []
# # #     current_history: List[float] = []

# # #     # initial readings
# # #     vs = sim.get_voltage(sweep_slot)
# # #     voltage = agilent.volt()
# # #     current = voltage*1e-6
# # #     vs_history.append(vs)
# # #     current_history.append(current)

# # #     # ramp down until we hit 0 V
# # #     while vs > 0:
# # #         # compute next voltage, clamp at zero
# # #         vs = max(vs - step, 0.0)
# # #         sim.set_voltage(sweep_slot, vs)

# # #         # optional delay
# # #         if delay > 0:
# # #             time.sleep(delay)

# # #         # read back actual sweep voltage and current
# # #         vs = sim.get_voltage(sweep_slot)
# # #         voltage = agilent.volt()
# # #         current = voltage*1e-6

# # #         # record
# # #         vs_history.append(vs)
# # #         current_history.append(current)

# # #     return vs_history, current_history

# # # sweep_1_Vent , sweep_1_current = sweep_gate_to_zero(sim, agilent, 4, 5, step=0.01, delay=0.1)

# # # print('Sweep voltages:', sweep_1_Vent)
# # # print('Currents:', sweep_1_current)

# # # data_sweep_1_Vent = pd.DataFrame({'sweep_1_Vent': sweep_1_Vent,'sweep_1_current': sweep_1_current})

# # # data_sweep_1_Vent.plot(kind = 'scatter',x = 'sweep_1_Vent', y = 'sweep_1_current' )

# # # directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\data_sweep_1_Vent.csv'
# # # data_sweep_1_Vent.to_csv(directory, index=False)

# # # data_sweep_1_Vent_zero = pd.read_csv(directory, sep = '\t')
# # # data_sweep_1_Vent_zero = data_sweep_1_Vent_zero[data_sweep_1_Vent_zero['sweep_1_current'] == 0.0]

# # # data_sweep_1_Vent_zero_sorted = data_sweep_1_Vent_zero.sort_values(by='sweep_1_Vent', ascending=False)
# # # data_sweep_1_Vent_zero_sorted = data_sweep_1_Vent_zero_sorted.head(20)

# # # Vent_mid =  data_sweep_1_Vent_zero_sorted['sweep_1_Vent'].mean()

# # # Vent_mid = round(Vent_mid, 3)

# # # print(f'The voltage of the entrance gate is {Vent_mid}.')

# # # sweep_2_Vexit , sweep_2_current = sweep_gate_to_zero(sim, agilent, 5, 4, step=0.01, delay=0.1)

# # ################# Unused lines of code - part II ##################
# # # print('Sweep voltages:', sweep_2_Vexit)
# # # print('Currents:', sweep_2_current)

# # # data_sweep_2_Vexit = pd.DataFrame({'sweep_2_Vexit': sweep_2_Vexit,'sweep_2_current': sweep_2_current})

# # # data_sweep_2_Vexit.plot(kind = 'scatter',x = 'sweep_2_Vexit', y = 'sweep_2_current' )

# # # directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\data_sweep_2_Vexit.csv'
# # # data_sweep_2_Vexit.to_csv(directory, index=False)

# # # data_sweep_2_Vexit_zero = pd.read_csv(directory, sep = '\t')
# # # data_sweep_2_Vexit_zero = data_sweep_2_Vexit_zero[data_sweep_2_Vexit_zero['sweep_2_current'] == 0.0]
# # # data_sweep_2_Vexit_zero_sorted = data_sweep_2_Vexit_zero.sort_values(by='sweep_2_Vexit', ascending=False)
# # # data_sweep_2_Vexit_zero_sorted = data_sweep_2_Vexit_zero_sorted.head(20)

# # # Vexit_mid =  data_sweep_2_Vexit_zero_sorted['sweep_2_Vexit'].mean()

# # # Vexit_mid = round(Vexit_mid, 3)

# # # print(f'The voltage of the entrance gate is {Vexit_mid}.')

# # ################ Unused lines of code - part III ###############

# # # This function was used to set the voltage of SR830.
# # # def set_voltage_sr(sr, target: float = 0.004, step: float = 0.01, delay: float = 0.5):
# # #     min_amp, max_amp = 0.004, 2.0
# # #     if target < min_amp:
# # #         print(f"Clamping target amplitude to minimum {min_amp} v.")
# # #         target = min_amp
# # #     elif target > max_amp:
# # #         print(f"Clamping target amplitude to minimum {max_amp} v.")
# # #         target = max_amp
    
# # #     current = sr.amplitude.get()

# # #     if abs(current - target) < 1e-12:
# # #         return

# # #     direction = np.sign(current - target)
# # #     setpoints = np.arange(current, target, direction * step)

# # #     for amp in setpoints:
# # #         sr.amplitude.set(float(amp))
# # #         time.sleep(delay)

# # #     sr.amplitude.set(target)

# # #     print(f'The amplitude of sr830 is set to {target}V.')

# # # set_voltage_sr(sr, target = 0.004, step = 0.01, delay = 0.5)

# # ############### Unused lines of code ####################
# # # def sweep_down_until_current(sim, agilent, slot1: int, slot2: int, step: float, delay: float,current_target: float, v_min: float,
# # # v_max: float) -> Tuple[List[float], List[float], List[float]]:
# # #     """
# # #     Sweep both gate voltages down from their initial values in steps of `step`
# # #     until the SR830’s R-reading ≥ current_target, while keeping voltages within
# # #     [v_min, v_max]. Record histories of V1, V2, and current.

# # #     Args:
# # #         sim:              Instrument with .get_voltage(slot) and .set_voltage(slot, V).
# # #         sr:               QCoDeS SR830 instance (use sr.R.get() for current).
# # #         slot1:            Index of the first gate.
# # #         slot2:            Index of the second gate.
# # #         step:             Voltage decrement per iteration (V).
# # #         delay:            Settle time after each step (s).
# # #         current_target:   Current (in A) at which to stop.
# # #         v_min:            Minimum allowed voltage (V).
# # #         v_max:            Maximum allowed voltage (V).

# # #     Returns:
# # #         v1_hist, v2_hist, current_hist:
# # #           Lists of the recorded V1, V2, and current values at each step.

# # #     Raises:
# # #         RuntimeError: If initial voltages are outside [v_min, v_max], or if
# # #                       both voltages reach v_min without achieving current_target.
# # #     """
# # #     # read initial voltages
# # #     v1 = sim.get_voltage(slot1)
# # #     v2 = sim.get_voltage(slot2)
# # #     if not (v_min <= v1 <= v_max) or not (v_min <= v2 <= v_max):
# # #         raise RuntimeError(
# # #             f"Initial voltages out of range: "
# # #             f"V1={v1:.3f} V, V2={v2:.3f} V not within [{v_min:.3f}, {v_max:.3f}] V"
# # #         )

# # #     # prepare history lists
# # #     v1_hist: List[float] = [v1]
# # #     v2_hist: List[float] = [v2]
# # #     voltage = agilent.volt()
# # #     current = voltage*1e-6

# # #     current_hist: List[float] = [current]

# # #     # sweep down
# # #     while current < current_target:
# # #         # compute next voltages
# # #         v1 = max(v1 - step, v_min)
# # #         v2 = max(v2 - step, v_min)

# # #         # apply and wait
# # #         sim.set_voltage(slot1, v1)
# # #         sim.set_voltage(slot2, v2)
# # #         time.sleep(delay)

# # #         # read back and record
# # #         v1 = sim.get_voltage(slot1)
# # #         v2 = sim.get_voltage(slot2)
# # #         voltage = agilent.volt()
# # #         current = voltage*1e-6

# # #         v1_hist.append(v1)
# # #         v2_hist.append(v2)
# # #         current_hist.append(current)

# # #         # check if we’ve exhausted the range
# # #         if v1 <= v_min and v2 <= v_min and current < current_target:
# # #             raise RuntimeError(
# # #                 f"Target current {current_target:.3e} A not reached; "
# # #                 f"both V1 and V2 have hit v_min={v_min:.3f} V "
# # #                 f"(final current={current:.3e} A)."
# # #             )

# # #     return v1_hist, v2_hist, current_hist

# # # v1_hist, v2_hist, i_hist = sweep_down_until_current(sim,agilent,4,5,0.01,delay=0.2,current_target=5e-9,v_min=0.0,
# # # v_max=2.0)

# # # print("V1 history:", v1_hist)
# # # print("V2 history:", v2_hist)
# # # print("Current history:", i_hist)

# # ############ Unused lines of code ##########
# # # def sweep_map(sim: Any, agilent: Any,
# # #               slot_exit: int,
# # #               slot_ent: int,
# # #               Vexit_avg: float,
# # #               Vent_avg: float,
# # #               Vexit_span: float,
# # #               Vent_span: float,
# # #               step_exit: float = 0.1,
# # #               step_ent: float = 0.1,
# # #               delay: float = 0.5,
# # #               output_csv: str = 'map_data.csv',
# # #               output_fig: str = 'map_colormap.png') -> None:
# # #     """
# # #     For each Vent in [Vent_avg, Vent_avg + Vent_span] (step_ent),
# # #     sweep Vexit in [Vexit_avg, Vexit_avg + Vexit_span] (step_exit),
# # #     record the measured current, save a CSV of (Vent, Vexit, current),
# # #     and produce a colormap image.

# # #     Args:
# # #         sim:            Instrument controller (with .set_voltage and .get_voltage).
# # #         agilent:        DMM instance (agilent.volt() → measured voltage).
# # #         slot_exit:      Channel index for Vexit.
# # #         slot_ent:       Channel index for Vent.
# # #         Vexit_avg:      Base Vexit.
# # #         Vent_avg:       Base Vent.
# # #         Vexit_span:     ΔVexit to sweep.
# # #         Vent_span:      ΔVent to sweep.
# # #         step_exit:      Vexit increment per point.
# # #         step_ent:       Vent increment per line.
# # #         delay:          Settle time after each set_voltage.
# # #         output_csv:     Path for CSV output.
# # #         output_fig:     Path for colormap PNG.
# # #     """
# # #     # 1) build axis arrays
# # #     Vexit_vals = np.arange(Vexit_avg,
# # #                            Vexit_avg + Vexit_span + 1e-9,
# # #                            step_exit)
# # #     Vent_vals = np.arange(Vent_avg,
# # #                           Vent_avg + Vent_span + 1e-9,
# # #                           step_ent)

# # #     # 2) prepare data structures
# # #     records = []
# # #     current_map = np.zeros((len(Vent_vals), len(Vexit_vals)))

# # #     # 3) nested sweep
# # #     for i, Vent in enumerate(Vent_vals):
# # #         sim.set_voltage(slot_ent, Vent)
# # #         time.sleep(delay)
# # #         for j, Vexit in enumerate(Vexit_vals):
# # #             sim.set_voltage(slot_exit, Vexit)
# # #             time.sleep(delay)

# # #             # measure current (convert voltage reading to amps if needed)
# # #             voltage = agilent.volt()
# # #             current = -1 * voltage * 1e-7

# # #             current_map[i, j] = current
# # #             records.append({'Vent': Vent,
# # #                             'Vexit': Vexit,
# # #                             'current': current})

# # #     # 4) save CSV
# # #     df = pd.DataFrame.from_records(records)
# # #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# # #     base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
# # #     filename_1 = f'{output_csv}_{timestamp}.csv'
# # #     directory = os.path.join(base_dir, filename_1)
# # #     df.to_csv(filename_1, index=False,float_format='%.3e')

# # #     # 5) plot and save colormap

# # #     qpc_voltage = sim.get_voltage(3)
# # #     legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

# # #     plt.figure()
# # #     X, Y = np.meshgrid(Vexit_vals, Vent_vals)
# # #     plt.pcolormesh(X, Y, current_map, shading='auto')
# # #     plt.xlabel('Vexit (V)')
# # #     plt.ylabel('Vent (V)')
# # #     plt.title('Current Map')
# # #     cbar = plt.colorbar()
# # #     cbar.set_label('Current (A)')

# # #     # 4) Save the plot next to the CSV
# # #     filename_2 = f'{output_fig}_{timestamp}.png'
# # #     plot_path = os.path.join(base_dir, filename_2)
# # #     plt.savefig(plot_path)
# # #     plt.close()

# # # sweep_map(sim,agilent,4,5,Vexit_avg,Vent_avg,0.100,0.100,0.001,0.001,0.1,output_csv='qpc_map.csv',output_fig='qpc_map.png')



# # ############# Unused lines of code ############
# # # def ramp_two_channels(sim: Any,
# # #                       agilent: Any,
# # #                       slot_exit: int,
# # #                       slot_ent: int,
# # #                       exit_target: float,
# # #                       ent_target: float,
# # #                       step_exit: float,
# # #                       step_ent: float,
# # #                       delay: float = 0.5
# # #                      ) -> Tuple[List[float], List[float], List[float]]:
# # #     """
# # #     Simultaneously ramp two gates from their current voltages up to the
# # #     specified targets, recording voltages and current.

# # #     Args:
# # #         sim:         Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
# # #         agilent:     DMM instance where agilent.volt() returns the measured voltage (V).
# # #         slot_exit:   Channel index for the “exit” gate.
# # #         slot_ent:    Channel index for the “ent” gate.
# # #         exit_target: Final voltage (V) for slot_exit.
# # #         ent_target:  Final voltage (V) for slot_ent.
# # #         step_exit:   Voltage increment per iteration for slot_exit.
# # #         step_ent:    Voltage increment per iteration for slot_ent.
# # #         delay:       Seconds to wait after setting voltages.

# # #     Returns:
# # #         (exit_vals, ent_vals, current_vals):
# # #            - exit_vals:  list of voltages read from slot_exit at each step
# # #            - ent_vals:   list of voltages read from slot_ent at each step
# # #            - current_vals: list of currents (in A) measured at each step
# # #     """
# # #     exit_vals: List[float] = []
# # #     ent_vals: List[float] = []
# # #     current_vals: List[float] = []

# # #     # read starting voltages
# # #     v_exit = sim.get_voltage(slot_exit)
# # #     v_ent  = sim.get_voltage(slot_ent)

# # #     # record initial point
# # #     exit_vals.append(v_exit)
# # #     ent_vals.append(v_ent)
# # #     voltage = agilent.volt()
# # #     current = -1 * voltage * 1e-7
# # #     current_vals.append(current)

# # #     # ramp until both targets are reached
# # #     while v_exit < exit_target or v_ent < ent_target:
# # #         if v_exit < exit_target:
# # #             v_exit = min(v_exit + step_exit, exit_target)
# # #             sim.set_voltage(slot_exit, v_exit)
# # #         if v_ent < ent_target:
# # #             v_ent = min(v_ent + step_ent, ent_target)
# # #             sim.set_voltage(slot_ent, v_ent)

# # #         time.sleep(delay)

# # #         # read back actual voltages & current
# # #         v_exit = sim.get_voltage(slot_exit)
# # #         v_ent  = sim.get_voltage(slot_ent)
# # #         voltage = agilent.volt()
# # #         current = -1 * voltage * 1e-7
# # #         exit_vals.append(v_exit)
# # #         ent_vals.append(v_ent)
# # #         current_vals.append(current)

# # #     df = pd.DataFrame({'V_exit (V)': exit_vals, 'V_ent (V)': ent_vals, 'Current (A)': current_vals})
# # #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# # #     base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
# # #     filename_1 = f'sweep_Vent_Vexit_{timestamp}.csv'
# # #     directory = os.path.join(base_dir, filename_1)
# # #     df.to_csv(directory, index=False,float_format='%.3e')

# # #     # 3) Scatter plot of the first two columns
# # #     x = df.iloc[:, 0]
# # #     y = df.iloc[:, 1]
# # #     z = df.iloc[:, 2]
# # #     x_label = df.columns[0]
# # #     y_label = df.columns[1]
# # #     z_label = df.columns[2]

# # #     # Read QPC gate voltage from channel 3
# # #     qpc_voltage = sim.get_voltage(3)
# # #     legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

# # #     plt.figure()
# # #     plt.scatter(x, z, label=legend_label)
# # #     plt.xlabel(x_label)
# # #     plt.ylabel(z_label)
# # #     plt.title(f"{z_label} vs {x_label}")
# # #     plt.legend(loc='best')

# # #     # 4) Save the plot next to the CSV
# # #     filename_2 = f'sweep_Vent_Vexit_VexitvsI_{timestamp}_scatter.png'
# # #     plot_path = os.path.join(base_dir, filename_2)
# # #     plt.savefig(plot_path)
# # #     plt.close()

# # #     plt.figure()
# # #     plt.scatter(y, z, label=legend_label)
# # #     plt.xlabel(y_label)
# # #     plt.ylabel(z_label)
# # #     plt.title(f"{z_label} vs {y_label}")
# # #     plt.legend(loc='best')

# # #     # 4) Save the plot next to the CSV
# # #     filename_3 = f'sweep_Vent_Vexit_VentvsI_{timestamp}_scatter.png'
# # #     plot_path_1 = os.path.join(base_dir, filename_3)
# # #     plt.savefig(plot_path_1)
# # #     plt.close()

# # # ramp_two_channels(
# # #     sim=sim,
# # #     agilent=agilent,
# # #     slot_exit=4,
# # #     slot_ent=5,
# # #     exit_target=0.819500,
# # #     ent_target=0.791500,
# # #     step_exit=0.001,
# # #     step_ent=0.001,
# # #     delay=0.2
# # # )
