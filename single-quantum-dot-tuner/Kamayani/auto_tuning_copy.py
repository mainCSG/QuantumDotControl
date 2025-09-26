import qcodes
from qcodes import instrument_drivers
from qcodes.dataset import do0d, load_or_create_experiment
from qcodes.instrument import Instrument
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
from qcodes.validators import Numbers
from qcodes import Parameter
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

    # Build DataFrame and write CSV

    df = pd.DataFrame({name: list(values) for name, values in arrays.items()})

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
    filename_1 = f'{filename}_{timestamp}.csv'
    directory = os.path.join(base_dir, filename_1)
    df.to_csv(directory, index=False,float_format='%.3e')

    # 3) Scatter plot of the first two columns
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_label = df.columns[0]
    y_label = df.columns[1]

    # Read QPC gate voltage from channel 3
    qpc_voltage = sim.get_voltage(3)
    legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

    plt.figure()
    plt.scatter(x, y, label=legend_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_2 = f'{filename}_{timestamp}_scatter.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()

# The following gates are assumed to be connected to the following channels in sim900:
# Channel 1: Top gate
# Channel 2: QPC bottom
# Channel 3: QPC top
# Channel 4: Exit gate
# Channel 5: Entrance gate
# Channel 6: Source-drain bias (currently set to zero)

# Connecting to SR830 and SIM900
# sr = SR830("lockin", "GPIB0::8::INSTR") -> We are not using SR830 anymore, instead, we are reading current from the agilent. 
sim = SIM928("sim900", "GPIB0::3::INSTR", slot_names={i: f'ch{i}' for i in range (1,9)})
agilent = Agilent34401A("agilent", "GPIB0::21::INSTR")
                                                                                                                                                                                                                                                                                                                                                      
# Step 1: Grounding the device 

# def ground_device_sim(sim, slot: int, step: float = 0.1, delay: float = 0.5):
#     V0 = sim.get_voltage(slot)
#     if abs(V0) < 1e-12:
#         return
    
#     direction = -np.sign(V0)
#     setpoints = np.arange(V0, 0, direction * step)

#     for V in setpoints:
#         sim.set_voltage(slot, float(V))
#         time.sleep(delay)

#     sim.set_voltage(slot, 0.0)
#     time.sleep(delay)

# for slot in reversed(range(1,9)):
#     ground_device_sim(sim, slot, step= 0.01, delay = 0.5)

# print('The voltage of all channels in sim900 is set to 0V.')
# print('Step 1: Grounding the devices: completed.')

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

# set_voltage_sim(sim, slot = 1, target = 5.0, step = 0.1, delay = 0.5)

# print('Step 2: Setting up the voltage of the top gate (channel 1 of sim900): completed.')

# # Step 3: Setting up the voltage of both of the QPC gates (channel 2 and channel 3 in sim900)

# def simultaneous_set_voltage_sim(sim, slot1: int, target1: float, slot2: int, target2: float, step: float = 0.01, delay: float = 0.5):

#     V1_0 = sim.get_voltage(slot1)
#     V2_0 = sim.get_voltage(slot2)

#     if abs(V1_0 - target1) < 1e-12 and abs(V2_0 - target2) < 1e-12:
#         return
    
#     dir1 = np.sign(target1 - V1_0)
#     dir2 = np.sign(target2 - V2_0)

#     sp1 = np.arange(V1_0, target1, dir1 * step)
#     sp2 = np.arange(V2_0, target2, dir2 * step)

#     if sp1.size == 0 or sp1[-1] != target1:
#         sp1 = np.append(sp1, target1)
#     if sp2.size == 0 or sp2[-1] != target2:
#         sp2 = np.append(sp2, target2)

#     for v1, v2 in zip(sp1,sp2):
#         sim.set_voltage(slot1, float(v1))
#         sim.set_voltage(slot2, float(v2))
#         time.sleep(delay)

#     sim.set_voltage(slot1, target1)
#     sim.set_voltage(slot2, target2)
#     time.sleep(delay)

#     print(f'The voltage of the channel {slot1} and {slot2} of sim900 is set to {target1: .6f} V and {target2: .6f} V.')

# simultaneous_set_voltage_sim(sim, 2, 0.2, 3, 0.2, step = 0.01, delay = 0.5)

# print('Step 3: Setting up the voltage of the QPC gates (channels 2 and 3 of sim900): completed.')

# # Step 4: Setting up the voltage of channel 7 of sim900 (which replaces setting the voltage of SR830)

# #set_voltage_sr(sr, target = 1.0, step = 0.01, delay = 0.5)
# # We are using a 1 M ohm 100 k ohm resistor that will give us 100 uV input. 
# set_voltage_sim(sim, slot = 6, target = 0.001, step = 0.0001, delay = 0.5)

# print('Step 4: Setting up the voltage of channel 6 of sim900: completed.')

# # Step 5: Sweeping the entrance and the exit gates to find the region of operation. 

# # Step (5i): Bringing the exit and the entrance gates symmetrically until current = 5nA. 

# def set_voltage_until_current(sim, agilent,slot1: int, slot2: int, step: float, delay: float, current_target: float):

#     V1 = sim.get_voltage(slot1)
#     V2 = sim.get_voltage(slot2)
#     voltage = agilent.volt()
#     current = -1*voltage*1e-7

#     v1_history: List[float] = []
#     v2_history: List[float] = []
#     current_history: List[float] = []

#     v1_history.append(V1)
#     v2_history.append(V2)
#     current_history.append(current)

#     while current < current_target:

#         if V1 >= 0.95 and V2 >= 0.95:
#             raise RuntimeError(
#                 f"Failed to reach {current_target} A: "
#                 f"V1 and V2 have both reached 1.1 V (current = {current:.3e} A)."
#             )
        
#         V1 += step
#         V2 += step 
#         sim.set_voltage(slot1, V1)
#         sim.set_voltage(slot2, V2)
#         time.sleep(delay)
#         voltage = agilent.volt()
#         current = -1*voltage*1e-7

#         v1_history.append(V1)
#         v2_history.append(V2)
#         current_history.append(current)

#     data_first_point = pd.DataFrame({'V_slot1': v1_history,'V_slot2': v2_history, 'I (A)': current_history})

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
#     filename = f'data_first_point_{timestamp}.csv'
#     directory = os.path.join(base_dir, filename)

#     data_first_point.to_csv(directory, index=False, float_format='%.3e')
    
#     V1_a = sim.get_voltage(slot1)
#     V2_b = sim.get_voltage(slot2)

#     first_point = [V1_a, V2_b]
#     return first_point

# first_point = set_voltage_until_current(sim,agilent,4, 5, 0.003, 0.2, 0.9e-9)

# print(f'first point: {first_point}')

# # # Step (5ii): Stay at the exit gate value found in Step (5i) - (first_point[0]) and bring the entrance gate down until
# # minimum current is reached and then sweep for a given number of points.

# set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.001, delay = 0.2)

# import time
# from typing import List, Tuple

# def sweep_gate_until_min_current(sim,
#                                  agilent,
#                                  constant_slot: int,
#                                  sweep_slot: int,
#                                  step: float,
#                                  delay: float,
#                                  min_current: float,
#                                  num_points: int
#                                 ) -> Tuple[List[float], List[float], float]:
#     """
#     Sweep the voltage on `sweep_slot` from its current value down in decrements of `step`
#     until the measured current falls to or below `min_current`. After that,
#     continue sweeping for `num_points` additional steps. The voltage on
#     `constant_slot` remains fixed throughout.

#     Returns the full voltage and current histories, plus the average of the
#     voltages recorded during the final `num_points` sweep.

#     Args:
#         sim:             Instrument with .get_voltage(slot) and .set_voltage(slot, V).
#         agilent:         DMM instance where agilent.volt() returns the measured voltage (V).
#         constant_slot:   Channel index to hold constant.
#         sweep_slot:      Channel index to sweep.
#         step:            Voltage decrement per iteration (V).
#         delay:           Seconds to wait after setting voltage before reading back.
#         min_current:     Current threshold (A) to trigger the extra sweep.
#         num_points:      Number of additional sweep points after threshold reached.

#     Returns:
#         vs_history:      List of all voltages read on `sweep_slot`.
#         current_history: List of all corresponding currents (in A) at each step.
#         avg_extra_vs:    Float — the average of the last `num_points` voltages.
#     """
#     vs_history: List[float] = []
#     current_history: List[float] = []
#     extra_vs: List[float] = []

#     # Initial readings
#     vs = sim.get_voltage(sweep_slot)
#     voltage = agilent.volt()
#     current = -1 * voltage * 1e-7
#     vs_history.append(vs)
#     current_history.append(current)

#     # 1) Sweep down until current ≤ min_current (or vs reaches 0)
#     while current > min_current and vs > 0:
#         vs = max(vs - step, 0.0)
#         sim.set_voltage(sweep_slot, vs)
#         time.sleep(delay)

#         vs = sim.get_voltage(sweep_slot)
#         voltage = agilent.volt()
#         current = -1 * voltage * 1e-7

#         vs_history.append(vs)
#         current_history.append(current)

#     # 2) Continue sweeping for num_points more steps, record voltages in extra_vs
#     for _ in range(num_points):
#         if vs > 0:
#             vs = max(vs - step, 0.0)
#             sim.set_voltage(sweep_slot, vs)

#         time.sleep(delay)
#         vs = sim.get_voltage(sweep_slot)
#         voltage = agilent.volt()
#         current = -1 * voltage * 1e-7

#         vs_history.append(vs)
#         current_history.append(current)
#         extra_vs.append(vs)

#     # Compute average of the extra sweep voltages
#     avg_extra_vs = sum(extra_vs) / len(extra_vs) if extra_vs else 0.0

#     return vs_history, current_history, avg_extra_vs

# sweep_1_Vent, sweep_1_Vent_current, Vent_avg = sweep_gate_until_min_current(sim,agilent,4,5,0.001,0.2,9.0e-13,40)

# save_arrays_to_csv('sweep_1_Vent',sim,                    
#                    **{
#         'V_ent (V)': sweep_1_Vent,
#         'Current (A)': sweep_1_Vent_current
#     })
# print(f'Vent : {Vent_avg}')

# # Step (5iii): Go back at the values of exit and entrance gates found in Step (5i) (first_point).

# set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.001, delay = 0.2)
# set_voltage_sim(sim, slot = 5, target = first_point[1], step = 0.001, delay = 0.2)

# # Step (5iv): Stay at the entrance gate value set in Step (5iii) - (first_point[1]) and bring the exit gate down until
# # minimum current is reached and then sweep for a given number of points.

# set_voltage_sim(sim, slot = 5, target = first_point[1], step = 0.001, delay = 0.2)

# sweep_2_Vexit, sweep_2_Vexit_current, Vexit_avg = sweep_gate_until_min_current(sim,agilent,5,4,0.001,0.2,9.0e-13,40)

# save_arrays_to_csv('sweep_2_Vexit',sim, 
#                    **{
#         'V_exit (V)': sweep_2_Vexit,
#         'Current (A)': sweep_2_Vexit_current
#     })
# print(f'Vexit : {Vexit_avg}')

# # Step (5v): Set the values of exit and entrance gates found as Vexit_avg and Vent_avg.

# set_voltage_sim(sim, slot = 4, target = Vexit_avg, step = 0.1, delay = 0.5)
# set_voltage_sim(sim, slot = 5, target = Vent_avg, step = 0.1, delay = 0.5)

# # # Step (5vi): Sweep the values V_exit for each V_ent and get a colormap.

# def sweep_map(sim: Any,
#               agilent: Any,
#               slot_exit: int,
#               slot_ent: int,
#               Vexit_avg: float,
#               Vent_avg: float,
#               Vexit_span: float,
#               Vent_span: float,
#               step_exit: float = 0.1,
#               step_ent: float = 0.1,
#               delay: float = 0.5,
#               output_csv: str = 'map_data.csv',
#               output_fig: str = 'map_colormap.png') -> None:
#     """
#     For each Vent in [Vent_avg, Vent_avg + Vent_span] (step_ent):
#       1) sweep Vexit from Vexit_avg → Vexit_avg+Vexit_span (step_exit),
#       2) record (Vent, Vexit, current),
#       3) then ramp Vexit back down from max → Vexit_avg (step_exit),
#     save all to CSV and draw a colormap.

#     Args:
#       sim:       Instrument controller (with .set_voltage / .get_voltage).
#       agilent:   DMM instance (agilent.volt() → measured voltage).
#       slot_exit: Channel index for Vexit.
#       slot_ent:  Channel index for Vent.
#       Vexit_avg: Base Vexit.
#       Vent_avg:  Base Vent.
#       Vexit_span: ΔVexit to sweep.
#       Vent_span:  ΔVent to sweep.
#       step_exit: Vexit increment/decrement per point.
#       step_ent:  Vent increment per line.
#       delay:     Settle time after each set_voltage.
#       output_csv: Path for CSV output.
#       output_fig: Path for colormap PNG.
#     """
#     # build the sweep axes
#     Vexit_vals = np.arange(Vexit_avg,
#                            Vexit_avg + Vexit_span + 1e-9,
#                            step_exit)
#     Vent_vals = np.arange(Vent_avg,
#                           Vent_avg + Vent_span + 1e-9,
#                           step_ent)

#     records = []
#     current_map = np.zeros((len(Vent_vals), len(Vexit_vals)))

#     for i, Vent in enumerate(Vent_vals):
#         # set Vent and wait
#         sim.set_voltage(slot_ent, Vent)
#         time.sleep(delay)

#         # forward sweep of Vexit
#         for j, Vexit in enumerate(Vexit_vals):
#             sim.set_voltage(slot_exit, Vexit)
#             time.sleep(delay)

#             voltage = agilent.volt()
#             current = voltage * 1e-6

#             current_map[i, j] = current
#             records.append({
#                 'Vent (V)': Vent,
#                 'Vexit (V)': Vexit,
#                 'Current (A)': current
#             })

#         # now ramp Vexit back down to Vexit_avg
#         # skip the first element since it's the starting point
#         for Vexit_down in reversed(Vexit_vals[:-1]):
#             sim.set_voltage(slot_exit, Vexit_down)
#             time.sleep(delay)

#     # 4) save CSV
#     df = pd.DataFrame.from_records(records)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
#     filename_1 = f'{output_csv}_{timestamp}.csv'
#     directory = os.path.join(base_dir, filename_1)
#     df.to_csv(filename_1, index=False,float_format='%.3e')

#     # 5) plot and save colormap

#     qpc_voltage = sim.get_voltage(3)
#     legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

#     plt.figure()
#     X, Y = np.meshgrid(Vexit_vals, Vent_vals)
#     plt.pcolormesh(X, Y, current_map, shading='auto')
#     plt.xlabel('Vexit (V)')
#     plt.ylabel('Vent (V)')
#     plt.title('Current Map')
#     cbar = plt.colorbar()
#     cbar.set_label('Current (A)')

#     # 4) Save the plot next to the CSV
#     filename_2 = f'{output_fig}_{timestamp}.png'
#     plot_path = os.path.join(base_dir, filename_2)
#     plt.savefig(plot_path)
#     plt.close()

# sweep_map(sim,agilent,4,5,Vexit_avg,Vent_avg,0.100,0.100,0.001,0.001,0.1,output_csv='qpc_map.csv',output_fig='qpc_map.png')


set_voltage_sim(sim, slot = 4, target = 0.719500, step = 0.1, delay = 0.5)
set_voltage_sim(sim, slot = 5, target = 0.691500, step = 0.1, delay = 0.5)

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
        print(f"[{iteration}] Vexit={v_exit:.3f}, Vent={v_ent:.3f}")

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
    legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

    plt.figure()
    plt.scatter(x, z, label=legend_label)
    plt.xlabel(x_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} vs {x_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_2 = f'sweep_Vent_Vexit_VexitvsI_{timestamp}_scatter.png'
    plot_path = os.path.join(base_dir, filename_2)
    plt.savefig(plot_path)
    plt.close()

    plt.figure()
    plt.scatter(y, z, label=legend_label)
    plt.xlabel(y_label)
    plt.ylabel(z_label)
    plt.title(f"{z_label} vs {y_label}")
    plt.legend(loc='best')

    # 4) Save the plot next to the CSV
    filename_3 = f'sweep_Vent_Vexit_VentvsI_{timestamp}_scatter.png'
    plot_path_1 = os.path.join(base_dir, filename_3)
    plt.savefig(plot_path_1)
    plt.close()

ramp_two_channels(
    sim=sim,
    agilent=agilent,
    slot_exit=4,
    slot_ent=5,
    exit_target=0.819500,
    ent_target=0.791500,
    step_exit=0.001,
    step_ent=0.001,
    delay=0.2
)




























































# # def set_voltage_until_current(sim,
# #                               sr,
# #                               slot1: int,
# #                               slot2: int,
# #                               step: float,
# #                               delay: float,
# #                               current_target: float
# #                               ) -> Tuple[List[float], List[float], List[float]]:
# #     """
# #     Ramp the voltages on sim slots `slot1` and `slot2` in increments of `step`
# #     until the lock‑in amplifier’s R‑reading reaches or exceeds `current_target`.
# #     At each step, record V1, V2, and the current into lists.

# #     Returns:
# #         V1_list, V2_list, current_list
# #     """
# #     # ensure we’re reading the SR830’s internal reference
# #     sr.reference_source.set('internal')

# #     # initialize lists to store the ramp history
# #     V1_list: List[float] = []
# #     V2_list: List[float] = []
# #     current_list: List[float] = []

# #     # initial readings
# #     V1 = sim.get_voltage(slot1)
# #     V2 = sim.get_voltage(slot2)
# #     current = sr.R.get()

# #     V1_list.append(V1)
# #     V2_list.append(V2)
# #     current_list.append(current)

# #     # ramp until target current reached
# #     while current < current_target:
# #         # increment desired voltages
# #         V1 += step
# #         V2 += step

# #         # apply them to the instrument
# #         sim.set_voltage(slot1, V1)
# #         sim.set_voltage(slot2, V2)

# #         # read back actual voltages
# #         V1 = sim.get_voltage(slot1)
# #         V2 = sim.get_voltage(slot2)

# #         # wait and then read current
# #         time.sleep(delay)
# #         current = sr.R.get()

# #         # store this step’s readings
# #         V1_list.append(V1)
# #         V2_list.append(V2)
# #         current_list.append(current)

# #     return V1_list, V2_list, current_list


# # # usage example:
# # V1_vals, V2_vals, I_vals = set_voltage_until_current(sim, sr,
# #                                                      slot1=4,
# #                                                      slot2=5,
# #                                                      step=0.01,
# #                                                      delay=0.5,
# #                                                      current_target=5.0)

# # print('V1 ramp history:', V1_vals)
# # print('V2 ramp history:', V2_vals)
# # print('current history:', I_vals)

# # def ramp_voltages(sim,
# #                   slot1: int,
# #                   slot2: int,
# #                   step: float,
# #                   target_v1: float,
# #                   target_v2: float,
# #                   delay: float = 0.0
# #                  ) -> Tuple[List[float], List[float]]:
# #     """
# #     Simultaneously ramp voltages on sim slots `slot1` and `slot2` up to
# #     `target_v1` and `target_v2` in increments of `step`, recording each reading.

# #     Args:
# #         sim:           Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
# #         slot1:         Channel index for V1.
# #         slot2:         Channel index for V2.
# #         step:          Voltage increment (V) per iteration.
# #         target_v1:     Final voltage (V) to reach on slot1.
# #         target_v2:     Final voltage (V) to reach on slot2.
# #         delay:         Seconds to wait after setting voltages before reading back (default 0.0).

# #     Returns:
# #         v1_history:   List of all V1 readings (including initial and final).
# #         v2_history:   List of all V2 readings.
# #     """
# #     v1_history: List[float] = []
# #     v2_history: List[float] = []

# #     # Read initial voltages
# #     v1 = sim.get_voltage(slot1)
# #     v2 = sim.get_voltage(slot2)
# #     v1_history.append(v1)
# #     v2_history.append(v2)

# #     # Ramp until both channels have reached their targets
# #     while v1 < target_v1 or v2 < target_v2:
# #         if v1 < target_v1:
# #             v1 = min(v1 + step, target_v1)
# #             sim.set_voltage(slot1, v1)
# #         if v2 < target_v2:
# #             v2 = min(v2 + step, target_v2)
# #             sim.set_voltage(slot2, v2)

# #         if delay > 0:
# #             time.sleep(delay)

# #         # Read back actual voltages
# #         v1 = sim.get_voltage(slot1)
# #         v2 = sim.get_voltage(slot2)
# #         v1_history.append(v1)
# #         v2_history.append(v2)

# #     return v1_history, v2_history

# # # Ramp both channels from their current voltages up to 2.0 V and 3.5 V
# # # in 0.1 V steps, pausing 0.2 s between each step.
# # v1_vals, v2_vals = ramp_voltages(sim,
# #                                  slot1=4,
# #                                  slot2=5,
# #                                  step=0.1,
# #                                  target_v1=5.0,
# #                                  target_v2=3.0,
# #                                  delay=0.2)

# # print('V1 history:', v1_vals)
# # print('V2 history:', v2_vals)

# # import pandas as pd

# # # assume v1_vals and v2_vals are your two lists of equal length
# # df = pd.DataFrame({
# #     'v1_vals': v1_vals,
# #     'v2_vals': v2_vals
# # })

# # print(df)

# # df.plot(kind = 'scatter',x = 'v1_vals', y = 'v2_vals' )

# # directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\voltage_history.csv'
# # # write to CSV (no index column)
# # df.to_csv(directory, index=False)

################ Unused lines of code #######################
# # Step (5ii): Stay at the exit gate value found in Step (5i) - (first_point[0]) and bring the entrance gate down to zero.
# set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.1, delay = 0.5)

# def sweep_gate_to_zero(sim,
#                        agilent,
#                        constant_slot: int,
#                        sweep_slot: int,
#                        step: float,
#                        delay: float
#                       ) -> Tuple[List[float], List[float]]:
#     """
#     Sweep the voltage on `sweep_slot` from its current value down to 0 V in decrements of `step`,
#     while holding the voltage on `constant_slot` constant. Record every sweep voltage and
#     the corresponding lock‑in current into two lists.

#     Args:
#         sim:             Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
#         constant_slot:   The channel index whose voltage remains unchanged.
#         sweep_slot:      The channel index to sweep down to zero.
#         step:            The decrement (V) per iteration.
#         delay:           Seconds to wait after setting voltage before reading back.

#     Returns:
#         vs_history:      List of all voltages read on `sweep_slot` (from start down to 0 V).
#         current_history: List of all corresponding SR830 R‑readings (in same order).
#     """

#     # storage for histories
#     vs_history: List[float] = []
#     current_history: List[float] = []

#     # initial readings
#     vs = sim.get_voltage(sweep_slot)
#     voltage = agilent.volt()
#     current = voltage*1e-6
#     vs_history.append(vs)
#     current_history.append(current)

#     # ramp down until we hit 0 V
#     while vs > 0:
#         # compute next voltage, clamp at zero
#         vs = max(vs - step, 0.0)
#         sim.set_voltage(sweep_slot, vs)

#         # optional delay
#         if delay > 0:
#             time.sleep(delay)

#         # read back actual sweep voltage and current
#         vs = sim.get_voltage(sweep_slot)
#         voltage = agilent.volt()
#         current = voltage*1e-6

#         # record
#         vs_history.append(vs)
#         current_history.append(current)

#     return vs_history, current_history

# sweep_1_Vent , sweep_1_current = sweep_gate_to_zero(sim, agilent, 4, 5, step=0.01, delay=0.1)

# print('Sweep voltages:', sweep_1_Vent)
# print('Currents:', sweep_1_current)

# data_sweep_1_Vent = pd.DataFrame({'sweep_1_Vent': sweep_1_Vent,'sweep_1_current': sweep_1_current})

# data_sweep_1_Vent.plot(kind = 'scatter',x = 'sweep_1_Vent', y = 'sweep_1_current' )

# directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\data_sweep_1_Vent.csv'
# data_sweep_1_Vent.to_csv(directory, index=False)

# data_sweep_1_Vent_zero = pd.read_csv(directory, sep = '\t')
# data_sweep_1_Vent_zero = data_sweep_1_Vent_zero[data_sweep_1_Vent_zero['sweep_1_current'] == 0.0]

# data_sweep_1_Vent_zero_sorted = data_sweep_1_Vent_zero.sort_values(by='sweep_1_Vent', ascending=False)
# data_sweep_1_Vent_zero_sorted = data_sweep_1_Vent_zero_sorted.head(20)

# Vent_mid =  data_sweep_1_Vent_zero_sorted['sweep_1_Vent'].mean()

# Vent_mid = round(Vent_mid, 3)

# print(f'The voltage of the entrance gate is {Vent_mid}.')

# sweep_2_Vexit , sweep_2_current = sweep_gate_to_zero(sim, agilent, 5, 4, step=0.01, delay=0.1)

################# Unused lines of code - part II ##################
# print('Sweep voltages:', sweep_2_Vexit)
# print('Currents:', sweep_2_current)

# data_sweep_2_Vexit = pd.DataFrame({'sweep_2_Vexit': sweep_2_Vexit,'sweep_2_current': sweep_2_current})

# data_sweep_2_Vexit.plot(kind = 'scatter',x = 'sweep_2_Vexit', y = 'sweep_2_current' )

# directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\data_sweep_2_Vexit.csv'
# data_sweep_2_Vexit.to_csv(directory, index=False)

# data_sweep_2_Vexit_zero = pd.read_csv(directory, sep = '\t')
# data_sweep_2_Vexit_zero = data_sweep_2_Vexit_zero[data_sweep_2_Vexit_zero['sweep_2_current'] == 0.0]
# data_sweep_2_Vexit_zero_sorted = data_sweep_2_Vexit_zero.sort_values(by='sweep_2_Vexit', ascending=False)
# data_sweep_2_Vexit_zero_sorted = data_sweep_2_Vexit_zero_sorted.head(20)

# Vexit_mid =  data_sweep_2_Vexit_zero_sorted['sweep_2_Vexit'].mean()

# Vexit_mid = round(Vexit_mid, 3)

# print(f'The voltage of the entrance gate is {Vexit_mid}.')

################ Unused lines of code - part III ###############

# This function was used to set the voltage of SR830.
# def set_voltage_sr(sr, target: float = 0.004, step: float = 0.01, delay: float = 0.5):
#     min_amp, max_amp = 0.004, 2.0
#     if target < min_amp:
#         print(f"Clamping target amplitude to minimum {min_amp} v.")
#         target = min_amp
#     elif target > max_amp:
#         print(f"Clamping target amplitude to minimum {max_amp} v.")
#         target = max_amp
    
#     current = sr.amplitude.get()

#     if abs(current - target) < 1e-12:
#         return

#     direction = np.sign(current - target)
#     setpoints = np.arange(current, target, direction * step)

#     for amp in setpoints:
#         sr.amplitude.set(float(amp))
#         time.sleep(delay)

#     sr.amplitude.set(target)

#     print(f'The amplitude of sr830 is set to {target}V.')

# set_voltage_sr(sr, target = 0.004, step = 0.01, delay = 0.5)

############### Unused lines of code ####################
# def sweep_down_until_current(sim, agilent, slot1: int, slot2: int, step: float, delay: float,current_target: float, v_min: float,
# v_max: float) -> Tuple[List[float], List[float], List[float]]:
#     """
#     Sweep both gate voltages down from their initial values in steps of `step`
#     until the SR830’s R-reading ≥ current_target, while keeping voltages within
#     [v_min, v_max]. Record histories of V1, V2, and current.

#     Args:
#         sim:              Instrument with .get_voltage(slot) and .set_voltage(slot, V).
#         sr:               QCoDeS SR830 instance (use sr.R.get() for current).
#         slot1:            Index of the first gate.
#         slot2:            Index of the second gate.
#         step:             Voltage decrement per iteration (V).
#         delay:            Settle time after each step (s).
#         current_target:   Current (in A) at which to stop.
#         v_min:            Minimum allowed voltage (V).
#         v_max:            Maximum allowed voltage (V).

#     Returns:
#         v1_hist, v2_hist, current_hist:
#           Lists of the recorded V1, V2, and current values at each step.

#     Raises:
#         RuntimeError: If initial voltages are outside [v_min, v_max], or if
#                       both voltages reach v_min without achieving current_target.
#     """
#     # read initial voltages
#     v1 = sim.get_voltage(slot1)
#     v2 = sim.get_voltage(slot2)
#     if not (v_min <= v1 <= v_max) or not (v_min <= v2 <= v_max):
#         raise RuntimeError(
#             f"Initial voltages out of range: "
#             f"V1={v1:.3f} V, V2={v2:.3f} V not within [{v_min:.3f}, {v_max:.3f}] V"
#         )

#     # prepare history lists
#     v1_hist: List[float] = [v1]
#     v2_hist: List[float] = [v2]
#     voltage = agilent.volt()
#     current = voltage*1e-6

#     current_hist: List[float] = [current]

#     # sweep down
#     while current < current_target:
#         # compute next voltages
#         v1 = max(v1 - step, v_min)
#         v2 = max(v2 - step, v_min)

#         # apply and wait
#         sim.set_voltage(slot1, v1)
#         sim.set_voltage(slot2, v2)
#         time.sleep(delay)

#         # read back and record
#         v1 = sim.get_voltage(slot1)
#         v2 = sim.get_voltage(slot2)
#         voltage = agilent.volt()
#         current = voltage*1e-6

#         v1_hist.append(v1)
#         v2_hist.append(v2)
#         current_hist.append(current)

#         # check if we’ve exhausted the range
#         if v1 <= v_min and v2 <= v_min and current < current_target:
#             raise RuntimeError(
#                 f"Target current {current_target:.3e} A not reached; "
#                 f"both V1 and V2 have hit v_min={v_min:.3f} V "
#                 f"(final current={current:.3e} A)."
#             )

#     return v1_hist, v2_hist, current_hist

# v1_hist, v2_hist, i_hist = sweep_down_until_current(sim,agilent,4,5,0.01,delay=0.2,current_target=5e-9,v_min=0.0,
# v_max=2.0)

# print("V1 history:", v1_hist)
# print("V2 history:", v2_hist)
# print("Current history:", i_hist)

############ Unused lines of code ##########
# def sweep_map(sim: Any, agilent: Any,
#               slot_exit: int,
#               slot_ent: int,
#               Vexit_avg: float,
#               Vent_avg: float,
#               Vexit_span: float,
#               Vent_span: float,
#               step_exit: float = 0.1,
#               step_ent: float = 0.1,
#               delay: float = 0.5,
#               output_csv: str = 'map_data.csv',
#               output_fig: str = 'map_colormap.png') -> None:
#     """
#     For each Vent in [Vent_avg, Vent_avg + Vent_span] (step_ent),
#     sweep Vexit in [Vexit_avg, Vexit_avg + Vexit_span] (step_exit),
#     record the measured current, save a CSV of (Vent, Vexit, current),
#     and produce a colormap image.

#     Args:
#         sim:            Instrument controller (with .set_voltage and .get_voltage).
#         agilent:        DMM instance (agilent.volt() → measured voltage).
#         slot_exit:      Channel index for Vexit.
#         slot_ent:       Channel index for Vent.
#         Vexit_avg:      Base Vexit.
#         Vent_avg:       Base Vent.
#         Vexit_span:     ΔVexit to sweep.
#         Vent_span:      ΔVent to sweep.
#         step_exit:      Vexit increment per point.
#         step_ent:       Vent increment per line.
#         delay:          Settle time after each set_voltage.
#         output_csv:     Path for CSV output.
#         output_fig:     Path for colormap PNG.
#     """
#     # 1) build axis arrays
#     Vexit_vals = np.arange(Vexit_avg,
#                            Vexit_avg + Vexit_span + 1e-9,
#                            step_exit)
#     Vent_vals = np.arange(Vent_avg,
#                           Vent_avg + Vent_span + 1e-9,
#                           step_ent)

#     # 2) prepare data structures
#     records = []
#     current_map = np.zeros((len(Vent_vals), len(Vexit_vals)))

#     # 3) nested sweep
#     for i, Vent in enumerate(Vent_vals):
#         sim.set_voltage(slot_ent, Vent)
#         time.sleep(delay)
#         for j, Vexit in enumerate(Vexit_vals):
#             sim.set_voltage(slot_exit, Vexit)
#             time.sleep(delay)

#             # measure current (convert voltage reading to amps if needed)
#             voltage = agilent.volt()
#             current = -1 * voltage * 1e-7

#             current_map[i, j] = current
#             records.append({'Vent': Vent,
#                             'Vexit': Vexit,
#                             'current': current})

#     # 4) save CSV
#     df = pd.DataFrame.from_records(records)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
#     filename_1 = f'{output_csv}_{timestamp}.csv'
#     directory = os.path.join(base_dir, filename_1)
#     df.to_csv(filename_1, index=False,float_format='%.3e')

#     # 5) plot and save colormap

#     qpc_voltage = sim.get_voltage(3)
#     legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

#     plt.figure()
#     X, Y = np.meshgrid(Vexit_vals, Vent_vals)
#     plt.pcolormesh(X, Y, current_map, shading='auto')
#     plt.xlabel('Vexit (V)')
#     plt.ylabel('Vent (V)')
#     plt.title('Current Map')
#     cbar = plt.colorbar()
#     cbar.set_label('Current (A)')

#     # 4) Save the plot next to the CSV
#     filename_2 = f'{output_fig}_{timestamp}.png'
#     plot_path = os.path.join(base_dir, filename_2)
#     plt.savefig(plot_path)
#     plt.close()

# sweep_map(sim,agilent,4,5,Vexit_avg,Vent_avg,0.100,0.100,0.001,0.001,0.1,output_csv='qpc_map.csv',output_fig='qpc_map.png')



############# Unused lines of code ############
# def ramp_two_channels(sim: Any,
#                       agilent: Any,
#                       slot_exit: int,
#                       slot_ent: int,
#                       exit_target: float,
#                       ent_target: float,
#                       step_exit: float,
#                       step_ent: float,
#                       delay: float = 0.5
#                      ) -> Tuple[List[float], List[float], List[float]]:
#     """
#     Simultaneously ramp two gates from their current voltages up to the
#     specified targets, recording voltages and current.

#     Args:
#         sim:         Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
#         agilent:     DMM instance where agilent.volt() returns the measured voltage (V).
#         slot_exit:   Channel index for the “exit” gate.
#         slot_ent:    Channel index for the “ent” gate.
#         exit_target: Final voltage (V) for slot_exit.
#         ent_target:  Final voltage (V) for slot_ent.
#         step_exit:   Voltage increment per iteration for slot_exit.
#         step_ent:    Voltage increment per iteration for slot_ent.
#         delay:       Seconds to wait after setting voltages.

#     Returns:
#         (exit_vals, ent_vals, current_vals):
#            - exit_vals:  list of voltages read from slot_exit at each step
#            - ent_vals:   list of voltages read from slot_ent at each step
#            - current_vals: list of currents (in A) measured at each step
#     """
#     exit_vals: List[float] = []
#     ent_vals: List[float] = []
#     current_vals: List[float] = []

#     # read starting voltages
#     v_exit = sim.get_voltage(slot_exit)
#     v_ent  = sim.get_voltage(slot_ent)

#     # record initial point
#     exit_vals.append(v_exit)
#     ent_vals.append(v_ent)
#     voltage = agilent.volt()
#     current = -1 * voltage * 1e-7
#     current_vals.append(current)

#     # ramp until both targets are reached
#     while v_exit < exit_target or v_ent < ent_target:
#         if v_exit < exit_target:
#             v_exit = min(v_exit + step_exit, exit_target)
#             sim.set_voltage(slot_exit, v_exit)
#         if v_ent < ent_target:
#             v_ent = min(v_ent + step_ent, ent_target)
#             sim.set_voltage(slot_ent, v_ent)

#         time.sleep(delay)

#         # read back actual voltages & current
#         v_exit = sim.get_voltage(slot_exit)
#         v_ent  = sim.get_voltage(slot_ent)
#         voltage = agilent.volt()
#         current = -1 * voltage * 1e-7
#         exit_vals.append(v_exit)
#         ent_vals.append(v_ent)
#         current_vals.append(current)

#     df = pd.DataFrame({'V_exit (V)': exit_vals, 'V_ent (V)': ent_vals, 'Current (A)': current_vals})
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
#     filename_1 = f'sweep_Vent_Vexit_{timestamp}.csv'
#     directory = os.path.join(base_dir, filename_1)
#     df.to_csv(directory, index=False,float_format='%.3e')

#     # 3) Scatter plot of the first two columns
#     x = df.iloc[:, 0]
#     y = df.iloc[:, 1]
#     z = df.iloc[:, 2]
#     x_label = df.columns[0]
#     y_label = df.columns[1]
#     z_label = df.columns[2]

#     # Read QPC gate voltage from channel 3
#     qpc_voltage = sim.get_voltage(3)
#     legend_label = f"QPC gate voltage = {qpc_voltage:.3f} V"

#     plt.figure()
#     plt.scatter(x, z, label=legend_label)
#     plt.xlabel(x_label)
#     plt.ylabel(z_label)
#     plt.title(f"{z_label} vs {x_label}")
#     plt.legend(loc='best')

#     # 4) Save the plot next to the CSV
#     filename_2 = f'sweep_Vent_Vexit_VexitvsI_{timestamp}_scatter.png'
#     plot_path = os.path.join(base_dir, filename_2)
#     plt.savefig(plot_path)
#     plt.close()

#     plt.figure()
#     plt.scatter(y, z, label=legend_label)
#     plt.xlabel(y_label)
#     plt.ylabel(z_label)
#     plt.title(f"{z_label} vs {y_label}")
#     plt.legend(loc='best')

#     # 4) Save the plot next to the CSV
#     filename_3 = f'sweep_Vent_Vexit_VentvsI_{timestamp}_scatter.png'
#     plot_path_1 = os.path.join(base_dir, filename_3)
#     plt.savefig(plot_path_1)
#     plt.close()

# ramp_two_channels(
#     sim=sim,
#     agilent=agilent,
#     slot_exit=4,
#     slot_ent=5,
#     exit_target=0.819500,
#     ent_target=0.791500,
#     step_exit=0.001,
#     step_ent=0.001,
#     delay=0.2
# )

# Sweep Vexit from 0 to (vexit_avg - 0.100) with 0.001 V steps
        # vexit_start_RF = 0.0
        # vexit_stop_RF = max(vexit_avg_RF - 0.100, 0)
        # vexit_step_RF = 0.001
        # vexit_threshold_current_RF = 1e-10  # stopping threshold

        # vexit_values_RF = []
        # currents_RF = []

        # v = vexit_start_RF
        # while v <= vexit_stop_RF:
        #     set_voltage_sim(sim, slot='ch5', target=v, step=0.001, delay=0.1)

        #     voltage_agilent_RF = abs(agilent.volt())
        #     current_RF = voltage_agilent_RF * agilent_sensitivity_RF

        #     vexit_values_RF.append(v)
        #     currents_RF.append(current_RF)

        #     if current_RF >= vexit_threshold_current_RF:
        #         print(f'Stopping sweep: current {current_RF:.2e} A exceeded threshold at Vexit = {v:.3f} V')
        #         break

        #     v += vexit_step_RF


# threshold_current = 150e-12  # 150 pA
# vexit_step       = 0.001     # 1 mV step for Vexit
# vent_step        = 0.01      # 10 mV step for Vent

# for v2_target, v3_target in qpc_targets_RF:

#     # 1) set QPC gates
#     simultaneous_set_voltage_sim(sim, 'ch2', v2_target, 'ch3', v3_target,
#                                 step=0.01, delay=0.5)

#     # retrieve Vent_avg and Vexit_avg
#     row = df_results[np.isclose(df_results['V2_target (V)'], v2_target, atol=1e-6)]
#     if row.empty:
#         print(f'No results for V2_target={v2_target}')
#         continue

#     vent_avg_RF  = row.iloc[0]['Vent_avg (V)']
#     vexit_avg_RF = row.iloc[0]['Vexit_avg (V)']

#     for power in user_power_dBm:
#         # 2) set RF power
#         rf.set_power(power)
        
#         # lookup vp for this power
#         vp_row = power_to_voltage[power_to_voltage['Power (dBm)'] == power]
#         if vp_row.empty:
#             print(f'Power {power} dBm not in lookup, skipping.')
#             continue
#         vp = vp_row['Voltage (Vp)'].values[0]

#         # define Vent sweep range
#         vent_min = V_ent_max - vp
#         vent_max = V_ent_max

#         for vent in np.arange(vent_min, vent_max + 1e-12, vent_step):
#             # 4) set Vent
#             set_voltage_sim(sim, slot='ch4', target=vent,
#                             step=0.001, delay=0.5)

#             # 5) sweep Vexit
#             vexit_vals = []
#             currents   = []
#             for vexit in np.arange(0.0, vexit_avg_RF + vexit_step/2, vexit_step):
#                 set_voltage_sim(sim, slot='ch5', target=vexit,
#                                 step=vexit_step, delay=0.1)
#                 voltage = abs(agilent.volt())
#                 I = voltage * agilent_sensitivity_RF
#                 vexit_vals.append(vexit)
#                 currents.append(I)
#                 if I >= threshold_current:
#                     break

#             # 6) save this Vent sweep
#             df_sweep = pd.DataFrame({
#                 'Vent (V)':    [vent] * len(vexit_vals),
#                 'Vexit (V)':   vexit_vals,
#                 'Current (A)': currents
#             })

#             fname = (f'QPC_V2_{v2_target:.3f}_'
#                      f'P_{power}dBm_'
#                      f'Vent_{vent:.3f}V_'
#                      f'{timestamp}.csv')
#             outpath = os.path.join(base_dir, fname)
#             df_sweep.to_csv(outpath, index=False)

#             print(f'Saved sweep: V2={v2_target:.3f} V, P={power} dBm, Vent={vent:.3f} V -> {fname}')

            
