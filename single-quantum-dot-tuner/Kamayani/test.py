import qcodes
from qcodes import instrument_drivers
from qcodes.dataset import do0d, load_or_create_experiment
from qcodes.instrument import Instrument
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
from qcodes.validators import Numbers
from qcodes import Parameter
from pprint import pprint
import numpy as np
import time

# sr = SR830("lockin", "GPIB0::8::INSTR")
# # load_or_create_experiment(experiment_name="SR830_notebook")

# # pprint(list(sr.SNAP_PARAMETERS.keys()))

# # print(sr.complex_voltage())

# # print(sr.snap("ch1", "ch2"))

# # print(sr.snap("ch1", "freq", "aux1", "aux2","aux3","aux4"))

# # print(sr.print_readable_snapshot())

# # sr.amplitude.set(0.250)

# # sr.frequency.set(17.371)

# # latest = sr.amplitude.get_latest()

# # print(latest)

# # def ground_device_sr(sr, target: float = 0.004, step: float = 0.001, delay: float = 0.5):
# #     min_amp, max_amp = 0.004, 5.0
# #     if target < min_amp:
# #         print(f"Clamping target amplitude to minimum {min_amp} v.")
# #         target = min_amp
# #     elif target > max_amp:
# #         print(f"Clamping target amplitude to minimum {max_amp} v.")
# #         target = max_amp
    
# #     current = sr.amplitude.get()

# #     if abs(current - target) < 1e-12:
# #         return
    
# #     direction = np.sign(current - target)
# #     setpoints = np.arange(current, target, direction * step)

# #     for amp in setpoints:
# #         sr.amplitude.set(float(amp))
# #         time.sleep(delay)

# #     sr.amplitude.set(target)

# # ground_device_sr(sr, target = 0.004, step = 0.001, delay = 0.5)

# # latest = sr.amplitude.get_latest()

# # print(latest)

# sim = SIM928("sim900", "GPIB0::3::INSTR")

# sr.reference_source.set('internal')
# print(sr.R.get())

# def set_voltage_sim(sim, slot: int, target: float, step: float = 0.1, delay: float = 0.5):
#     current = sim.get_voltage(slot)
#     if abs(current - target) < 1e-12:
#         return
    
#     direction = -np.sign(current - target)
#     setpoints = np.arange(current, target, direction * step)

#     for V in setpoints:
#         sim.set_voltage(slot, float(V))
#         time.sleep(delay)

#     sim.set_voltage(slot, target)
#     time.sleep(delay)
#     voltage = sim.get_voltage(slot)

#     return voltage

# voltage = set_voltage_sim(sim, 1, 3, step = 0.1, delay = 0.5)
# print(voltage)






########### code for auto_tuning #####################

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
from typing import List, Tuple
from pprint import pprint
import numpy as np

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

for slot in range(1,9):
    ground_device_sim(sim, slot, step= 0.1, delay = 0.5)

print('The voltage of all channels in sim900 is set to 0V.')

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

print('Step 1: Grounding the devices: completed.')

# Step 2: Setting up the voltage of the top gate (channel 1 in sim900)

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

simultaneous_set_voltage_sim(sim, 2, 0.3, 3, 0.3, step = 0.01, delay = 0.5)

print('Step 3: Setting up the voltage of the QPC gates (channels 2 and 3 of sim900): completed.')

# Step 4: Setting up the voltage of channel 7 of sim900 (which replaces setting the voltage of SR830)

#set_voltage_sr(sr, target = 1.0, step = 0.01, delay = 0.5)
set_voltage_sim(sim, slot = 6, target = 0.001, step = 0.0001, delay = 0.5)

print('Step 4: Setting up the voltage of channel 6 of sim900: completed.')

# Step 5: Sweeping the entrance and the exit gates to find the region of operation. 

# # Step (5i): Bringing the exit and the entrance gates symmetrically until current = 5nA. 

# def set_voltage_until_current(sim,sr,slot1: int, slot2: int, step: float, delay: float, current_target: float):

#     V1 = sim.get_voltage(slot1)
#     V2 = sim.get_voltage(slot2)
#     voltage = agilent.volt()
#     current = voltage*1e-6

#     while current < current_target:
#         V1 += step
#         V2 += step 
#         V1 = sim.get_voltage(slot1)
#         V2 = sim.get_voltage(slot2)
#         time.sleep(delay)
#         voltage = agilent.volt()
#         current = voltage*1e-6
    
#     V1_a = sim.get_voltage(slot1)
#     V2_b = sim.get_voltage(slot2)

#     first_point = [V1_a, V2_b]
#     return first_point

# first_point = set_voltage_until_current(sim,sr,4, 5, 0.01, 0.5, 5.0)
# print(f'first point: {first_point}')

# # Step (5ii): Stay at the exit gate value found in Step (5i) - (first_point[0]) and bring the entrance gate down until current = 0.

# set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.1, delay = 0.5)

# def set_voltage_one_gate_constant(sim,sr,slot1: int, slot2: int, step: float, delay: float, current_target: float):
#     sr.reference_source.set('internal')

#     V2 = sim.get_voltage(slot2)
#     current = sr.R.get()

#     while current < current_target:
#         V2 += step 
#         V2 = sim.get_voltage(slot2)
#         time.sleep(delay)
#         current = sr.R.get()
    
#     V1_a = sim.get_voltage(slot1)
#     V2_c = sim.get_voltage(slot2)

#     second_point = [V1_a, V2_c]
#     return second_point

# second_point = set_voltage_one_gate_constant(sim,sr,4, 5, 0.01, 0.5, 0.0)
# print(f'second point: {second_point}')

# # Step (5iii): Go back at the values of exit and entrance gates found in Step (5i) (first_point).

# set_voltage_sim(sim, slot = 4, target = first_point[0], step = 0.1, delay = 0.5)
# set_voltage_sim(sim, slot = 5, target = first_point[1], step = 0.1, delay = 0.5)

# # Step (5iv): Stay at the entrance gate value set in Step (5iii) - (first_point[1]) and bring the exit gate down until current = 0.

# set_voltage_sim(sim, slot = 5, target = first_point[1], step = 0.1, delay = 0.5)

# third_point = set_voltage_one_gate_constant(sim,sr,5, 4, 0.01, 0.5, 0.0)
# print(f'third point: {third_point}')













# def set_voltage_until_current(sim,
#                               sr,
#                               slot1: int,
#                               slot2: int,
#                               step: float,
#                               delay: float,
#                               current_target: float
#                               ) -> Tuple[List[float], List[float], List[float]]:
#     """
#     Ramp the voltages on sim slots `slot1` and `slot2` in increments of `step`
#     until the lock‑in amplifier’s R‑reading reaches or exceeds `current_target`.
#     At each step, record V1, V2, and the current into lists.

#     Returns:
#         V1_list, V2_list, current_list
#     """
#     # ensure we’re reading the SR830’s internal reference
#     sr.reference_source.set('internal')

#     # initialize lists to store the ramp history
#     V1_list: List[float] = []
#     V2_list: List[float] = []
#     current_list: List[float] = []

#     # initial readings
#     V1 = sim.get_voltage(slot1)
#     V2 = sim.get_voltage(slot2)
#     current = sr.R.get()

#     V1_list.append(V1)
#     V2_list.append(V2)
#     current_list.append(current)

#     # ramp until target current reached
#     while current < current_target:
#         # increment desired voltages
#         V1 += step
#         V2 += step

#         # apply them to the instrument
#         sim.set_voltage(slot1, V1)
#         sim.set_voltage(slot2, V2)

#         # read back actual voltages
#         V1 = sim.get_voltage(slot1)
#         V2 = sim.get_voltage(slot2)

#         # wait and then read current
#         time.sleep(delay)
#         current = sr.R.get()

#         # store this step’s readings
#         V1_list.append(V1)
#         V2_list.append(V2)
#         current_list.append(current)

#     return V1_list, V2_list, current_list


# # usage example:
# V1_vals, V2_vals, I_vals = set_voltage_until_current(sim, sr,
#                                                      slot1=4,
#                                                      slot2=5,
#                                                      step=0.01,
#                                                      delay=0.5,
#                                                      current_target=5.0)

# print('V1 ramp history:', V1_vals)
# print('V2 ramp history:', V2_vals)
# print('current history:', I_vals)



# def ramp_voltages(sim,
#                   slot1: int,
#                   slot2: int,
#                   step: float,
#                   target_v1: float,
#                   target_v2: float,
#                   delay: float = 0.0
#                  ) -> Tuple[List[float], List[float]]:
#     """
#     Simultaneously ramp voltages on sim slots `slot1` and `slot2` up to
#     `target_v1` and `target_v2` in increments of `step`, recording each reading.

#     Args:
#         sim:           Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
#         slot1:         Channel index for V1.
#         slot2:         Channel index for V2.
#         step:          Voltage increment (V) per iteration.
#         target_v1:     Final voltage (V) to reach on slot1.
#         target_v2:     Final voltage (V) to reach on slot2.
#         delay:         Seconds to wait after setting voltages before reading back (default 0.0).

#     Returns:
#         v1_history:   List of all V1 readings (including initial and final).
#         v2_history:   List of all V2 readings.
#     """
#     v1_history: List[float] = []
#     v2_history: List[float] = []

#     # Read initial voltages
#     v1 = sim.get_voltage(slot1)
#     v2 = sim.get_voltage(slot2)
#     v1_history.append(v1)
#     v2_history.append(v2)

#     # Ramp until both channels have reached their targets
#     while v1 < target_v1 or v2 < target_v2:
#         if v1 < target_v1:
#             v1 = min(v1 + step, target_v1)
#             sim.set_voltage(slot1, v1)
#         if v2 < target_v2:
#             v2 = min(v2 + step, target_v2)
#             sim.set_voltage(slot2, v2)

#         if delay > 0:
#             time.sleep(delay)

#         # Read back actual voltages
#         v1 = sim.get_voltage(slot1)
#         v2 = sim.get_voltage(slot2)
#         v1_history.append(v1)
#         v2_history.append(v2)

#     return v1_history, v2_history




# # Ramp both channels from their current voltages up to 2.0 V and 3.5 V
# # in 0.1 V steps, pausing 0.2 s between each step.
# v1_vals, v2_vals = ramp_voltages(sim,
#                                  slot1=4,
#                                  slot2=5,
#                                  step=0.1,
#                                  target_v1=5.0,
#                                  target_v2=3.0,
#                                  delay=0.2)

# print('V1 history:', v1_vals)
# print('V2 history:', v2_vals)

# import pandas as pd

# # assume v1_vals and v2_vals are your two lists of equal length
# df = pd.DataFrame({
#     'v1_vals': v1_vals,
#     'v2_vals': v2_vals
# })

# print(df)

# df.plot(kind = 'scatter',x = 'v1_vals', y = 'v2_vals' )

# directory = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\voltage_history.csv'
# # write to CSV (no index column)
# df.to_csv(directory, index=False)







import time
from typing import List, Tuple

def sweep_gate_to_zero(sim,
                       sr,
                       constant_slot: int,
                       sweep_slot: int,
                       step: float,
                       delay: float
                      ) -> Tuple[List[float], List[float]]:
    """
    Sweep the voltage on `sweep_slot` from its current value down to 0 V in decrements of `step`,
    while holding the voltage on `constant_slot` constant. Record every sweep voltage and
    the corresponding lock‑in current into two lists.

    Args:
        sim:             Instrument controller with .get_voltage(slot) and .set_voltage(slot, V).
        sr:              QCoDeS SR830 instance (must support sr.R.get()).
        constant_slot:   The channel index whose voltage remains unchanged.
        sweep_slot:      The channel index to sweep down to zero.
        step:            The decrement (V) per iteration.
        delay:           Seconds to wait after setting voltage before reading back.

    Returns:
        vs_history:      List of all voltages read on `sweep_slot` (from start down to 0 V).
        current_history: List of all corresponding SR830 R‑readings (in same order).
    """
    # ensure the lock‑in is using its internal reference
    sr.reference_source.set('internal')

    # storage for histories
    vs_history: List[float] = []
    current_history: List[float] = []

    # initial readings
    vs = sim.get_voltage(sweep_slot)
    current = sr.R.get()
    vs_history.append(vs)
    current_history.append(current)

    # ramp down until we hit 0 V
    while vs > 0:
        # compute next voltage, clamp at zero
        vs = max(vs - step, 0.0)
        sim.set_voltage(sweep_slot, vs)

        # optional delay
        if delay > 0:
            time.sleep(delay)

        # read back actual sweep voltage and current
        vs = sim.get_voltage(sweep_slot)
        current = sr.R.get()

        # record
        vs_history.append(vs)
        current_history.append(current)

    return vs_history, current_history


# Example usage:
vs_vals, currents = sweep_gate_to_zero(
    sim=sim,
    sr=sr,
    constant_slot=4,   # e.g. keep slot 4 constant
    sweep_slot=5,      # sweep slot 5 from its current value down to 0 V
    step=0.01,         # 10 mV steps
    delay=0.1          # 100 ms settle time
)

print('Sweep voltages:', vs_vals)
print('Currents:', currents)

# vent_vexit_avg_data = pd.DataFrame({
#     'V_qpc (V)': qpc_vs,
#     'Peak Frequency (Hz)': peak_fs,
#     'Peak Period (s)': peak_periods
# })

# vent_vexit_avg_dataname = f'vent_vexit_avg_dat_{timestamp}.csv'
# directory_vent_vexit_avg_data = os.path.join(base_dir, vent_vexit_avg_dataname)
# vent_vexit_avg_data.to_csv(directory_avg_voltage, index=False)

# # Find the row with the maximum period
# row_max_period = vent_vexit_avg_data.loc[vent_vexit_avg_data['Peak Period (s)'].idxmax()]

# # Access the V_qpc and max period
# v_qpc_at_max_period = row_max_period['V_qpc (V)']
# max_period_value = row_max_period['Peak Period (s)']

# print(f'Maximum period: {max_period_value:.6f} s at V_qpc = {v_qpc_at_max_period:.3f} V')

# tolerance = 1e-6  # adjust based on expected precision
# match_row = df_results[np.isclose(df_results['V2_target (V)'], v_qpc_at_max_period, atol=tolerance)]

# if not match_row.empty:
#     vent_avg_at_max_qpc = match_row.iloc[0]['Vent_avg (V)']
#     vexit_avg_at_max_qpc = match_row.iloc[0]['Vexit_avg (V)']
#     print(f'Max period occurs at V_qpc = {v_qpc_at_max_period:.3f} V')
#     print(f'Corresponding Vent_avg = {vent_avg_at_max_qpc:.6f} V')
#     print(f'Corresponding Vexit_avg = {vexit_avg_at_max_qpc:.6f} V')
# else:
#     print('No matching V2_target found within tolerance.')

# limiting_vent_RF = vent_avg_at_max_qpc - 0.100
# limiting_vexit_RF = vexit_avg_at_max_qpc - 0.100







