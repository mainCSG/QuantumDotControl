import os
import time
from datetime import datetime
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
import qcodes as qc

########## USER-CONFIGURABLE PARAMETERS ##########
base_dir              = r'C:\Users\coher\Desktop\Kamayani\Data'
agilent_sensitivity   = 1e-7        # DC DMM sensitivity (A/V)
agilent_sensitivity_RF= 1e-9        # RF DMM sensitivity (A/V)
I_min_DC              = 30e-13     # NEW: DC threshold (A)
I_min_RF              = 5e-12      # NEW: RF lower threshold for gate bound (A)
I_max_RF              = 150e-12    # NEW: RF plateau current target (A)
window_DC             = 5          # NEW: points for DC averaging
window_plateau        = 10         # NEW: points for derivative-flat plateau
step_exit_RF          = 0.001      # NEW: RF exit sweep step (V)
delay_RF              = 0.1        # NEW: RF exit sweep delay (s)
user_power_dBm        = [1,2,3]    # RF powers to loop over
qpc_gate_voltages_RF  = [0.0, 0.1]

# timestamp for this run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

########## UTILITY FUNCTIONS ##########

def reinitialize_instrument(name: str, driver, address: str, **kwargs):
    # CHANGED: close any existing QCoDeS instrument handle first
    try:
        inst = qc.Instrument.find_instrument(name)
        inst.close()
    except Exception:
        pass
    instrument = driver(name, address, **kwargs)
    return instrument

def ground_all_channels(sim: Any, slots: List[str], step: float=0.01, delay: float=0.5):
    # NEW: ground a list of channels smoothly
    for slot in slots:
        V0 = sim.get_voltage(slot)
        if abs(V0) < 1e-12:
            continue
        direction = -np.sign(V0)
        for v in np.arange(V0, 0, direction*step):
            sim.set_voltage(slot, v)
            time.sleep(delay)
        sim.set_voltage(slot, 0.0)
        time.sleep(delay)

def sweep_until_I_or_V(
    sim: Any,
    agilent: Any,
    slot: str,
    I_thresh: float,
    V_thresh: float,
    step: float,
    delay: float
) -> Tuple[np.ndarray, np.ndarray]:
    # NEW: sweep one gate until current ≥ I_thresh or voltage ≥ V_thresh
    vs, Is = [], []
    v = sim.get_voltage(slot)
    while v <= V_thresh:
        sim.set_voltage(slot, v)
        time.sleep(delay)
        I = abs(agilent.volt()) * agilent_sensitivity_RF
        vs.append(v); Is.append(I)
        if I >= I_thresh:
            break
        v += step
    return np.array(vs), np.array(Is)

def find_flat_plateau(
    Vs: np.ndarray,
    Is: np.ndarray,
    tol: float=1e-12,
    window: int=10
) -> Optional[Tuple[float,float]]:
    # NEW: detect first region of `window` consecutive derivative < tol
    dI = np.abs(np.diff(Is)/np.diff(Vs))
    for i in range(len(dI) - window + 1):
        if np.all(dI[i:i+window] < tol):
            return Vs[i], Vs[i+window]
    return None

def fit_decay_cascade(df_plateau: pd.DataFrame, v_start: float, v_end: float) -> float:
    """
    Stub for your decay-cascade model. Replace with your actual fitting code.
    """
    # NEW: placeholder
    return np.random.random()

def find_gate_bound(
    csv_path: str,
    vent_col: str,
    curr_col: str,
    I_min: float,
    window: int = 5
) -> Optional[float]:
    # NEW: post-hoc search in saved CSV for first run of window points ≤ I_min
    df = pd.read_csv(csv_path)
    ent = df[vent_col].to_numpy()
    I   = df[curr_col].to_numpy()
    for i in range(len(I) - window + 1):
        if np.all(I[i:i+window] <= I_min):
            return float(np.mean(ent[i:i+window]))
    return None

########## CORE RAMP + DC POST-HOC ##########

def ramp_two_channels(
    sim: Any,
    agilent: Any,
    slot_exit: str,
    slot_ent: str,
    exit_target: float,
    ent_target: float,
    step_exit: float,
    step_ent: float,
    delay: float = 0.5,
    I_min_DC: float = 0.0,
    window_DC: int = 5
) -> Tuple[str, Optional[float]]:
    """
    Ramp two gates together, record current, then post-hoc find V_ent_max_DC.
    Returns (sweep_csv_path, V_ent_max_DC).
    """
    # unchanged ramp logic
    qpc_voltage = sim.get_voltage('ch3')
    exit_vals, ent_vals, current_vals = [], [], []

    v_exit = sim.get_voltage(slot_exit)
    v_ent  = sim.get_voltage(slot_ent)
    exit_vals.append(v_exit)
    ent_vals .append(v_ent)
    current_vals.append(abs(agilent.volt()) * agilent_sensitivity)

    max_iters=10000; iteration=0
    while v_exit < exit_target and v_ent < ent_target and iteration < max_iters:
        iteration += 1
        if v_exit < exit_target:
            v_exit = min(v_exit + step_exit, exit_target)
            sim.set_voltage(slot_exit, v_exit)
        if v_ent < ent_target:
            v_ent = min(v_ent + step_ent, ent_target)
            sim.set_voltage(slot_ent, v_ent)
        time.sleep(delay)
        exit_vals.append(v_exit)
        ent_vals .append(v_ent)
        current_vals.append(abs(agilent.volt()) * agilent_sensitivity)
    if iteration>=max_iters:
        raise RuntimeError("Ramp did not complete within the maximum iteration count")

    # save sweep CSV
    df = pd.DataFrame({
        'V_exit (V)': exit_vals,
        'V_ent (V)' : ent_vals,
        'Current (A)': current_vals
    })
    sweep_csv = os.path.join(base_dir, f'{qpc_voltage:.3f}_sweep_Vent_Vexit_{timestamp}.csv')
    df.to_csv(sweep_csv, index=False, float_format='%.3e')

    # NEW: post-hoc DC V_ent_max detection (first 5 pts with I <= I_min_DC)
    V_ent_max_DC = None
    for i in range(len(current_vals) - window_DC + 1):
        if all(c <= I_min_DC for c in current_vals[i:i+window_DC]):
            V_ent_max_DC = float(np.mean(ent_vals[i:i+window_DC]))
            pd.DataFrame([{
                'V_qpc (V)': qpc_voltage,
                'V_ent_max_DC (V)': V_ent_max_DC,
                'I_min_DC (A)': I_min_DC
            }]).to_csv(
                os.path.join(base_dir, f'{qpc_voltage:.3f}_V_ent_max_summary_{timestamp}.csv'),
                index=False, float_format='%.3e'
            )
            break

    # unchanged diagnostic plots
    for xs, ys, lbl, fname in [
        (df['V_exit (V)'], df['Current (A)'],
         f"V_qpc={qpc_voltage:.3f}, ent=[{ent_vals[0]:.3f},{ent_target:.3f}]",
         f'{qpc_voltage:.3f}_Vexit_Coulomb_{timestamp}.png'),
        (df['V_ent (V)'], df['Current (A)'],
         f"V_qpc={qpc_voltage:.3f}, exit=[{exit_vals[0]:.3f},{exit_target:.3f}]",
         f'{qpc_voltage:.3f}_Vent_Coulomb_{timestamp}.png'),
    ]:
        plt.figure(); plt.plot(xs, ys, label=lbl)
        plt.xlabel(xs.name); plt.ylabel(ys.name); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(base_dir, fname)); plt.close()

    return sweep_csv, V_ent_max_DC

########## MAIN AUTO-TUNING LOOP ##########

# 1) Connect instruments
sim     = reinitialize_instrument('sim900', SIM928,       'GPIB0::3::INSTR', slot_names={i:f'ch{i}' for i in range(1,9)})
agilent = reinitialize_instrument('agilent', Agilent34401A,'GPIB0::21::INSTR')
# rf class placeholder – replace with your actual RF driver
# rf = reinitialize_instrument('rf_gen', AgilentE4432B, 'GPIB0::19::INSTR')

# 2) QPC voltages list for RF stage
qpc_targets_RF = [(v,v) for v in qpc_gate_voltages_RF]

all_results = []

for v2_target, v3_target in qpc_targets_RF:
    # NEW: ground all channels at start of each power loop
    ground_all_channels(sim, [f'ch{i}' for i in range(1,9)])

    # set QPC gates
    sim.set_voltage('ch2', v2_target); sim.set_voltage('ch3', v3_target)
    time.sleep(0.5)

    # DC diagonal sweep → detect V_ent_max_DC
    sweep_csv, V_ent_max_DC = ramp_two_channels(
        sim, agilent,
        'ch4', 'ch5',
        exit_target=v2_target+0.1,
        ent_target =v3_target+0.1,
        step_exit  =0.001,
        step_ent   =0.001,
        delay      =0.2,
        I_min_DC   =I_min_DC,
        window_DC  =window_DC
    )

    # Loop over RF powers (RF stage)
    for power in user_power_dBm:
        # … continue with RF exit sweep, plateau detection, decay-cascade fit, etc.
        pass
