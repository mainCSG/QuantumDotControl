import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# # # # data = pd.read_csv('C:\\Users\\coher\\Desktop\\Kamayani\\SEP14\\C1\\2025-07-11_13-49-51.csv')
# # # # threshold = 3e-9
# # # # data = data[data['I_ac (A) (V)'] < threshold]

# # # def sigmoid(x, a, b, x0, y0):
# # #     return a/(1+np.exp(b * (x-x0))) + y0

# # # # Current_values = data['I_ac (A) (V)'].values
# # # # V_ent_values = data['V_ent (V)'].values

# # # # plot1 = data.plot(kind = 'line', x = 'V_ent (V)', y = 'I_ac (A) (V)')

# # # # popt, pcov = curve_fit(sigmoid, V_ent_values, Current_values, p0 = [0.1, 10,0.5,10], maxfev = 10000)

# # # # a_opt, b_opt, c_opt, d_opt = popt

# # # # x_model = np.linspace(min(V_ent_values), max(V_ent_values), 100)
# # # # y_model = sigmoid(x_model, a_opt, b_opt, c_opt, d_opt)

# # # # plt.scatter(V_ent_values, Current_values)
# # # # plt.plot(x_model, y_model, color='r' )
# # # # plt.show()

# # # # popt

# # # # def fit(x,a,k):
# # # #     return a*np.exp(k*x)

# # # # plot1 = data.plot(kind = 'line', x = 'V_ent (V)', y = 'I_ac (A) (V)')

# # # # popt, pcov = curve_fit(fit, V_ent_values, Current_values, p0 = [1.6e-29,35], maxfev = 10000)

# # # # a_opt, b_opt = popt

# # # # x_1_model = np.linspace(min(V_ent_values), max(V_ent_values), 100)
# # # # y_1_model = fit(x_model, a_opt, b_opt)

# # # # plt.scatter(V_ent_values, Current_values)
# # # # plt.plot(x_1_model, y_1_model, color='r' )
# # # # plt.show()

# # # # popt

# # # # print(data)

# # # # # import qcodes
# # # # # from qcodes import instrument_drivers
# # # # # from qcodes.dataset import do0d, load_or_create_experiment
# # # # # from qcodes.instrument import Instrument
# # # # # from qcodes.instrument_drivers.stanford_research import SR830
# # # # # from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
# # # # # from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
# # # # # from qcodes.validators import Numbers
# # # # # from qcodes import Parameter
# # # # # import time
# # # # # from pprint import pprint
# # # # # import numpy as np

# # # # # agilent = Agilent34401A("agilent", "GPIB0::21::INSTR")

# # # # # current = agilent.volt()
# # # # # print(current)

# # # # import qcodes
# # # # from qcodes import instrument_drivers
# # # # from qcodes.dataset import do0d, load_or_create_experiment
# # # # from qcodes.instrument import Instrument
# # # # from qcodes.instrument_drivers.stanford_research import SR830
# # # # from qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928 import SIM928
# # # # from qcodes.instrument_drivers.agilent.Agilent_34401A import Agilent34401A
# # # # from qcodes.validators import Numbers
# # # # from qcodes import Parameter
# # # # import time
# # # # from pprint import pprint
# # # # import numpy as np

# # # # sim = SIM928("sim900", "GPIB0::3::INSTR", slot_names={i: f'ch{i}' for i in range (1,9)})
# # # # agilent = Agilent34401A("agilent", "GPIB0::21::INSTR")


# # # # # Step (5i): Bringing the exit and the entrance gates symmetrically until current = 5nA. 

# # # # # array_one = []
# # # # # array_two = []
# # # # # def set_voltage_until_current(sim,sr,slot1: int, slot2: int, step: float, delay: float, current_target: float):
# # # # #     sr.reference_source.set('internal')

# # # # #     V1 = sim.get_voltage(slot1)
# # # # #     V2 = sim.get_voltage(slot2)
# # # # #     current = sr.R.get()

# # # # #     while current < current_target:
# # # # #         V1 += step
# # # # #         V2 += step 
# # # # #         V1 = sim.get_voltage(slot1)
# # # # #         V2 = sim.get_voltage(slot2)
# # # # #         array_one.append(V1)
# # # # #         array_two.append(V2)
# # # # #         time.sleep(delay)
# # # # #         current = sr.R.get()
    
# # # # #     V1_a = sim.get_voltage(slot1)
# # # # #     V2_b = sim.get_voltage(slot2)

# # # # #     first_point = [V1_a, V2_b]
# # # # #     return first_point



# # # # # first_point = set_voltage_until_current(sim,sr,4, 5, 0.01, 0.5, 5.0)
# # # # # print(f'first point: {first_point}')

# # # # # Step 3: Setting up the voltage of both of the QPC gates (channel 2 and channel 3 in sim900)

# # # # array_one = []
# # # # array_two = []
# # # # def simultaneous_set_voltage_sim(sim, slot1: int, target1: float, slot2: int, target2: float, step: float = 0.01, delay: float = 0.5):

# # # #     V1_0 = sim.get_voltage(slot1)
# # # #     V2_0 = sim.get_voltage(slot2)

# # # #     if abs(V1_0 - target1) < 1e-12 and abs(V2_0 - target2) < 1e-12:
# # # #         return
    
# # # #     dir1 = np.sign(target1 - V1_0)
# # # #     dir2 = np.sign(target2 - V2_0)

# # # #     sp1 = np.arange(V1_0, target1, dir1 * step)
# # # #     sp2 = np.arange(V2_0, target2, dir2 * step)

# # # #     if sp1.size == 0 or sp1[-1] != target1:
# # # #         sp1 = np.append(sp1, target1)
# # # #     if sp2.size == 0 or sp2[-1] != target2:
# # # #         sp2 = np.append(sp2, target2)

# # # #     for v1, v2 in zip(sp1,sp2):
# # # #         sim.set_voltage(slot1, float(v1))
# # # #         V1 = sim.get_voltage(slot1)
# # # #         array_one.append(V1)
# # # #         sim.set_voltage(slot2, float(v2))
# # # #         V2 = sim.get_voltage(slot2)
# # # #         array_one.append(V2)
# # # #         time.sleep(delay)

# # # #     sim.set_voltage(slot1, target1)
# # # #     sim.set_voltage(slot2, target2)
# # # #     time.sleep(delay)


# # # #     print(f'The voltage of the channel {slot1} and {slot2} of sim900 is set to {target1: .6f} V and {target2: .6f} V.')

# # # #     return array_one, array_two

# # # # q = simultaneous_set_voltage_sim(sim, 2, 0.3, 3, 0.3, step = 0.01, delay = 0.5)
# # # # print(q)


# # # # print('Step 3: Setting up the voltage of the QPC gates (channels 2 and 3 of sim900): completed.')



# # # # import time
# # # # from typing import Tuple, List

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

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



directory = "C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv"
df = pd.read_csv(directory)


# # 2) Determine original steps and shift between channels
# step_exit = df['V_exit (V)'].diff().mode()[0]
# step_ent  = df['V_ent (V)'].diff().mode()[0]
# # average shift V_exit - V_ent
# shift = (df['V_exit (V)'] - df['V_ent (V)']).mean()

# # 3) Build extended ranges ±0.08 around original min/max
# min_exit, max_exit = df['V_exit (V)'].min(), df['V_exit (V)'].max()
# low_exit  = np.arange(min_exit - 0.08, min_exit, step_exit)
# high_exit = np.arange(max_exit + step_exit, max_exit + 0.08 + step_exit/2, step_exit)

# # corresponding vents by preserving original shift
# low_ent  = low_exit - shift
# high_ent = high_exit - shift

# # 4) Create DataFrames for new points (current = 0)
# df_low  = pd.DataFrame({'V_exit (V)': low_exit, 'V_ent (V)': low_ent,   'Current (A)': 0.0})
# df_high = pd.DataFrame({'V_exit (V)': high_exit,'V_ent (V)': high_ent,  'Current (A)': 0.0})

# # 5) Concatenate and sort by V_exit
# df_ext = pd.concat([df_low, df, df_high], ignore_index=True)
# df_ext.sort_values('V_exit (V)', inplace=True)
# df_ext.reset_index(drop=True, inplace=True)

# df = df_ext


# # # import pandas as pd

# # # # 1) Load your data (adjust path / delimiter as needed)
# # # data = pd.read_csv('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\test_data.csv', sep = '\t')
# # # data = data[data['I_ac (A) (V)'] == 0.0]
# # # data = data.sort_values(by='V_ent (V)', ascending=False)
# # # data = data.head(20)

# # # V_mid =  data['V_ent (V)'].mean()

# # # print(V_mid)
# # # print(f'Average of top 30 voltages (V_mid): {V_mid:.6f} V')

# # import numpy as np
# # import pandas as pd
# # from typing import Tuple

# # def fft_from_df(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
# #     """
# #     Compute the single-sided FFT amplitude spectrum of df[y_col] sampled at df[x_col].

# #     Args:
# #         df:      pandas DataFrame containing the data.
# #         x_col:   Name of the column for the independent variable (must be uniformly spaced).
# #         y_col:   Name of the column for the dependent variable (signal).

# #     Returns:
# #         freqs:   1D array of non-negative frequency bins (cycles per unit of x_col).
# #         amps:    1D array of corresponding single-sided FFT amplitudes.
# #     """
# #     # Extract data as NumPy arrays
# #     x = df[x_col].to_numpy()
# #     y = df[y_col].to_numpy()

# #     # Verify uniform spacing in x
# #     dx = x[1] - x[0]
# #     if not np.allclose(np.diff(x), dx, atol=1e-9):
# #         raise ValueError(f"Values in '{x_col}' must be uniformly spaced")

# #     N = len(y)

# #     # Compute the FFT and normalize
# #     Y = np.fft.fft(y)
# #     Y_mag = np.abs(Y) / N

# #     # Generate the frequency axis
# #     freqs = np.fft.fftfreq(N, d=dx)

# #     # Keep only non-negative frequencies (single-sided spectrum)
# #     mask = freqs >= 0
# #     return freqs[mask], 2 * Y_mag[mask]

# # df = pd.read_csv("C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv")
# # #df = df[df['Current (A)'] > 1e-11]
# # freqs, amps = fft_from_df(df, 'V_ent (V)', 'Current (A)')

# # print(f'freq: {freqs}, amp: {amps}')

# # import matplotlib.pyplot as plt

# # # assume you’ve already done:
# # # freqs, amps = fft_from_df(df, 'V_exit (V)', 'Current (A)')

# # plt.figure()
# # plt.plot(freqs, amps, '-')
# # plt.xlabel('Frequency (1/V)')
# # plt.ylabel('Amplitude (A)')
# # plt.title('FFT Spectrum of Current vs. V_exit')
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # # import numpy as np

# # # # freqs, amps = fft_from_df(df, 'V_exit (V)', 'Current (A)')

# # # # 1) find the index of the maximum amplitude
# # # peak_idx = np.argmax(amps)

# # # # 2) pull out the corresponding frequency and amplitude
# # # peak_freq = freqs[peak_idx]
# # # peak_amp  = amps[peak_idx]

# # # print(f"Peak at {peak_freq:.3f} 1/V with amplitude {peak_amp:.3e} A")

# # from scipy.signal import find_peaks

# # # find all peaks above some minimal height
# # peaks, props = find_peaks(amps, height=1e-16)

# # # freqs[peaks] is an array of peak‑frequencies
# # # props['peak_heights'] is an array of their amplitudes

# # for f, a in zip(freqs[peaks], props['peak_heights']):
# #     print(f"Peak at {f:.3f}1/V → {a:.3e} A")

# # # from scipy.signal import find_peaks

# # # # find all local peaks in the amplitude array
# # # peaks, props = find_peaks(amps)

# # # # drop the DC peak (freq = 0)
# # # peaks = peaks[freqs[peaks] > 0]

# # # # pick the largest of the remaining peaks
# # # second_peak = peaks[np.argmax(amps[peaks])]
# # # second_peak_freq = freqs[second_peak]
# # # second_peak_amp  = amps[second_peak]

# # # print(f"Second peak at {second_peak_freq:.3f} 1/V with amplitude {second_peak_amp:.3e} A")

# # # from scipy.signal import find_peaks

# # # # find all local peaks in the amplitude array
# # # peaks, props = find_peaks(amps)

# # # # drop the DC peak (freq = 0)
# # # peaks = peaks[freqs[peaks] > 0]

# # # # pick the largest of the remaining peaks
# # # second_peak = peaks[np.argmax(amps[peaks])]
# # # second_peak_freq = freqs[second_peak]
# # # second_peak_amp  = amps[second_peak]

# # # print(f"Second peak at {second_peak_freq:.3f} 1/V with amplitude {second_peak_amp:.3e} A")


# # # from scipy.signal import find_peaks

# # # # find all local peaks in the amplitude array
# # # peaks, props = find_peaks(amps)

# # # # drop the DC peak (freq = 0)
# # # peaks = peaks[freqs[peaks] > 0]

# # # # pick the largest of the remaining peaks
# # # second_peak = peaks[np.argmax(amps[peaks])]
# # # second_peak_freq = freqs[second_peak]
# # # second_peak_amp  = amps[second_peak]

# # # print(f"Second peak at {second_peak_freq:.3f} 1/V with amplitude {second_peak_amp:.3e} A")

import pandas as pd

# 1) Load your data
# df = pd.read_csv('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv')  # replace with your filename

# 2) Choose your moving‐average window (e.g. 5 points)
window = 5

# 3) Compute the centered moving average of the current
df['Current_MA'] = df['Current (A)'] \
    .rolling(window=window, center=True, min_periods=1) \
    .mean()

# 4) Create the delta column: current minus its moving average
df['Current_minus_MA'] = df['Current (A)'] - df['Current_MA']

df.to_csv('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\output.csv', index=False, float_format='%.3e')


df.plot(kind = 'line', x = 'V_ent (V)', y = 'Current_minus_MA')

import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\output.csv')


# infer step
step = df['V_exit (V)'].diff().mode()[0]

# extend ±0.08
vmin, vmax = df['V_exit (V)'].min(), df['V_exit (V)'].max()
new_low  = np.arange(vmin - 0.08, vmin, step)
new_high = np.arange(vmax + step, vmax + 0.08 + step/2, step)

# shift
shift = (df['V_exit (V)'] - df['V_ent (V)']).mean()

# helper to build zero rows
def make_zero_rows(vs):
    return pd.DataFrame({
        'V_exit (V)': vs,
        'V_ent (V)':  vs - shift,
        'Current (A)': 0.0,
        'Current_MA':  0.0,
        'Current_minus_MA': 0.0
    })

df_low  = make_zero_rows(new_low)
df_high = make_zero_rows(new_high)

df_new = pd.concat([df_low, df, df_high], ignore_index=True)
df_new.sort_values('V_exit (V)', inplace=True)
df_new.reset_index(drop=True, inplace=True)

# Save if you like
df_new.to_csv('extended_dataset.csv', index=False)

df = df_new

ax = df.plot(x='V_ent (V)',
             y='Current (A)',
             label='I vs V_ent',
             legend=True)

df.plot(x='V_ent (V)',
        y='Current_minus_MA',
        label='I_MA vs V_ent',
        ax=ax)

ax.set_xlabel('V_ent (V)')
ax.set_ylabel('Current (A)')
#ax.set_title('Current vs Gate Voltages')
plt.tight_layout()
plt.savefig('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\plot.png')



# 5) Inspect the result
print(df.head(10))

import numpy as np
import pandas as pd
from typing import Tuple

# df = pd.read_csv("C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv")

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


# df = pd.read_csv("C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv")
#df = df[df['Current (A)'] > 1e-11]
freqs, amps = fft_from_df(df, 'V_ent (V)', 'Current_minus_MA')

print(f'freq: {freqs}, amp: {amps}')

import matplotlib.pyplot as plt

# assume you’ve already done:
# freqs, amps = fft_from_df(df, 'V_exit (V)', 'Current (A)')

plt.figure()
plt.plot(freqs, amps, '-')
plt.xlabel('Frequency (1/V)')
plt.ylabel('Amplitude (A)')
plt.title('FFT Spectrum of Current vs. V_exit')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\plot1.png')


from scipy.signal import find_peaks

# find all local peaks in the amplitude array
peaks, props = find_peaks(amps)

# drop the DC peak (freq = 0)
# peaks = peaks[freqs[peaks] > 0]

# pick the largest of the remaining peaks
second_peak = peaks[np.argmax(amps[peaks])]
second_peak_freq = freqs[second_peak]
second_peak_amp  = amps[second_peak]

print(f"Second peak at {second_peak_freq:.3f} 1/V with amplitude {second_peak_amp:.3e} A")

freqs, amps = fft_from_df(df, 'V_exit (V)', 'Current_minus_MA')
print(freqs, amps)

import numpy as np
from scipy.signal import find_peaks

# 1) Locate all local maxima in the amplitude array
peaks, _ = find_peaks(amps)

# 2) Exclude the DC peak at freq = 0
nonzero_peaks = peaks[freqs[peaks] > 0]

if nonzero_peaks.size == 0:
    raise RuntimeError("No non‑zero peaks found")

# 3) Find the peak whose frequency is smallest (the “first” in frequency)
first_peak_idx = nonzero_peaks[np.argmin(freqs[nonzero_peaks])]
first_peak_freq = freqs[first_peak_idx]
first_peak_amp  = amps[first_peak_idx]


print(f"First peak at {first_peak_freq:.3f} 1/V with amplitude {first_peak_amp:.3e} A")

# plt.figure()
# for 'V_ent (V)', currents, label in df:
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


# ax = df.plot(x='V_ent (V)',
#              y='Current (A)',
#              label='I vs V_ent',
#              legend=True)



# ax.set_xlabel('V_ent (V)')
# ax.set_ylabel('Current (A)')
# #ax.set_title('Current vs Gate Voltages')
# plt.tight_layout()
# plt.savefig('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\plot.png')

import numpy as np
from scipy.signal import find_peaks

# 1) Locate all local maxima in the amplitude array
peaks, _ = find_peaks(amps)

# 2) Exclude the DC peak at freq = 0
nonzero_peaks = peaks[freqs[peaks] > 0]

if nonzero_peaks.size == 0:
    raise RuntimeError("No non‑zero peaks found")

# 3) Find the peak whose frequency is smallest (the “first” in frequency)
first_peak_idx = nonzero_peaks[np.argmin(freqs[nonzero_peaks])]
first_peak_freq = freqs[first_peak_idx]
first_peak_amp  = amps[first_peak_idx]


print(f"First peak at {first_peak_freq:.3f} 1/V with amplitude {first_peak_amp:.3e} A")


import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 1) Load your data
# df = pd.read_csv("C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv")
#df = pd.read_csv('your_data.csv')   # or however you have your DataFrame

# 2) Extract the series you want peak‑find on
y = df['Current (A)'].to_numpy()

# 3) Find peaks
#    - height=… can be used to ignore small fluctuations
#    - distance=… can be used to enforce a minimum spacing between peaks
peaks, props = find_peaks(y,
                          height=9.0e-13,    # only peaks above 1e-13 A
                          distance=2       # at least 2 samples apart
                         )

# 4) Inspect peak locations and heights
peak_freqs = df['V_exit (V)'].to_numpy()[peaks]
peak_amps  = props['peak_heights']

for f, a in zip(peak_freqs, peak_amps):
    print(f"Peak at V_exit = {f:.3f} V → Current = {a:.3e} A")

# 5) (Optional) mark them on a plot
plt.figure()
plt.plot(df['V_exit (V)'], y,   label='Current sweep')
plt.plot(peak_freqs, peak_amps, 'ro', label='Detected peaks')
plt.xlabel('V_exit (V)')
plt.ylabel('Current (A)')
plt.legend()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks

# # 1) Load your data
# #df = pd.read_csv('your_data.csv')   # replace with your filename
# df = pd.read_csv("C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv")

# # 2) Extract the current trace
# y = df['Current (A)'].to_numpy()
# x = df['V_exit (V)'].to_numpy()

# # 3) Find all peaks
# #    You can tune `height` or `prominence` to ignore tiny fluctuations:
# peaks, props = find_peaks(y,
#                           height=3,        # only peaks > 0 A
#                           prominence=1e-13 # peaks must stand out by at least 1e-11 A
#                          )

# # 4) Print them
# print("Detected peaks:")
# for idx in peaks:
#     print(f" • V_exit = {x[idx]:.3f} V → Current = {y[idx]:.3e} A")

# # 5) Plot and mark them
# plt.figure()
# plt.plot(x, y, label='Current sweep')
# plt.plot(x[peaks], y[peaks], 'ro', label='Detected peaks')
# plt.xlabel('V_exit (V)')
# plt.ylabel('Current (A)')
# plt.title('Peak detection in I vs V_exit')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# from io import StringIO
# df = pd.read_csv(StringIO('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv'))

# # 2) Determine original steps and shift between channels
# step_exit = df['V_exit (V)'].diff().mode()[0]
# step_ent  = df['V_ent (V)'].diff().mode()[0]
# # average shift V_exit - V_ent
# shift = (df['V_exit (V)'] - df['V_ent (V)']).mean()

# # 3) Build extended ranges ±0.08 around original min/max
# min_exit, max_exit = df['V_exit (V)'].min(), df['V_exit (V)'].max()
# low_exit  = np.arange(min_exit - 0.08, min_exit, step_exit)
# high_exit = np.arange(max_exit + step_exit, max_exit + 0.08 + step_exit/2, step_exit)

# # corresponding vents by preserving original shift
# low_ent  = low_exit - shift
# high_ent = high_exit - shift

# # 4) Create DataFrames for new points (current = 0)
# df_low  = pd.DataFrame({'V_exit (V)': low_exit, 'V_ent (V)': low_ent,   'Current (A)': 0.0})
# df_high = pd.DataFrame({'V_exit (V)': high_exit,'V_ent (V)': high_ent,  'Current (A)': 0.0})

# # 5) Concatenate and sort by V_exit
# df_ext = pd.concat([df_low, df, df_high], ignore_index=True)
# df_ext.sort_values('V_exit (V)', inplace=True)
# df_ext.reset_index(drop=True, inplace=True)

# # # Display the new dataset
# # import ace_tools as tools
# # tools.display_dataframe_to_user("Extended dataset", df_ext)

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import StringIO

directory = "C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv"

df = pd.read_csv(directory)


# 2) Determine original steps and shift between channels
step_exit = df['V_exit (V)'].diff().mode()[0]
step_ent  = df['V_ent (V)'].diff().mode()[0]
# average shift V_exit - V_ent
shift = (df['V_exit (V)'] - df['V_ent (V)']).mean()

# 3) Build extended ranges ±0.08 around original min/max
min_exit, max_exit = df['V_exit (V)'].min(), df['V_exit (V)'].max()
low_exit  = np.arange(min_exit - 0.08, min_exit, step_exit)
high_exit = np.arange(max_exit + step_exit, max_exit + 0.08 + step_exit/2, step_exit)

# corresponding vents by preserving original shift
low_ent  = low_exit - shift
high_ent = high_exit - shift

# 4) Create DataFrames for new points (current = 0)
df_low  = pd.DataFrame({'V_exit (V)': low_exit, 'V_ent (V)': low_ent,   'Current (A)': 0.0})
df_high = pd.DataFrame({'V_exit (V)': high_exit,'V_ent (V)': high_ent,  'Current (A)': 0.0})

# 5) Concatenate and sort by V_exit
df_ext = pd.concat([df_low, df, df_high], ignore_index=True)
df_ext.sort_values('V_exit (V)', inplace=True)
df_ext.reset_index(drop=True, inplace=True)

print(df_ext)

# # Display the new dataset
# import ace_tools as tools
# tools.display_dataframe_to_user("Extended dataset", df_ext)

import numpy as np
from scipy.signal import find_peaks

# freqs, amps = fft_from_df(padded_data, 'V_ent (V)', 'Current_minus_MA')

# 1) find all local maxima
peaks, _ = find_peaks(amps)

# 2) filter out the DC bin (and any tiny residual around 0)
tol = 1e-12
peaks = peaks[np.abs(freqs[peaks]) > tol]

# 3) if there are any remaining peaks, pick the one with smallest frequency
if peaks.size:
    first_peak_idx = peaks[np.argmin(freqs[peaks])]
    first_peak_freq = freqs[first_peak_idx]
    first_peak_amp  = amps[first_peak_idx]
    print(f"First non‑zero peak at {first_peak_freq:.3f} 1/V with amplitude {first_peak_amp:.3e} A")
else:
    print("No non‑zero peaks found.")


import numpy as np

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

    #         if V1 >= 1.1 and V2 >= 1.1:
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




    # Step (5vi): Sweep the values V_exit for each V_ent and get a colormap.

    # def sweep_map(sim: Any,
    #             agilent: Any,
    #             slot_exit: int,
    #             slot_ent: int,
    #             Vexit_avg: float,
    #             Vent_avg: float,
    #             Vexit_span: float,
    #             Vent_span: float,
    #             step_exit: float = 0.1,
    #             step_ent: float = 0.1,
    #             delay: float = 0.5,
    #             output_csv: str = 'map_data.csv',
    #             output_fig: str = 'map_colormap.png') -> None:
    #     """
    #     For each Vent in [Vent_avg, Vent_avg + Vent_span] (step_ent):
    #     1) sweep Vexit from Vexit_avg → Vexit_avg+Vexit_span (step_exit),
    #     2) record (Vent, Vexit, current),
    #     3) then ramp Vexit back down from max → Vexit_avg (step_exit),
    #     save all to CSV and draw a colormap.

    #     Args:
    #     sim:       Instrument controller (with .set_voltage / .get_voltage).
    #     agilent:   DMM instance (agilent.volt() → measured voltage).
    #     slot_exit: Channel index for Vexit.
    #     slot_ent:  Channel index for Vent.
    #     Vexit_avg: Base Vexit.
    #     Vent_avg:  Base Vent.
    #     Vexit_span: ΔVexit to sweep.
    #     Vent_span:  ΔVent to sweep.
    #     step_exit: Vexit increment/decrement per point.
    #     step_ent:  Vent increment per line.
    #     delay:     Settle time after each set_voltage.
    #     output_csv: Path for CSV output.
    #     output_fig: Path for colormap PNG.
    #     """
    #     # build the sweep axes
    #     Vexit_vals = np.arange(Vexit_avg,
    #                         Vexit_avg + Vexit_span + 1e-9,
    #                         step_exit)
    #     Vent_vals = np.arange(Vent_avg,
    #                         Vent_avg + Vent_span + 1e-9,
    #                         step_ent)

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
    #     df.to_csv(directory, index=False,float_format='%.3e')

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

    # #sweep_map(sim,agilent,4,5,Vexit_avg,Vent_avg,0.020,0.020,0.001,0.001,0.1,output_csv='qpc_map',output_fig='qpc_map')
