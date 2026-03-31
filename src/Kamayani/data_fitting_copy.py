import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple

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
directory = "C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv"
raw_data = pd.read_csv(directory)

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

