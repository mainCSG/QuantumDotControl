import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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


directory = "C:\\Users\\coher\\Desktop\\Kamayani\\Data\\sweep_Vent_Vexit_20250724_152631.csv"
df = pd.read_csv(directory)

print(df)

# 2) Choose your moving‐average window (e.g. 5 points)
window = 5

# 3) Compute the centered moving average of the current
df['Current_MA'] = df['Current (A)'] \
    .rolling(window=window, center=True, min_periods=1) \
    .mean()

# 4) Create the delta column: current minus its moving average
df['Current_minus_MA'] = df['Current (A)'] - df['Current_MA']

print(df) 

# list the columns you want
cols = ['V_exit (V)','V_ent (V)', 'Current_minus_MA']

# slice out just those columns and make a copy
new_data = df[cols].copy()

# inspect
print(new_data)



# infer step
step = new_data['V_exit (V)'].diff().mode()[0]

# extend ±0.08
vmin, vmax = new_data['V_exit (V)'].min(), new_data['V_exit (V)'].max()
new_low  = np.arange(vmin - 0.8, vmin, step)
new_high = np.arange(vmax + step, vmax + 0.8 + step/2, step)

# shift
shift = (new_data['V_exit (V)'] - new_data['V_ent (V)']).mean()

# helper to build zero rows
def make_zero_rows(vs):
    return pd.DataFrame({
        'V_exit (V)': vs,
        'V_ent (V)':  vs - shift,
        'Current_minus_MA': 0.0
    })

df_low  = make_zero_rows(new_low)
df_high = make_zero_rows(new_high)

df_new = pd.concat([df_low, df, df_high], ignore_index=True)
df_new.sort_values('V_exit (V)', inplace=True)
df_new.reset_index(drop=True, inplace=True)


# Save if you like
df_new.to_csv('extended_dataset.csv', index=False)



print(df_new.to_string())

df_new.plot(kind = 'line', x = 'V_ent (V)', y = 'Current_minus_MA')
plt.show()

freqs, amps = fft_from_df(df_new, 'V_ent (V)', 'Current_minus_MA')

print(f'freq: {freqs}, amp: {amps}')

plt.figure()
plt.plot(freqs, amps, '-')
plt.xlabel('Frequency (1/V)')
plt.ylabel('Amplitude (A)')
plt.title('FFT Spectrum of Current vs. V_exit')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('C:\\Users\\coher\\Desktop\\Kamayani\\Data\\plot1.png')