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

sim = SIM928("sim900", "GPIB0::3::INSTR", slot_names={i: f'ch{i}' for i in range (1,9)})
agilent = Agilent34401A("agilent", "GPIB0::21::INSTR")

# The following gates are assumed to be connected to the following channels in sim900:
# Channel 1: Top gate
# Channel 2: QPC bottom
# Channel 3: QPC top
# Channel 4: Exit gate
# Channel 5: Entrance gate
# Channel 6: Source-drain bias (currently set to zero)

# # Connecting to SIM900 and Agilent

sim = SIM928("sim900", "GPIB0::3::INSTR", slot_names={i: f'ch{i}' for i in range (1,9)})
agilent = Agilent34401A("agilent", "GPIB0::21::INSTR")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
qpc_voltage = sim.get_voltage(3)

########## Functions ###########
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
    base_dir = 'C:\\Users\\coher\\Desktop\\Kamayani\\Data\\'
    filename_1 = f'{qpc_voltage}V_{filename}_{timestamp}.csv'
    directory = os.path.join(base_dir, filename_1)
    df.to_csv(directory, index=False,float_format='%.3e')

    # 3) Scatter plot of the first two columns
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_label = df.columns[0]
    y_label = df.columns[1]

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

