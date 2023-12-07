from charge_noise_tool import *

# import qcodes as qc

# database_path = './data/db_20231121b.db'
# qc.initialise_or_create_database_at(database_path)
# Load DataSet by ID
# dataset_id = 1  # Replace with the actual DataSet ID
# dataset = qc.load_by_run_spec(captured_run_id=dataset_id)

# # Extract data as a pandas DataFrame
# data_frame = dataset.get_data_as_pandas_dataframe()

# # Load DataSet by ID
# dataset_id = 1  # Replace with the actual DataSet ID
# dataset = qc.load_by_run_spec(captured_run_id=dataset_id)
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.dat")])
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

    try:
        data_csv = pd.read_csv(file_path, skiprows=[0,2], sep='\t')
        data_df = pd.DataFrame(data_csv)

        # Remove artifacts
        data_df.columns = data_df.columns.str.replace('[#, ,"]','',regex=True)
        update_column_comboboxes(data_df)
        
    except pd.errors.EmptyDataError:
        result_label.config(text="Error: The selected file is empty.")
        return
    except pd.errors.ParserError:
        result_label.config(text="Error: Unable to parse the selected file. Please check the file format.")
        return

def update_column_comboboxes(df):
    # Update the combobox options with column names
    columns = df.columns.tolist()
    x1_combobox['values'] = columns
    x2_combobox['values'] = columns
    y_combobox['values'] = columns
    
def analyze():

    X1 = x1_combobox.get()
    X2 = x2_combobox.get()
    Y = y_combobox.get()

    file_path = file_path_entry.get()

    root.destroy()

    data_csv = pd.read_csv(file_path, skiprows=[0,2], sep='\t')
    data_df = pd.DataFrame(data_csv)

    # Remove artifacts
    data_df.columns = data_df.columns.str.replace('[#, ,"]','',regex=True)

    Palpatine = ChargeNoiseExtractor()

    VST_sweep = np.unique(np.array(data_df[X1]))
    VSD_sweep = np.unique(np.array(data_df[X2]))

    ISD_2D = np.rot90(
        np.array(data_df[Y]).reshape(len(VSD_sweep),len(VST_sweep)),0
    )

    ISD_1D = ISD_2D.T[:,7]
    VSD_1D = VSD_sweep[7]

    VST_max, G_max = Palpatine.get_VST_for_Gmax(VST_sweep, ISD_1D, VSD_1D, plot=False)

    Palpatine.get_lever_arms(
        VST_sweep, 
        VSD_sweep, 
        ISD_2D, 
        VST_window=(-np.inf, np.inf), 
        VSD_window=(-np.inf, np.inf),
        automated=False
    )

# Create the main window
root = tk.Tk()
root.title("Data Selection")

# Create and place widgets
frame = ttk.Frame(root, padding="10")
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# File selection widgets
file_path_label = ttk.Label(frame, text="Select a .dat file:")
file_path_label.grid(column=0, row=0, padx=10, pady=5, sticky=tk.W)

file_path_entry = ttk.Entry(frame, width=30)
file_path_entry.grid(column=1, row=0, padx=10, pady=5, sticky=tk.W)

browse_button = ttk.Button(frame, text="Browse", command=browse_file)
browse_button.grid(column=2, row=0, pady=5)

# Column selection labels
x1_label = ttk.Label(frame, text="Select VST column:")
x1_label.grid(column=0, row=1, padx=10, pady=5, sticky=tk.W)

x2_label = ttk.Label(frame, text="Select VSD column:")
x2_label.grid(column=0, row=2, padx=10, pady=5, sticky=tk.W)

y_label = ttk.Label(frame, text="Select ISD column:")
y_label.grid(column=0, row=3, padx=10, pady=5, sticky=tk.W)

# Create and populate comboboxes with column names
columns = []  # Empty list initially, to be populated after file selection
x1_combobox = ttk.Combobox(frame, values=columns, state="readonly")
x1_combobox.grid(column=1, row=1, padx=10, pady=5)
x1_combobox.set("")

x2_combobox = ttk.Combobox(frame, values=columns, state="readonly")
x2_combobox.grid(column=1, row=2, padx=10, pady=5)
x2_combobox.set("")

y_combobox = ttk.Combobox(frame, values=columns, state="readonly")
y_combobox.grid(column=1, row=3, padx=10, pady=5)
y_combobox.set("")

# Button to show selected columns
select_button = ttk.Button(frame, text="Analyze", command=analyze)
select_button.grid(column=0, row=4, columnspan=2, pady=10)

# Label to display the selected columns
result_label = ttk.Label(frame, text="")
result_label.grid(column=0, row=5, columnspan=2, pady=5)

# Start the GUI event loop
root.mainloop()
