import tkinter as tk
from tkinter import ttk

from tkinter import messagebox, filedialog
import yaml
import qcodes
import qcodes
from qcodes.dataset import (
    LinSweep,
    TogetherSweep,
    do1d,
    do2d,
    dond,
    Measurement,
    load_or_create_experiment,
    plot_dataset,
    initialise_or_create_database_at
)
import datetime

class StationWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Station Configuration")
        self.geometry("550x275")
        self.devices = []

        self.label = tk.Label(self, text="QCoDeS Station", font=("Helvetica", 14, "bold"))
        self.label.pack()

        self.load_frame = tk.Frame(self)
        self.load_frame.pack(fill="x", pady=(0, 5))
        self.load_label = tk.Label(self.load_frame, text="Config File:", width=10)
        self.load_label.pack(side="left")
        self.load_text = tk.StringVar()
        self.load_text.set("No file chosen")
        self.load_entry = tk.Entry(self.load_frame, textvariable=self.load_text, state="readonly", width=40)
        self.load_entry.pack(side="left", padx=(0, 5))
        self.load_button = tk.Button(self.load_frame, text="Load", command=self.load_config_file)
        self.load_button.pack(side="left")

        self.device_label = tk.Label(self, text="Select Device:", font=("Helvetica", 10))
        self.device_label.pack()

        self.device_var = tk.StringVar()
        self.device_dropdown = tk.OptionMenu(self, self.device_var, "")
        self.device_dropdown.pack()

        self.device_listbox_frame = tk.Frame(self)
        self.device_listbox_frame.pack(pady=(5, 0))
        self.device_listbox = tk.Listbox(self.device_listbox_frame, height=5, width=50)
        self.device_listbox.pack()

        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=5)

        self.add_button = tk.Button(self.buttons_frame, text="Add", command=self.add_device, width=10)
        self.add_button.pack(side="left", padx=(0, 5))
        self.delete_button = tk.Button(self.buttons_frame, text="Delete", command=self.delete_device, width=10)
        self.delete_button.pack(side="left")

        self.create_button = tk.Button(self, text="Create Station", command=self.create_station, width=30)
        self.create_button.pack(pady=(10, 0))

    def load_config_file(self):
        filename = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml"), ("All Files", "*.*")])
        if filename:
            try:
                self.load_text.set(filename)
                self.station = qcodes.Station(config_file=filename)
                with open(filename, "r") as file:
                    self.config = yaml.safe_load(file)
                    self.available_devices = list(self.config["instruments"].keys())
                    self.device_var.set("")
                    menu = self.device_dropdown["menu"]
                    menu.delete(0, "end")
                    for device in self.available_devices:
                        menu.add_command(label=device, command=lambda d=device: self.device_var.set(d))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config file: {str(e)}")

    def add_device(self):
        selected_device = self.device_var.get()
        if selected_device:
            self.devices.append(selected_device)
            self.device_listbox.insert(tk.END, selected_device)
        else:
            messagebox.showerror("Error", "Please select a device to add.")

    def delete_device(self):
        selected_index = self.device_listbox.curselection()
        if selected_index:
            selected_device = self.device_listbox.get(selected_index)
            self.device_listbox.delete(selected_index)
            self.devices.remove(selected_device)
        else:
            messagebox.showerror("Error", "Please select a device to delete.")

    def create_station(self):
        if self.devices:
            isOneError = False
            for device in self.devices:
                try:
                    self.station.load_instrument(device)
                except ValueError as e:
                    isOneError = True
                    messagebox.showinfo("Error", f"Failed to load instrument {str(device)}: {str(e)}")        
            if not(isOneError):
                messagebox.showinfo("Success", "Station created successfully.")
                self.master.load_station()  # Set the check mark in the main GUI
        else:
            messagebox.showerror("Error", "No devices added. Please add at least one device.")

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QCoDeS Measurement GUI")

        # Set-up window height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        self.geometry(f"{window_width}x{window_height}")

        self.partition_main_frame()

        self.configure_settings(self.settings_frame)

        self.configure_misc(self.misc_frame)

        self.configure_constants(self.constants_frame)

        self.configure_sweeps(self.sweep_frame)

        self.configure_analysis(self.analysis_frame)

    def partition_main_frame(self):
        self.settings_frame = tk.Frame(self, background="#FFF0C1", bd=1, relief="raised")
        self.settings_frame_title = tk.Label(self.settings_frame, text="Settings", font=("Helvetica", 16, "underline"))
        self.settings_frame_title.pack()

        self.misc_frame = tk.Frame(self, background="#FEF0C1", bd=1, relief="raised")
        self.misc_frame_title = tk.Label(self.misc_frame, text="Misc.", font=("Helvetica", 16, "underline"))
        self.misc_frame_title.pack()

        self.constants_frame = tk.Frame(self, background="#D2E2FB", bd=1, relief="raised")
        self.constants_frame_title = tk.Label(self.constants_frame, text="Voltage Constant(s)", font=("Helvetica", 16, "underline"))
        self.constants_frame_title.pack()

        self.sweep_frame = tk.Frame(self, background="#CCE4CA", bd=1, relief="raised")
        self.sweep_frame_title = tk.Label(self.sweep_frame, text="Voltage Sweep(s)", font=("Helvetica", 16, "underline"))
        self.sweep_frame_title.pack()

        self.analysis_frame = tk.Frame(self, background="#F5C2C1", bd=1, relief="raised")
        self.analysis_frame_title = tk.Label(self.analysis_frame, text="Data Analysis", font=("Helvetica", 16, "underline"))
        self.analysis_frame_title.pack()

        self.settings_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.misc_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.constants_frame.grid(row=1, column=0,  columnspan=2, sticky="nsew", padx=2, pady=2)
        self.sweep_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        self.analysis_frame.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=2, pady=2)

        self.grid_rowconfigure(0, weight=2)
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=3)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=4)

    def configure_settings(self, frame):
        # Button to construct station
        def open_station_window():
            self.station_window = StationWindow(self)
            
        self.construct_station_button = tk.Button(frame, text="Load Station", command=open_station_window)
        self.construct_station_button.pack()

        today_date = datetime.date.today().strftime("%Y-%m-%d")
        db_name = f"~/experiments_{today_date}.db"

        # Create a StringVar to hold the text in the entry widget
        db_name_var = tk.StringVar()
        db_name_var.set(db_name)

        self.sample_name_label = tk.Label(frame, text="Active Database:")
        self.sample_name_label.pack()
        entry = tk.Entry(frame, textvariable=db_name_var, state="disabled", width=25)
        entry.pack()

        # Create labeled entries
        self.sample_name_entry = LabeledEntry(frame, "Sample:", "Set", self.set_sample_name)
        self.sample_name_entry.pack(anchor="n")
        self.exp_name_entry = LabeledEntry(frame, "Experiment:", "Set", self.set_exp_name)
        self.exp_name_entry.pack(anchor="n")
        self.measurement_name_entry = LabeledEntry(frame, "Measurement:", "Set", self.set_measurement_name)
        self.measurement_name_entry.pack(anchor="n")

    def set_sample_name(self):
        # Set database name action
        self.sample_name = self.sample_name_entry.entry.get()

    def set_exp_name(self):
        # Set experiment name action
        self.exp_name = self.exp_name_entry.entry.get()
        self.experiment = load_or_create_experiment(self.exp_name, sample_name=self.sample_name)

    def set_measurement_name(self):
        # Set experiment name action
        self.measurement_name = self.exp_name_entry.entry.get()
        self.measurement = Measurement(exp=self.experiment)

    def configure_misc(self, frame):
        self.sensitivity_entry = LabeledEntry(frame, "Sensitivity:", "Set", self.set_sensitivity, entry_text="1")
        self.sensitivity_entry.pack(anchor="n")
        
        self.preamp_bias_entry = LabeledEntry(frame, "Pre-Amp Bias (A/V):", "Set", self.set_preamp_bias, entry_text="0")
        self.preamp_bias_entry.pack(anchor="n")

    def set_sensitivity(self):
        self.sensitivity = self.sensitivity_entry.entry.get()

    def set_preamp_bias(self):
        self.preamp_bias_entry = self.preamp_bias_entry.entry.get()

    def configure_constants(self, frame):
        columns = ("name", "device", "module", "value")
        constantsTree = ttk.Treeview(frame, columns=columns)
        constantsTree.heading('name', text='Name')
        constantsTree.heading('device', text='Device')
        constantsTree.heading('module', text='Module')
        constantsTree.heading('value', text='Value (V)')

        constantsTree.column("#0", width=0)
        constantsTree.column("name", width=100)
        constantsTree.column("device", width=100)
        constantsTree.column("module", width=100)
        constantsTree.column("value", width=100)
        constantsTree.pack()

        # Add button to add constant
        add_button = tk.Button(frame, text="Add", command=lambda: self.add_element(constantsTree, columns))
        add_button.pack()

        # Add button to delete selected constant
        delete_button = tk.Button(frame, text="Delete", command=lambda: self.delete_element(constantsTree))
        delete_button.pack()

        # Add button to delete selected constant
        group_button = tk.Button(frame, text="Group", command=lambda: self.group_elements(constantsTree))
        group_button.pack()
        
        # Add button to delete selected constant
        group_button = tk.Button(frame, text="Set", command=lambda: self.set_constants(constantsTree))
        group_button.pack()

    def add_element(self, tree, columns):
        # Insert a new row with empty values    
        tree.insert("", "end", values=columns, tags="editable")

        def onDoubleClick(event):
            ''' Executed, when a row is double-clicked. Opens 
            read-only EntryPopup above the item's column, so it is possible
            to select text '''

            # close previous popups
            try:  # in case there was no previous popup
                tree.entryPopup.destroy()
            except AttributeError:
                pass

            # what row and column was clicked on
            rowid = tree.identify_row(event.y)
            column = tree.identify_column(event.x)

            # handle exception when header is double click
            if not rowid:
                return

            # get column position info
            x,y,width,height = tree.bbox(rowid, column)

            # y-axis offset
            pady = height // 2

            # place Entry popup properly
            text = tree.item(rowid, 'values')[int(column[1:])-1]
            tree.entryPopup = EntryPopup(tree, rowid, int(column[1:])-1, text)
            tree.entryPopup.place(x=x, y=y+pady, width=width, height=height, anchor='w')

        # Bind the <Return> key to the on_enter function for the Treeview
        tree.bind("<Double-1>", onDoubleClick)

    def delete_element(self, tree):
        # Implement delete constant functionality
            # Get the ID(s) of the selected item(s)
        selected_items = tree.selection()
        
        # Iterate over the selected item(s) and delete them
        for item_id in selected_items:
            tree.delete(item_id)

    def group_elements(self, tree):
        # Get the selected items
        selected_items = tree.selection()

        # Check if any items are selected
        if not selected_items:
            messagebox.showwarning("Warning", "No sweeps selected.")
            return

        # Gather the selected sweeps
        selected_sweeps = []
        for item_id in selected_items:
            # Get the values of the selected item
            values = tree.item(item_id, "values")
            # Append the values to the selected_sweeps list
            selected_sweeps.append(values)

        # Create a TogetherSweep with the selected sweeps
        together_sweep = selected_sweeps
        print(together_sweep)

        # Change the background color of the selected rows
        for item_id in selected_items:
            tree.tag_configure("grouped", background="red")  # Configure the tag to change background color
            tree.item(item_id, tags="grouped") 

    def set_constants(self, tree):
        # Get the IDs of all the items (rows) in the tree
        item_ids = tree.get_children()

        # Iterate over each item ID and access its values
        for item_id in item_ids:
            # Access the values of the item
            values = tree.item(item_id, "values")
            model, device, module, value = values
            print(model,device, module, value)

    def run_sweeps(self,tree):
        pass

    def configure_sweeps(self, frame):
        columns = ("name", "device", "module", "min_value", 'max_value', 'num_steps')
        sweepsTree = ttk.Treeview(frame, columns=columns)
        sweepsTree.heading('name', text='Name')
        sweepsTree.heading('device', text='Device')
        sweepsTree.heading('module', text='Module')
        sweepsTree.heading('min_value', text='Vmin (V)')
        sweepsTree.heading('max_value', text='Vmax (V)')
        sweepsTree.heading('num_steps', text='N')

        sweepsTree.column("#0", width=0)
        sweepsTree.column("name", width=100)
        sweepsTree.column("device", width=100)
        sweepsTree.column("module", width=100)
        sweepsTree.column("min_value", width=100)
        sweepsTree.column("max_value", width=100)
        sweepsTree.column("num_steps", width=100)
        sweepsTree.pack()

        # Add button to add constant
        add_button = tk.Button(frame, text="Add", command=lambda: self.add_element(sweepsTree, columns))
        add_button.pack()

        # Add button to delete selected constant
        delete_button = tk.Button(frame, text="Delete", command=lambda: self.delete_element(sweepsTree))
        delete_button.pack()

        # Add button to delete selected constant
        group_button = tk.Button(frame, text="Group", command=lambda: self.group_elements(sweepsTree))
        group_button.pack()

        # Add button to delete selected constant
        group_button = tk.Button(frame, text="Run", command=lambda: self.run_sweeps(sweepsTree))
        group_button.pack()

    def configure_analysis(self, frame):
        pass

    def load_station(self):
        self.station = self.station_window.station
        self.devices = self.station_window.devices

class EntryPopup(ttk.Entry):
    def __init__(self, parent, iid, column, text, **kw):
        ttk.Style().configure('pad.TEntry', padding='1 1 1 1')
        super().__init__(parent, style='pad.TEntry', **kw)
        self.tv = parent
        self.iid = iid
        self.column = column

        self.insert(0, text) 
        # self['state'] = 'readonly'
        # self['readonlybackground'] = 'white'
        # self['selectbackground'] = '#1BA1E2'
        self['exportselection'] = False

        self.focus_force()
        self.select_all()
        self.bind("<Return>", self.on_return)
        self.bind("<Control-a>", self.select_all)
        self.bind("<Escape>", lambda *ignore: self.destroy())


    def on_return(self, event):
        rowid = self.tv.focus()
        vals = self.tv.item(rowid, 'values')
        vals = list(vals)
        vals[self.column] = self.get()
        self.tv.item(rowid, values=vals)
        self.destroy()


    def select_all(self, *ignore):
        ''' Set selection on the whole text '''
        self.selection_range(0, 'end')

        # returns 'break' to interrupt default key-bindings
        return 'break'

class LabeledEntry(tk.Frame):
    def __init__(self, master, label_text, button_text, button_command,entry_text=""):
        super().__init__(master)
        self.label = tk.Label(self, text=label_text)
        self.label.pack(side="left")
        self.entry_var = tk.StringVar()
        self.entry_var.set(entry_text)
        self.entry = tk.Entry(self, textvariable=self.entry_var)
        self.entry.pack(side="left")
        self.entry.bind("<Double-1>", self.enable_entry)

        self.button = tk.Button(self, text=button_text, command=lambda: self.button_click(button_command))
        self.button.pack(side="left")

    def button_click(self, command):
        command()
        self.entry.configure(state="disabled")

    def enable_entry(self, event):
        self.entry.configure(state="normal")

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
