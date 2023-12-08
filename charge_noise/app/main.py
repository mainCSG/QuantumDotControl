import tkinter as tk
from tkinter import ttk, filedialog
import qcodes as qc
import numpy as np
import matplotlib.pyplot as plt
import scipy 

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Function Selector")

        # Welcome message label
        welcome_label = ttk.Label(root, text="Welcome to the Function Selector App!")
        welcome_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Create menu
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Create submenu
        function_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Functions", menu=function_menu)

        # Add buttons to the submenu
        function_menu.add_command(label="LB-RB Scan", command=self.lb_rb_scan_window)
        function_menu.add_command(label="Coulomb Oscillations", command=self.coulomb_oscillations_window)
        function_menu.add_command(label="Current Noise", command=self.current_noise_window)
        function_menu.add_command(label="Lever Arm", command=self.lever_arm_window)

    def create_file_selection_window(self, function_name, process_function, additional_entries=None):
        file_selection_window = tk.Toplevel(self.root)
        file_selection_window.title(f"{function_name} - File Selection")

        file_path_label = ttk.Label(file_selection_window, text="Select a .db file:")
        file_path_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

        file_path_entry = ttk.Entry(file_selection_window, width=30)
        file_path_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        browse_button = ttk.Button(file_selection_window, text="Browse", command=lambda: self.browse_file(file_path_entry))
        browse_button.grid(row=0, column=2, pady=5)

        if additional_entries:
            for i, entry_label in enumerate(additional_entries):
                ttk.Label(file_selection_window, text=entry_label).grid(row=2+int(np.floor(i/2)), column=2*(i % 2), padx=10, pady=5, sticky=tk.W)
                ttk.Entry(file_selection_window, width=10).grid(row=int(np.floor(i/2)) + 2, column=1 +  2*(i % 2), padx=10, pady=5, sticky=tk.W)

        process_button = ttk.Button(file_selection_window, text="Process File", command=lambda: process_function(file_path_entry.get()))
        process_button.grid(row=len(additional_entries) + 2, column=0, columnspan=3, pady=10)

    def browse_file(self, entry_widget):
        file_path = filedialog.askopenfilename(filetypes=[("DB Files", "*.db")])
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)

    def current_noise_window(self):
        additional_entries = ["Run-ID [X,Y] (HS)", 
                              "VST HS (mV)",
                              "Run-ID [X,Y] (MS)",
                              "VST MS (mV)",
                              "Run-ID [X,Y] (LS)",
                              "VST LS (mV)",
                              "Run-ID [X,Y] (CB)",
                              "VST CB (mV)",
                              "LB (mV)", 
                              "RB (mV)", 
                              "VSD (mV)", 
                              "C (mV)"]
        self.create_file_selection_window("Current Noise", self.current_noise_function, additional_entries)

    def current_noise_function(self, file_path):
        # Retrieve values from entry boxes
        lb_value = self.get_entry_value("LB (mV)")
        rb_value = self.get_entry_value("RB (mV)")
        vsd_value = self.get_entry_value("VSD (mV)")
        c_value = self.get_entry_value("C (mV)")

        hs_runids = self.get_entry_value("Run-ID [X,Y] (HS)")
        vst_hs_value = self.get_entry_value("VST HS (mV)")
        ms_runids = self.get_entry_value("Run-ID [X,Y] (MS)")
        vst_ms_value = self.get_entry_value("VST MS (mV)")
        ls_runids = self.get_entry_value("Run-ID [X,Y] (LS)")
        vst_ls_value = self.get_entry_value("VST LS (mV)")
        cb_runids = self.get_entry_value("Run-ID [X,Y] (CB)")
        vst_cb_value = self.get_entry_value("VST CB (mV)")

        hs_runids = '[]' if hs_runids == '' else hs_runids
        ms_runids = '[]' if ms_runids == '' else ms_runids
        ls_runids = '[]' if ls_runids == '' else ls_runids
        cb_runids = '[]' if cb_runids == '' else cb_runids

        plot_labels = {}
        if hs_runids != '':
            plot_labels[str(tuple(eval(hs_runids)))] = "High Sensitivity"
        if ms_runids != '':
            plot_labels[str(tuple(eval(ms_runids)))] = "Medium Sensitivity"
        if ls_runids != '':
            plot_labels[str(tuple(eval(ls_runids)))] = "Low Sensitivity"
        if cb_runids != '':
            plot_labels[str(tuple(eval(cb_runids)))] = "Coulomb Blockade"

        run_ids = [eval(hs_runids)] + [eval(ms_runids)] + [eval(ls_runids)] + [eval(cb_runids)]
        run_ids = [ele for ele in run_ids if ele != []]

        data = {}
        # print(f"Current Noise function called with file: {file_path}, Run Id: {run_id}, LB: {lb_value}, RB: {rb_value}, VST: {vst_value}, VSD: {vsd_value}, C: {c_value}")
        for run_id in run_ids:
            t, Iavg = self.parse_db_file(file_path, run_id, is1Dsweep=True)
            data[str(tuple(run_id))] = (t, Iavg, plot_labels[str(tuple(run_id))])

        if eval(cb_runids) != []:
            t, Iavg = self.parse_db_file(file_path, eval(cb_runids), is1Dsweep=True)
            offset = np.mean(Iavg, keepdims=True).repeat(len(t[0]))

            for key, d in data.items():
                t, Iavg, label = d
                data[key] = (t, Iavg - offset,label)

        t = data[str(tuple(run_ids[0]))][0][0]
        sampling_rate = int(1/(t[1]-t[0]))

        # Plotting
        self.plot_current_and_spectrum(t, data, sampling_rate)

    def lb_rb_scan_window(self):
        additional_entries = ['Run-ID',"VST (mV)", "VSD (mV)", "C (mV)"]
        self.create_file_selection_window("LB-RB Scan", self.lb_rb_scan_function, additional_entries)

    def lb_rb_scan_function(self, file_path):
        vsd_value = self.get_entry_value("VSD (mV)")
        vst_value = self.get_entry_value("VST (mV)")
        c_value = self.get_entry_value("C (mV)")
        run_id = self.get_entry_value("Run-ID")
        
        labels = {str(tuple(eval(run_id))),"Current"}
        lb, rb, I = self.parse_db_file(file_path, eval(run_id), is2Dsweep=True)
        
        self.plot_lb_rb_scan(lb, rb, I)
        # print(f"LB-RB Scan function called with file: {file_path} and Run Id: {run_id}")

    def coulomb_oscillations_window(self):
        additional_entries = ["LB (mV)", "RB (mV)", "VSD (mV)", "C (mV)"]
        self.create_file_selection_window("Coulomb Oscillations", self.coulomb_oscillations_function, additional_entries)

    def coulomb_oscillations_function(self, file_path, run_id):
        print(f"Coulomb Oscillations function called with file: {file_path} and Run Id: {run_id}")

    def lever_arm_window(self):
        additional_entries = ["LB (mV)", "RB (mV)", "C (mV)"]
        self.create_file_selection_window("Lever Arm", self.lever_arm_function, additional_entries)

    def lever_arm_function(self, file_path, run_id):
        print(f"Lever Arm function called with file: {file_path} and Run Id: {run_id}")

    def get_entry_value(self, entry_label):
        # Helper method to retrieve the value from an entry box by label
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Toplevel):
                for index, child_widget in enumerate(widget.winfo_children()):
                    if isinstance(child_widget, ttk.Entry) and child_widget.master.winfo_children()[index-1].cget("text") == entry_label:
                        return child_widget.get()
        return None

    def parse_db_file(self, db_file_path, run_id, is1Dsweep=False, is2Dsweep=False):

        qc.initialise_or_create_database_at(db_file_path)

        assert isinstance(run_id, list), "Run ID is not given as a list, double check!"

        if len(run_id) == 1:

            dataset = qc.load_by_run_spec(captured_run_id=run_id[0])
            params = dataset.parameters.split(',')
            
            if is1Dsweep:
                xname, zname = params
                x = np.array(dataset.get_parameter_data(xname)[xname][xname])[0]
                z = np.array(dataset.get_parameter_data(zname)[zname][zname])
                return x, z

            elif is2Dsweep:
                xname, yname, zname = params
                x = np.array(dataset.get_parameter_data(xname)[xname][xname])
                y = np.array(dataset.get_parameter_data(yname)[yname][yname])
                z = np.array(dataset.get_parameter_data(zname)[zname][zname])
                return x,y,z
            
        elif len(run_id) == 2:
            # averaging needs to be done

            if is1Dsweep:
                run_id_sweep = [i for i in range(run_id[0], run_id[1]+1)]

                datasets = [qc.load_by_run_spec(captured_run_id=run) for run in run_id_sweep]
                
                params = datasets[0].parameters.split(',')

                xname, zname = params
                x = np.array([datasets[index].get_parameter_data(xname)[xname][xname] for index in range(len(datasets))])[0]
                z = np.array([datasets[index].get_parameter_data(zname)[zname][zname] for index in range(len(datasets))])
                
                z = np.mean(z, axis=0)

                return x, z

    def plot_current_and_spectrum(self, t, data, sampling_rate):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

        # Plot raw data
        for key, d in data.items():
            ax1.plot(t,d[1],label=d[2])
        ax1.set_ylabel(r'$I_{SD}\ (A)$')
        ax1.set_title(r'$I_{SD}\ (t)$')
        ax1.set_xlabel(r'$t$ (s)')
        
        # Plot noise spectrum
        for key, d in data.items():
            f, Pxx = scipy.signal.periodogram(d[1], fs=sampling_rate, scaling='density')
            ax2.loglog(f, Pxx, label=d[2])
        ax2.set_xlabel(r'$\omega$ (Hz)')
        ax2.set_ylabel(r'$S_I$ (A$^2$/Hz)')
        ax2.set_title(r'$S_I$')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.legend(loc='best')
        plt.show()
        
    def plot_lb_rb_scan(self, lb, rb, current):
        fig, ax = plt.subplots(1, sharex=False)

        # Plot raw data
        ax.imshow(current, extent=[rb[0],rb[-1], lb[0], lb[-1]])
        # ax1.set_extent()
        ax.set_ylabel(r'LB (mV)')
        ax.set_title(r'BB Scan')
        ax.set_xlabel(r'RB (mV)')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
