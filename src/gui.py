'''
File: gui.py
Authors: Mason Daub (mjdaub@uwaterloo.ca), Benjamin Van Osch (bvanosch@uwaterloo.ca)

This file contains the gui class that runs the auto tuner.
As of now, we are using the nicegui web server as the user interface for the auto tuner.
'''

# Imports

import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui, app
import os
import threading
import time
from buffered_readout import create_buffer_instance
import time
from experiment_thread import ExperimentThread


class tuner_gui:
    
    # The below methods define the layout of the GUI

    def __init__(self):
        
        '''
        Creates an instance of the tuner gui

        params: 
            self:
        '''
        global _FirstPass
        
        # print(threading.current_thread().name)
        global _gui_instances
        _gui_instances.append(self)

        self.start_time = 0

        self.readout = create_buffer_instance()
        self.readout.run()

    # The below methods define the layout of the GUI

    def root_page(self):

        """
        The method that intialises the gui. As of now, it also defines the main page of the within itself.

        params: 
            self:
        """

        self.header()
        self.footer()

        # I tried putting these splitters into a separate function, but then the gui wouldn't start. 

        with ui.splitter(value = 54, limits = (54,54)) as splitter1:
            
            with splitter1.before:

                with ui.dropdown_button('', icon = 'menu', auto_close=True):
                    ui.item('Load Config Files', on_click=lambda : ui.notify("Fetching Configuration Files..."))
                    ui.item('Instrument Information', on_click=lambda : ui.notify("Loading Instrument Information..."))
                    ui.item('Device Information', on_click=lambda : ui.notify("Loading Device Information..."))

                stages = ['Setup','Bootstrapping','Coarse Tuning','Virtual Gating','Charge State Tuning','Fine Tuning']

                with ui.tabs() as tabs:
                    
                    for stage in stages:
                        ui.tab(stage)

                with ui.tab_panels(tabs, value='Home').classes('w-full'):
            
                    with ui.tab_panel('Setup'):

                        self.results_plot_panel()

                        self.instr = ui.button('Connect to instruments', on_click = self.experiment_progress_bar)
                        self.autotune = ui.button('Autotune')
                        
                        config_files = os.listdir('../configs')
                        config_dict = {i : config_files[i] for i in range(len(config_files))}

                    with ui.tab_panel('Bootstrapping'):

                        ui.label('Collecting Bootstrapping Information...')

                    
                    with ui.tab_panel('Coarse Tuning'):
                        
                        ui.label('Collecting Coarse Tuning Information...')   


                    with ui.tab_panel('Virtual Gating'):
                        
                        ui.label('Collecting Virtual Gating Information...')


                    with ui.tab_panel('Charge State Tuning'):
                        
                        ui.label('Collecting Charge State Tuning...')


                    with ui.tab_panel('Fine Tuning'):
                        
                        ui.label('Collecting Fine Tuning Information...')


            with splitter1.after:

                with ui.splitter(horizontal = True) as splitter2:
                    
                    with splitter2.before:
                        
                        self.live_plot_window()

                    with splitter2.after:

                        ui.label('Logger Information').classes('ml-2')


                ui.timer(0.05, self.update_liveplot)
                ui.timer(0.25, self.update_experiment_progress_bar)

    def header(self):
        
        """
        The method that defines the header of the gui.

        params: 
            self:
        """

        with ui.header().classes(replace='row items-center') as header:
            ui.label('Welcome to the Quantum Spin Qubit Device Autotuner!!!')

    def footer(self):
        
        """
        The method that intialises the footer of the gui.

        params: 
            self:
        """

        with ui.footer(value=True) as footer:
            ui.button('ABORT', on_click = self.on_abort, color='red')

    # The below methods define all features and general functions of the GUI

    def results_plot_panel(self):

        """
        The method that defines the results plots. This method takes the output plots from data_analysis 
        and displays them in its corresponding autotuning stage tab.

        params:
            self:
            results:


        """

        with ui.matplotlib().figure as fig:
            #fig = plt.gcf()
            axs = fig.subplots(1, 2)
            xs = np.linspace(-np.pi, np.pi, 100)
            axs[0].plot(xs, np.sin(xs))
            axs[1].plot(xs, np.cos(xs))
            fig.tight_layout()

    def live_plot_window(self):

        """
        The method that defines the live plot window, which streams the measurement of our readout instrument.

        params:
            self:
        
        """

        self.liveplot = ui.matplotlib(figsize = (30,20))

        fig = self.liveplot.figure
        self.ax = fig.subplots(1,1)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel('Signal (V)')
        xs = np.linspace(-1, 1)
        self.line = self.ax.plot(xs, np.sin(xs))
        self.ax.set_xlabel('time (s)', fontsize = 75)
        self.ax.set_ylabel('Current (A)', fontsize = 75)
        self.ax.tick_params(labelsize = 50)
        self.liveplot.update()

    def update_liveplot(self): 
        retval = self.readout.get_buffer()
        if retval is None:
            return
        else:
            data, times = retval

            times_offset = np.array(times) - self.start_time
            self.line[0].set_ydata(data)
            self.line[0].set_xdata(times_offset)
            self.ax.set_xlim(min(times_offset), max(times_offset))
            self.ax.set_ylim(-0.5, 1.5)

            self.liveplot.update()
            ui.update()

    def experiment_progress_bar(self):

        self.pb = ui.linear_progress(show_value = False)
        self.instr.disable()

    def update_experiment_progress_bar(self):

        if self.pb.value < 1.02:
            self.pb.value += 0.01

        if round(self.pb.value, 1) == 1.0:
            
            self.pb.delete()
            self.instr.enable()

    def split_view(self, page1, page2, horizontal_split: bool = False):
        
        """
        This method creates a split view of two specified pages. 

        params:
            self:
            page1: The first page of the split. Depending on if horizontal_split is True or False, 
                   this page will be on the top, or the left, respectively.
            page2: The second page of the split. Depending on if horizontal_split is True or False, 
                   this page will be on the bottom, or the right, respectively.
            horizontal_split: Determines whether the split creates a left/right splitting, or a top/bottom splitting. True implies
                              the split is horizontal, meaning it will be top/bottom. False means a vertical split, or left/right splitting.
        """

        with ui.splitter(horizontal = horizontal_split) as splitter:
            with splitter.after:
                page2()
            with splitter.before:
                page1()

    # The below methods define all the button press logic of the GUI

    def on_connect(self):
        
        """
        This method will connect to the load_config_files method in write control and XXX in buffered readout, to initialise connections 
        to the instruments for setting voltages, and the buffered readout to start capturing data on the live plotting window.

        params:
            self:
        """
        
        pass

    def on_autotune(self):
        
        """
        This method starts the defined autotuning protocol. Currently, the autotuning protocol is specific to the Intel Tunnel Falls
        devices, by following the autotuning_protocol.py file. This will be updated to allow for application to general devices.


        params:
            self:

        """
        
        pass

    def on_abort(self):
        ui.notify('Aborting...')
    
    def on_shutdown(self):
        
        self.readout.join()


