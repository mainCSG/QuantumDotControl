'''
File: gui.py
Authors: Mason Daub (mjdaub@uwaterloo.ca), Benjamin Van Osch (bvanosch@uwaterloo.ca)

This file contains the gui class that runs the auto tuner.
As of now, we are using the nicegui web server as the user interface for the auto tuner.
'''

# Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nicegui import ui, app

import threading
import time
from instrument_handler import create_buffer_instance
from experiment_handler import get_experiment_handler
from autotuning_handler import get_autotuning_handler
from qcodes.station import Station
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
import random
import os, sys
from tunerlog import TunerLog
from experiment_base import SweepParam, SweepLayer, Sweep
from autotuning_protocol import Protocol
from tunerlog import TunerLog
import yaml
from pathlib import Path

from paths import CONFIG_FOLDER, STATION_CONFIG
from instrument_registry import INITIALIZERS, MONITORS, init_agilent, init_spi_rack

logger = TunerLog('GUI')

class RandomDummy(DummyInstrument):
    '''
    A dummy instrument for testing the readout buffer
    '''
    def __init__(self,
        name: str = "dummy",
        gates = ("dac1",),
        **kwargs):

        super().__init__(name, gates, **kwargs)

        self.add_parameter("rand1",
                parameter_class=Parameter,
                initial_value=0,
                label=f"Gate rand",
                unit="V",
                set_cmd=None,
                get_cmd=lambda : random.Random(time.monotonic()).random())
        self.add_parameter("rand2",
                parameter_class=Parameter,
                initial_value=0,
                label=f"Gate rand",
                unit="V",
                set_cmd=None,
                get_cmd=lambda : random.Random(time.monotonic()).random())

class tuner_gui:
    
    # The below methods define the layout of the GUI

    def __init__(self):
        
        '''
        Creates an instance of the tuner gui
        '''
        self.logger = TunerLog("TunerGUI")
        self.start_time = time.monotonic()

        ## try to instantiate Station() from most recent settings, if fail then instantiate empty station
        try: 
            with open(CONFIG_FOLDER / 'config_gui.yaml') as f:
                settings = yaml.safe_load(f)

            last_config = settings['last_station_config']
            self.station = Station(config_file=str(CONFIG_FOLDER / last_config))
            print(f'Loading {CONFIG_FOLDER/ last_config} succesful')

        except Exception as e:
            print(f"Could not load previous Station: {e}")
            self.station = Station()

        self.station_lock = threading.Lock()

        self.instrument_handler = create_buffer_instance(self.station, self.station_lock) 
        self.experiment_handler = get_experiment_handler()
        self.autotuning_handler = get_autotuning_handler()

        self.abort_signal = threading.Event()

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
                    ui.item('Load Config Files', on_click=lambda : self.on_load_config())
                    ui.item('Instrument Information', on_click=lambda : self.on_load_instrument_info())
                    ui.item('Device Information', on_click=lambda : ui.notify("Loading Device Information..."))

                stages = ['Debug', 'Setup','Bootstrapping','Coarse Tuning','Virtual Gating','Charge State Tuning','Fine Tuning']

                with ui.tabs() as tabs:
                    
                    for stage in stages:
                        ui.tab(stage)

                with ui.tab_panels(tabs, value='Home').classes('w-full'):
            
                    # with ui.tab_panel('Setup'):

                    #     self.device_config = "Intel_Config.yaml"

                    #     self.autotune = ui.button(
                    #                         'Autotune',
                    #                         on_click = Protocol(device_config = self.device_config)
                    #                              )

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


                    with ui.tab_panel('Debug'):

                        ui.label('Debug / Manual Controls')

                        """ ui.button(
                            'Run Test Sweep',
                            on_click=self.run_test_sweep
                        ) """
                        
                        ui.button(
                            'Run Test Sweep 2',
                            on_click=self.run_test_sweep_2
                        )

                        """ ui.button(
                            'Run Test Sweep 3',
                            on_click=self.run_test_sweep_3
                        ) """

                        ui.button(
                            'Run Bootstrapping',
                            on_click = self.run_bootstrapping
                        )

                        ui.button(
                            'Run Global Charge Tuning',
                            on_click = self.run_global_charge_tuning
                        )

                        ui.button(
                            'Run Virtual Gating',
                            on_click = self.run_virtual_gating
                        )

                        self.debug_status = ui.label('Idle')

            with splitter1.after:

                with ui.splitter(horizontal = True) as splitter2:
                    
                    with splitter2.before:
                        
                        self.live_plot_window()

                    with splitter2.after:

                        self.ui_log = ui.log()
                        self.logger.add_ui_handler(self.ui_log)
                        self.logger.info("Added NiceGUI UI handler to logger.")

                ui.timer(0.025, self.update_liveplot)
                ui.timer(0.5, self.watchdog_timer)

    def run_test_sweep(self):

        self.debug_status.set_text("Running sweep...")
        self.logger.info("Sweep job queued")

        sweep = Sweep(
            layers=[
                SweepLayer(
                    targets=[
                        SweepParam('spi_rack.module1.dac0.voltage', 0.1, 0.0),
                        SweepParam('spi_rack.module1.dac1.voltage', 0.1, 0.0)
                    ],
                    num_points=20,
                    measurement_time=0.1
                )
            ],
            measure=lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ])
            )
        )

        future = self.experiment_handler.do_sweep(
            sweep=sweep,
            instrument_handler=self.instrument_handler,
            wait=False
        )

        def check_result():
            try:
                result = future.result(timeout=0)

            except TimeoutError:
                ui.timer(0.1, check_result, once=True)

            except Exception as e:
                self.debug_status.set_text(f"Error: {e}")

            else:
                self.debug_status.set_text(f"Sweep complete!")

        check_result()

    def run_test_sweep_2(self):

        self.debug_status.set_text("Running sweep...")
        self.logger.info("Sweep job queued")

        sweep = Sweep(
            layers=[
                SweepLayer(
                    targets=[
                        SweepParam('spi_rack.module2.dac13.voltage', 0.0, 0.3)
                    ],
                    num_points=20,
                    measurement_time=0.05
                ),
                SweepLayer(
                    targets=[
                        SweepParam('spi_rack.module2.dac15.voltage', 0.0, 0.3)
                    ],
                    num_points=20,
                    measurement_time=0.05
                ),
                SweepLayer(
                    targets=[
                        SweepParam('spi_rack.module2.dac14.voltage', 0.0, 0.3)
                    ],
                    num_points=20,
                    measurement_time=0.05
                )
            ],
            measure=lambda ih, sp: (
                ih.read_buffer([
                    'agilent_left.volt',
                    'agilent_right.volt'
                ]),
                ['agilent_left.volt', 'agilent_right.volt']
            )
        )

        future = self.experiment_handler.do_sweep(
            sweep=sweep,
            instrument_handler=self.instrument_handler,
            wait=False
        )

        def check_result():
            try:
                result = future.result(timeout=0)

            except TimeoutError:
                ui.timer(0.1, check_result, once=True)

            except Exception as e:
                self.debug_status.set_text(f"Error: {e}")

            else:
                self.debug_status.set_text(f"Sweep complete!")

        check_result()

    def run_test_sweep_3(self):

        self.debug_status.set_text("Running sweep...")
        self.logger.info("Sweep job queued")

        sweep = Sweep(
            layers=[
                SweepLayer(
                    targets=[
                        SweepParam('spi_rack.module1.dac0.voltage', 0.0, 0.5)
                    ],
                    num_points=20,
                    measurement_time=0.05
                ),
                SweepLayer(
                    targets=[
                        SweepParam('spi_rack.module1.dac1.voltage', 0.0, 0.3)
                    ],
                    num_points=20,
                    measurement_time=0.05
                )
            ],
            measure=lambda ih, sp: ih.read_buffer(
                ['agilent_left.volt',
                'agilent_right.volt'
            ])
        )

        future = self.experiment_handler.set_voltage_configuration(
            sweep=sweep,
            instrument_handler=self.instrument_handler,
            wait=False
        )

        def check_result():
            try:
                result = future.result(timeout=0)

            except TimeoutError:
                ui.timer(0.1, check_result, once=True)

            except Exception as e:
                self.debug_status.set_text(f"Error: {e}")

            else:
                self.debug_status.set_text(f"Sweep complete!")

        check_result()

    def run_bootstrapping(self):

        self.debug_status.set_text("Running Bootstrapping...")
        self.logger.info("Bootstrapping Jobs queued")

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_bootstrapping(device_config = device_config,
                                                           instrument_handler = self.instrument_handler,
                                                           experiment_handler = self.experiment_handler,
                                                           wait = False
                                                          )

    def run_global_charge_tuning(self):

        self.debug_status.set_text("Running Global Charge Tuning...")
        self.logger.info("Global Charge Tuning Jobs queued")

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_global_charge_tuning(device_config = device_config,
                                                                  instrument_handler = self.instrument_handler,
                                                                  experiment_handler = self.experiment_handler,
                                                                  wait = False
                                                                 )

    def run_virtual_gating(self):

        self.debug_status.set_text("Running Virtual Gating...")
        self.logger.info("Virtual Gating Jobs queued")

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_virtual_gating(device_config = device_config,
                                                                  instrument_handler = self.instrument_handler,
                                                                  experiment_handler = self.experiment_handler,
                                                                  wait = False
                                                                 )

    def header(self):
        
        """
        The method that defines the header of the gui.

        params: 
            self:
        """

        with ui.header().classes(replace='row items-center') as header:
            ui.label('Welcome to the QAT!!!')

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

        self.liveplot = ui.matplotlib(figsize = (8,6))

        fig = self.liveplot.figure
        self.ax = fig.subplots(1,1)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel('Signal (V)')
        xs = np.linspace(-1, 1)
        self.lines = self.ax.plot(xs, np.sin(xs))
        self.ax.set_xlabel('time (s)', fontsize = 16)
        self.ax.set_ylabel('Signal (V)', fontsize = 16)
        self.ax.tick_params(labelsize = 12)
        fig.tight_layout()
        self.liveplot.update()

    def update_liveplot(self): 
        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green']
        linestyles = ['-', '--', '-.', ':']

        #TODO: uncomment the below line when the instrument handler is implemented
        retval = self.instrument_handler.get_buffer()
        # retval = None
        if retval is None:
            return
        else:
            keys =  list(retval.keys())
            num_keys = len(keys)
            if num_keys != len(self.lines):
                # Clear the axis.
                for line in self.lines:
                    line.remove()
                # add the lines back in
                for j in range(num_keys):
                    col = colors[j % len(colors)]
                    ls = linestyles[j // len(colors)]
                    self.ax.add_line(Line2D([0],[0], lw = 2, color = col, linestyle=ls))
                self.lines = self.ax.get_lines()

            for j in range(num_keys):
                key = keys[j]
                data = retval[key]
                data = [retval[key][i][0] for i in range(len(retval[key]))]
                times = [retval[key][i][1] for i in range(len(retval[key]))]

                times_offset = np.array(times) - times[-1]
                self.lines[j].set_ydata(data)
                self.lines[j].set_xdata(times_offset)
                self.ax.set_xlim(min(times_offset), max(times_offset))
                self.ax.set_ylim(-0.1, 1.4)

            self.ax.legend(self.lines, keys, )
            self.liveplot.update()

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

    def watchdog_timer(self):
        if not self.instrument_handler.watchdog():
            logger.info("Watchdog Failed!")
            # Trigger a reset
            self.logger.error("Readout buffer watchdog detected a problem. Triggering a reset.")
            #python = sys.executable  # path to the Python interpreter
            #os.execv(python, [python] + sys.argv)

    def on_abort(self):
        ui.notify('Aborting...')
        self.logger.warning("Abort requested: draining all worker queues")
        self.autotuning_handler.autotuning_thread.abort()
        self.experiment_handler.experiment_thread.abort()
        self.instrument_handler.abort_instruments()

    def on_shutdown(self):
        self.abort_signal.set() # Abort any currently running experiments.
        self.autotuning_handler.autotuning_thread.join()
        self.experiment_handler.experiment_thread.join()
        self.instrument_handler.shutdown_instruments()

    async def on_load_config(self) -> None:
        """
        This method is used to load the config files for the device and update the Station object.
        It uses the local_file_picker script, obtained from the examples from the nicegui library on github. 
        Source: 
        """
        ui.notify("Loading Config Files...")
        self.device_config = await local_file_picker(directory = STATION_CONFIG, upper_limit = STATION_CONFIG)
        self.station.load_config_files(*self.device_config)
        # logger.info('printing loaded config')
        
        configured = set(self.station.config['instruments'])
        connected = set(self.instrument_handler.instrument_threads.keys())

        # remove elements that were connected but should not be
        for name in connected - configured:
            self.instrument_handler.remove_instrument(name)

        # add components
        for name in configured - connected:
            self.instrument_handler.add_instrument(name, INITIALIZERS.get(name))

            if name in MONITORS:
                self.instrument_handler.monitor_parameter(name, MONITORS.get(name))

    def on_save_config(self):
        """
        This method is used to save the config files for the device.
        """
        pass

    def on_load_instrument_info(self):
        """
        This method is used to load and display the device information. 
        """
        ui.notify("Loading Instrument Information...")
        instrument_manager = InstrumentManager

    # The below classes

class local_file_picker(ui.dialog):

    def __init__(self, directory: str, *,
                 upper_limit: str | None = ...,) -> None:
        """Local File Picker

        This is a simple file picker that allows you to select a file from the local filesystem where NiceGUI is running.

        :param directory: The directory to start in.
        :param upper_limit: The directory to stop at (None: no limit, default: same as the starting directory).
        :param multiple: Whether to allow multiple files to be selected.
        :param show_hidden_files: Whether to show hidden files.
        """
        super().__init__()

        self.directory = directory
        with self, ui.card().classes('q-pa-md q-ma-sm bg-primary text-white'):
            ui.label(f"Select a file from {self.directory}")
            self.grid = ui.aggrid({
                'columnDefs': [
                    {'headerName': 'File', 'field': 'file'}
                ],
                'rowData': [
                    {'file': f.name} for f in Path(self.directory).iterdir()
                    ],
                'rowSelection': {'mode': 'multiRow'}
            })
            with ui.row():
                self.cancel_button = ui.button("Cancel", on_click=self.close)
                self.select_button = ui.button("Select", on_click=self._select)
        self.update_grid()

    def update_grid(self):
        self.grid.update()

    async def _select(self):
        selected_rows = await self.grid.get_selected_rows()
        self.submit([str(STATION_CONFIG / row['file']) for row in selected_rows])

class InstrumentManager(ui.dialog):
    def __init__(self, instrument_handler):
        super().__init__()
        self.instrument_handler = instrument_handler
        configured = set(self.instrument_handler.config['instruments'])
        connected = set(self.instrument_handler.instrument_threads.keys())
        with self, ui.card().classes('q-pa-md q-ma-sm bg-primary text-white'):
            ui.label('Instrument Manager')
            self.grid = ui.aggrid({
                'columnsDef': [
                    {'headerName': 'Instrument', 'field': 'name'},
                    {'headerName': 'Status', 'field': 'status'},
                    {'headerName': 'Action', 'field': 'action'}
                ],
                'rowData':[{'name': name, 'status': 'connected', 'action': 'None'} for name in configured - connected]
                })
            
            ui.button('Close', on_click=self.close)
        self.update_grid()
        self.open()

    def update_grid(self):
        self.grid.update()