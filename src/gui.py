'''
File: gui.py
Authors: Mason Daub (mjdaub@uwaterloo.ca), Benjamin Van Osch (bvanosch@uwaterloo.ca)

This file contains the gui class that runs the auto tuner.
As of now, we are using the nicegui web server as the user interface for the auto tuner.
'''

# Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from nicegui import ui, app
import plotly.graph_objects as go
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
    Description
    -----------
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

    '''
    Description
    -----------
    The below methods define the layout of the GUI
    '''

    def __init__(self, bridge):
        
        '''
        Description
        -----------
        Creates an instance of the tuner gui. Initializes the setup with the station config.
        '''

        # Store the shared event bridge reference
        self.bridge = bridge
        
        # Dictionary to store references to scroll areas for each stage tab
        self.tab_containers = {}

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

    def root_page(self):

        """
        Description
        -----------
        The method that intializes the gui's main page. As of now, it also defines the main page within itself.
        Contains tabs for each stage of the auto tuning process. Each tab contains a scrollable area that will be populated with plots and metrics as the auto tuner runs.
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
                        self.tab_containers['Bootstrapping'] = ui.scroll_area().classes('w-full h-[60vh] border p-2')

                    
                    with ui.tab_panel('Coarse Tuning'):
                        
                        ui.label('Collecting Coarse Tuning Information...')
                        self.tab_containers['Coarse Tuning'] = ui.scroll_area().classes('w-full h-[60vh] border p-2')


                    with ui.tab_panel('Virtual Gating'):
                        
                        ui.label('Collecting Virtual Gating Information...')
                        self.tab_containers['Virtual Gating'] = ui.scroll_area().classes('w-full h-[60vh] border p-2')


                    with ui.tab_panel('Charge State Tuning'):
                        
                        ui.label('Collecting Charge State Tuning...')
                        self.tab_containers['Charge State Tuning'] = ui.scroll_area().classes('w-full h-[60vh] border p-2')


                    with ui.tab_panel('Fine Tuning'):
                        
                        ui.label('Collecting Fine Tuning Information...')
                        self.tab_containers['Fine Tuning'] = ui.scroll_area().classes('w-full h-[60vh] border p-2')


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

                        ui.button(
                            'Take Snapshot',
                            on_click = self.run_snapshot
                        )

                        # ui.button(
                        #     'Run Plot Test',
                        #     on_click = run_test_plot
                        # )

                        self.debug_status = ui.label('Idle')

            with splitter1.after:

                with ui.splitter(horizontal = True) as splitter2:
                    
                    with splitter2.before:
                        
                        self.live_plot_window()

                    with splitter2.after:

                        self.ui_log = ui.log()
                        self.logger.add_ui_handler(self.ui_log)
                        self.logger.info("Added NiceGUI UI handler to logger.")

                ui.timer(0.025, self.update_liveplot) # updates the liveplot every 25ms
                ui.timer(0.5, self.watchdog_timer) # updates the watchdog timer every 500ms

                ui.timer(0.2, self.check_for_tuning_plots) # checks the plot queue for new plots every 200ms

    # Height of every tuning plot, in CSS pixels. The element is given a definite
    # height so the chart never has to infer its own size from its contents.
    PLOT_HEIGHT_PX = 320

    @staticmethod
    def figure_to_plot_data(fig) -> dict:

        """
        Description
        -----------
        Pull the plotted data back out of a Matplotlib figure.

        Lets a caller build a figure as usual and still get an interactive chart:
        the numbers are extracted here and redrawn by Plotly. Rendering the
        Matplotlib figure itself would only ever give a static image, which cannot
        support zooming or hover readout.

        Handles 1D line plots (including several lines on one axes) and 2D maps from
        either ``imshow`` or ``pcolormesh``. When a 2D map also carries line
        overlays, such as diamond outlines drawn on a scan, the lines are kept and
        drawn on top of the heatmap.

        ``contourf`` is not supported: it keeps only the contour polygons, not the
        grid they came from, so the original array cannot be recovered.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to extract data from.
        
        Returns
        -------
        plot_data : dict
            A dict with keys ``kind``, ``x_label``, ``y_label``, ``title`` and either
            ``series`` (for 1D line plots) or ``z``, ``x``, ``y`` and optionally
            ``overlays`` (for 2D heatmaps). The dict can be passed to
            ``build_plotly_spec()`` to get a Plotly spec for the same chart.
        """

        if not fig.axes:
            raise ValueError("Matplotlib figure has no axes to extract.")
        ax = fig.axes[0]

        common = {
            'x_label': ax.get_xlabel() or 'x',
            'y_label': ax.get_ylabel() or 'y',
            'title': ax.get_title() or '',
        }

        def _line_series():
            """
            Description
            -----------
            Every drawn line on the axes, as plot-data series.

            Returns
            -------
            series : list[dict]
                Each dict has keys ``x``, ``y``, ``name``, ``mode`` and optionally ``color``.
            """
            series = []
            for line in ax.lines:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if xdata.size == 0:
                    continue
                label = line.get_label()
                colour = None
                try:
                    colour = mcolors.to_hex(line.get_color())
                except Exception:
                    pass
                series.append({
                    'x': xdata,
                    'y': ydata,
                    'name': label if label and not label.startswith('_') else 'Series',
                    'mode': ('lines+markers'
                             if line.get_marker() not in (None, 'None', '') else 'lines'),
                    'color': colour,
                })
            return series

        # 2D first, so a map with outlines drawn over it is not mistaken for a line plot
        if ax.images:
            image = ax.images[0]
            z = np.asarray(image.get_array(), dtype=float)
            left, right, bottom, top = image.get_extent()
            # extent gives the outer edges, so sample at pixel centres
            dx = (right - left) / z.shape[1]
            dy = (top - bottom) / z.shape[0]
            return {
                'kind': 'heatmap',
                'z': z,
                'x': left + dx * (np.arange(z.shape[1]) + 0.5),
                'y': bottom + dy * (np.arange(z.shape[0]) + 0.5),
                'z_label': 'z',
                'overlays': _line_series(),
                **common,
            }

        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        if meshes:
            mesh = meshes[0]
            z = np.ma.filled(np.asarray(mesh.get_array(), dtype=float), np.nan)
            # get_coordinates returns the cell corners, shaped (ny + 1, nx + 1, 2),
            # so the cell centres are the midpoints between neighbouring corners
            corners = mesh.get_coordinates()
            if z.ndim == 1:
                z = z.reshape(corners.shape[0] - 1, corners.shape[1] - 1)
            x_centres = 0.5 * (corners[0, :-1, 0] + corners[0, 1:, 0])
            y_centres = 0.5 * (corners[:-1, 0, 1] + corners[1:, 0, 1])
            return {
                'kind': 'heatmap',
                'z': z,
                'x': np.asarray(x_centres, dtype=float),
                'y': np.asarray(y_centres, dtype=float),
                'z_label': 'z',
                'overlays': _line_series(),
                **common,
            }

        series = _line_series()
        if series:
            return {'kind': 'line', 'series': series, **common}

        if ax.collections:
            raise ValueError(
                "Matplotlib figure has no lines, image or mesh that can be read back. "
                "It holds only generic collections, which is what contourf produces; "
                "contourf keeps the contour polygons rather than the grid, so the data "
                "cannot be recovered. Use imshow or pcolormesh, or pass the arrays in "
                "'plot_data' instead of a figure.")

        raise ValueError("Matplotlib figure contains no lines, image or mesh to extract.")

    @classmethod
    def build_plotly_spec(cls, plot_data: dict) -> dict:

        """
        Description
        -----------
        Build a Plotly spec dict with data, layout and interaction config.

        The dict form is used rather than a ``go.Figure`` because only the dict
        carries a ``config`` entry, which is where the zoom and double-click
        behaviour is set.

        Parameters
        ----------
        plot_data : dict
            A dict with keys ``kind``, ``x_label``, ``y_label``, ``title`` and either
            ``series`` (for 1D line plots) or ``z``, ``x``, ``y`` and optionally
            ``overlays`` (for 2D heatmaps). The dict can be produced by
            ``figure_to_plot_data()`` or constructed directly.

        Returns
        -------
        spec : dict
            A Plotly spec dict with keys ``data``, ``layout`` and ``config``.
        """
        kind = plot_data.get('kind', 'line')
        title = plot_data.get('title', '')

        if kind in {'line', 'scatter'}:
            # accept either a single x/y pair or a list of series
            if 'series' in plot_data:
                series = plot_data['series']
            else:
                series = [{
                    'x': plot_data.get('x', []),
                    'y': plot_data.get('y', []),
                    'name': plot_data.get('name', 'Series'),
                    'mode': 'lines',
                }]

            traces, all_x, all_y = [], [], []
            for entry in series:
                x = np.asarray(entry.get('x', []), dtype=float)
                y = np.asarray(entry.get('y', []), dtype=float)
                if x.size == 0:
                    continue
                all_x.append(x)
                all_y.append(y)
                traces.append(go.Scatter(
                    x=x, y=y,
                    mode=entry.get('mode', 'lines'),
                    name=entry.get('name', 'Series'),
                    line={'color': entry['color']} if entry.get('color') else None,
                    # readable point readout on hover
                    hovertemplate=(f"{plot_data.get('x_label', 'x')}: %{{x:.6g}}<br>"
                                   f"{plot_data.get('y_label', 'y')}: %{{y:.6g}}"
                                   "<extra>%{fullData.name}</extra>"),
                ))

            if not traces:
                raise ValueError("No data to plot.")

            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            x_range = [float(np.nanmin(x_all)), float(np.nanmax(x_all))]
            y_range = [float(np.nanmin(y_all)), float(np.nanmax(y_all))]
            fig = go.Figure(data=traces)

        elif kind in {'heatmap', 'image', '2d'}:
            z = np.asarray(plot_data['z'], dtype=float)
            x = np.asarray(plot_data.get('x', np.arange(z.shape[1])), dtype=float)
            y = np.asarray(plot_data.get('y', np.arange(z.shape[0])), dtype=float)
            fig = go.Figure(data=[go.Heatmap(
                z=z, x=x, y=y, colorscale='Viridis', hoverongaps=False,
                colorbar={'title': {'text': plot_data.get('z_label', 'z')}},
                hovertemplate=(f"{plot_data.get('x_label', 'x')}: %{{x:.6g}}<br>"
                               f"{plot_data.get('y_label', 'y')}: %{{y:.6g}}<br>"
                               f"{plot_data.get('z_label', 'z')}: %{{z:.6g}}"
                               "<extra></extra>"),
            )])
            x_range = [float(np.nanmin(x)), float(np.nanmax(x))]
            y_range = [float(np.nanmin(y)), float(np.nanmax(y))]

            # Lines drawn on top of the map, such as fitted diamond outlines. They
            # do not widen the axes: the map already sets the extent.
            for overlay in plot_data.get('overlays') or []:
                ox = np.asarray(overlay.get('x', []), dtype=float)
                oy = np.asarray(overlay.get('y', []), dtype=float)
                if ox.size == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=ox, y=oy,
                    mode=overlay.get('mode', 'lines'),
                    name=overlay.get('name', 'Overlay'),
                    line={'color': overlay.get('color') or 'red', 'width': 1.5},
                    hovertemplate=(f"{plot_data.get('x_label', 'x')}: %{{x:.6g}}<br>"
                                   f"{plot_data.get('y_label', 'y')}: %{{y:.6g}}"
                                   "<extra>%{fullData.name}</extra>"),
                ))

        else:
            raise ValueError(f"Unsupported plot_data kind: {kind}")

        # Pin both axes to the data range. autorange=False stops the view drifting,
        # and because the range is stated explicitly a double-click returns exactly
        # here rather than to a recomputed range.
        fig.update_xaxes(title_text=plot_data.get('x_label', 'x'),
                         range=x_range, autorange=False)
        fig.update_yaxes(title_text=plot_data.get('y_label', 'y'),
                         range=y_range, autorange=False)

        fig.update_layout(
            title=title or None,
            # A definite height and automatic width. Without autosize the chart
            # would keep its own idea of its width and could not fit the panel.
            autosize=True,
            height=cls.PLOT_HEIGHT_PX,
            margin={'l': 60, 'r': 20, 't': 40 if title else 20, 'b': 50},
            dragmode='zoom',
            hovermode='closest',
            showlegend=len(fig.data) > 1,
            # uirevision keeps the user's zoom across any re-render of the element,
            # and stops a re-render nudging the axis ranges
            uirevision='tuning-plot',
        )

        spec = fig.to_plotly_json()
        spec['config'] = {
            'responsive': True,
            # double-click returns to the ranges set above, which is the undo-zoom
            'doubleClick': 'reset',
            'scrollZoom': True,
            'displaylogo': False,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d'],
        }
        return spec

    @classmethod
    def build_plotly_figure(cls, plot_data: dict):

        """
        Description
        -----------
        Backwards-compatible alias returning the spec dict.
        
        Parameters
        ----------
        plot_data : dict
            A dict with keys ``kind``, ``x_label``, ``y_label``, ``title`` and either
            ``series`` (for 1D line plots) or ``z``, ``x``, ``y`` and optionally
            ``overlays`` (for 2D heatmaps). The dict can be produced by
            ``figure_to_plot_data()`` or constructed directly.
        
        Returns
        -------
        spec : dict
            A Plotly spec dict with keys ``data``, ``layout`` and ``config``.
        """

        return cls.build_plotly_spec(plot_data)

    def check_for_tuning_plots(self):

        """
        Description
        -----------
        Periodically polls the thread-safe bridge queue. 
        Creates a collapsible expansion panel, renders the figure, and extracts tuning metrics to display directly below the chart.
        """

        while not self.bridge.plot_queue.empty():
            try:
                payload = self.bridge.plot_queue.get_nowait()
                
                stage_name = payload.get('stage')
                step_name = payload.get('step_name', 'Process')
                fig = payload.get('figure_object')
                plot_data = payload.get('plot_data')
                results_data = payload.get('results', {})
                
                if stage_name in self.tab_containers:
                    target_container = self.tab_containers[stage_name]

                    with target_container:
                        # Build the container for this specific step at the top of the tab
                        with ui.expansion(text=step_name, icon='analytics', value=True).classes('w-full min-w-0 border rounded-lg bg-slate-50 my-2 shadow-sm') as expansion_panel:

                            # A chart sized by its container, inside a container sized
                            # by its contents, feeds back on itself and the plot creeps
                            # wider on every resize. `min-w-0` lets this wrapper shrink
                            # below its contents (flex children default to min-width
                            # auto, which is what blocks that), and the fixed height
                            # plus overflow-hidden give the chart a definite box to
                            # measure. This is what stops the axis drifting rightwards.
                            plot_source = plot_data
                            if plot_source is None and fig is not None:
                                try:
                                    plot_source = self.figure_to_plot_data(fig)
                                except Exception as exc:
                                    self.logger.warning(
                                        f"Could not extract data from the Matplotlib "
                                        f"figure for '{step_name}', falling back to a "
                                        f"static image: {exc}")

                            if plot_source is not None:
                                with ui.element('div').classes('w-full min-w-0 overflow-hidden p-2') \
                                        .style(f'height: {self.PLOT_HEIGHT_PX + 16}px'):
                                    ui.plotly(self.build_plotly_spec(plot_source)) \
                                        .classes('w-full min-w-0') \
                                        .style(f'height: {self.PLOT_HEIGHT_PX}px')
                            elif fig is not None:
                                # Static image only: a Matplotlib render cannot zoom
                                # or report values on hover.
                                static_plot = ui.pyplot(close=False)
                                static_plot.fig = fig
                                static_plot._convert_to_html()
                                static_plot.classes('w-full min-w-0 p-2')
                            
                            # Add an explicit Results section below the plot if data exists
                            if results_data:
                                ui.separator().classes('my-2 px-4')
                                with ui.column().classes('w-full px-6 pb-4'):
                                    ui.label('Step Results & Parameter Updates:').classes('text-sm font-bold text-slate-700 mb-1')
                                    
                                    # Use a responsive grid layout to organize your numbers cleanly
                                    with ui.grid(columns=2).classes('w-full gap-x-8 gap-y-1 bg-white p-3 border rounded'):
                                        for key, value in results_data.items():
                                            ui.label(f"{key}:").classes('text-xs font-semibold text-slate-500')
                                            ui.label(str(value)).classes('text-xs font-mono text-slate-900 text-right')
                        
                else:
                    self.logger.warning(f"Received plot for unknown stage: {stage_name}")
                    
                self.bridge.plot_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error drawing tuning plot and results: {e}")

    def run_test_sweep(self):

        '''
        Description
        -----------
        A test sweep that sweeps two DACs and measures two Agilent voltages.
        This is a simple example of how to use the Sweep class and the ExperimentHandler to run a sweep.
        '''

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

        '''
        Description
        -----------
        A test sweep that sweeps three DACs and measures two Agilent voltages.
        This is a simple example of how to use the Sweep class and the ExperimentHandler to run a sweep.
        '''

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

        '''
        Description
        -----------
        A test sweep that sweeps three DACs and measures two Agilent voltages.
        This is a simple example of how to use the Sweep class and the ExperimentHandler to run a sweep.
        '''

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

        '''
        Description
        -----------
        A method that runs the bootstrapping stage of the auto tuning process.
        '''

        self.debug_status.set_text("Running Bootstrapping...")
        self.logger.info("Bootstrapping Jobs queued")

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_bootstrapping(device_config = device_config,
                                                           instrument_handler = self.instrument_handler,
                                                           experiment_handler = self.experiment_handler,
                                                           wait = False
                                                          )

    def run_global_charge_tuning(self):

        '''
        Description
        -----------
        A method that runs the global charge tuning stage of the auto tuning process.
        '''

        self.debug_status.set_text("Running Global Charge Tuning...")
        self.logger.info("Global Charge Tuning Jobs queued")

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_global_charge_tuning(device_config = device_config,
                                                                  instrument_handler = self.instrument_handler,
                                                                  experiment_handler = self.experiment_handler,
                                                                  wait = False
                                                                 )

    def run_virtual_gating(self):

        '''
        Description
        -----------
        A method that runs the virtual gating stage of the auto tuning process.
        '''

        self.debug_status.set_text("Running Virtual Gating...")
        self.logger.info("Virtual Gating Jobs queued")

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_virtual_gating(device_config = device_config,
                                                                  instrument_handler = self.instrument_handler,
                                                                  experiment_handler = self.experiment_handler,
                                                                  wait = False
                                                                 )

    def run_snapshot(self):

        device_config = os.path.join("configs", "Intel_Config.yaml")

        future = self.autotuning_handler.run_snapshot(device_config = device_config,
                                                      instrument_handler = self.instrument_handler,
                                                      experiment_handler = self.experiment_handler,
                                                      wait = False
                                                     )

    def header(self):
        
        """
        Description
        -----------
        The method that defines the header of the gui.
        """

        # Uses bg-primary to match the default NiceGUI footer color
        with ui.header().classes('bg-primary text-white').style('height: 2px;'):
            ui.label(' ').classes('opacity-0')

    def footer(self):
        
        """
        Description
        -----------
        The method that intialises the footer of the gui.
        """

        with ui.footer(value=True).classes('items-center'):
            with ui.row().classes('w-full items-center justify-between'):
                # left: abort button
                ui.button('ABORT', on_click=self.on_abort, color='red').classes('mx-4')
                # center: large centered welcome text using a Unicode ket
                ui.label('Welcome to |QAT⟩').classes('text-3xl font-bold text-center flex-1')
                # right: spacer to balance the layout
                ui.label('').style('width:64px')

    # The below methods define all features and general functions of the GUI

    def results_plot_panel(self):

        """
        Description
        -----------
        The method that defines the results plots. This method takes the output plots from data_analysis 
        and displays them in its corresponding autotuning stage tab.
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
        Description
        -----------
        The method that defines the live plot window, which streams the measurement of our readout instrument.
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

        '''
        Description
        -----------
        This method updates the live plot window with the latest data from the readout instrument.
        '''
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

            all_times = []
            for j in range(num_keys):
                key = keys[j]
                values = retval[key]
                data = [v[0] for v in values]
                times = np.asarray([v[1] for v in values], dtype=float)
                all_times.extend(times.tolist())

                if len(times) > 1:
                    times_offset = times - times[-1]
                    self.lines[j].set_ydata(data)
                    self.lines[j].set_xdata(times_offset)
                else:
                    self.lines[j].set_ydata(data)
                    self.lines[j].set_xdata(np.array([0.0]))

            if all_times:
                newest_time = max(all_times)
                window_seconds = 10.0
                self.ax.set_xlim(-window_seconds, 0.5)
                self.ax.set_ylim(-0.1, 1.4)

            self.ax.legend(self.lines, keys, )
            self.liveplot.update()

    def experiment_progress_bar(self):

        '''
        Description
        -----------
        This method creates a progress bar for the experiment, which is updated by the update_experiment_progress_bar method.
        '''

        self.pb = ui.linear_progress(show_value = False)
        self.instr.disable()

    def update_experiment_progress_bar(self):

        '''
        Description
        -----------
        This method updates the experiment progress bar.
        '''

        if self.pb.value < 1.02:
            self.pb.value += 0.01

        if round(self.pb.value, 1) == 1.0:
            
            self.pb.delete()
            self.instr.enable()

    def split_view(self, page1, page2, horizontal_split: bool = False):
        
        """
        Description
        -----------
        This method creates a split view of two specified pages. 

        Parameters
        ----------
        page1 : object containing the page
            The first page of the split. Depending on if horizontal_split is True or False, 
            this page will be on the top, or the left, respectively.
        page2 : object containing the page
            The second page of the split. Depending on if horizontal_split is True or False, 
            this page will be on the bottom, or the right, respectively.
        horizontal_split : bool, optional
            Determines whether the split creates a left/right splitting, or a top/bottom splitting. True implies
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
        Description
        -----------
        This method will connect to the load_config_files method in write control and XXX in buffered readout, to initialise connections 
        to the instruments for setting voltages, and the buffered readout to start capturing data on the live plotting window.

        Not currently implemented, as the load_config_files method is not yet implemented in write control.
        """
        
        pass

    def on_autotune(self):
        
        """
        Description
        -----------
        This method starts the defined autotuning protocol. Currently, the autotuning protocol is specific to the Intel Tunnel Falls
        devices, by following the autotuning_protocol.py file. This will be updated to allow for application to general devices.

        Not currently implemented, as the autotuning protocol is not yet fully defined.
        """
        
        pass

    def watchdog_timer(self):

        '''
        Description
        -----------
        This method checks the watchdog of the readout buffer, and if it fails, it will trigger a reset of the GUI.
        '''

        if not self.instrument_handler.watchdog():
            logger.info("Watchdog Failed!")
            # Trigger a reset
            self.logger.error("Readout buffer watchdog detected a problem. Triggering a reset.")
            #python = sys.executable  # path to the Python interpreter
            #os.execv(python, [python] + sys.argv)

    def on_abort(self):

        '''
        Description
        -----------
        This method is called when the user requests to abort the current operation through a button on the GUI.
        '''
        ui.notify('Aborting...')
        self.logger.warning("Abort requested: draining all worker queues")
        self.autotuning_handler.autotuning_thread.abort()
        self.experiment_handler.experiment_thread.abort()
        self.instrument_handler.abort_instruments()

    def on_shutdown(self):

        '''
        Description
        -----------
        This method is called when the user requests to shutdown the GUI. It will abort any currently running experiments, and shutdown the instruments.
        '''
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