'''
File: gui.py
Authors: Mason Daub (mjdaub@uwaterloo.ca), Benjamin Van Osch (bvanosch@uwaterloo.ca)

This file contains the gui class that runs the auto tuner.
As of now, we are using the nicegui web server as the user interface for the auto tuner.
'''


import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui, app
import os
import threading
import time
from buffered_readout import create_buffer_instance

_gui_instances = []

class tuner_gui:
    
    def __init__(self):
        
        '''
        Creates an instance of the tuner gui

        params: 
            self:
        '''
        global _FirstPass
        

        self.lipsum_text = 'Lorem ipsum dolor sit, amet consectetur adipisicing elit. Quis praesentium cumque magnam odio iure quidem, quod illum numquam possimus obcaecati commodi minima assumenda consectetur culpa fuga nulla ullam. In, libero.'
        print(threading.current_thread().name)
        global _gui_instance
        _gui_instances.append(self)

        self.readout = create_buffer_instance()

        self.readout.run()

    def start(self):

        """
        The method that intialises the gui. As of now, it also defines the main page of the within itself.

        params: 
            self:
        """

        self.header()

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

                        ui.button('Connect to instruments')
                        ui.button('Autotune')
                        ui.label(self.lipsum_text)
                        
                        config_files = os.listdir('../configs')
                        config_dict = {i : config_files[i] for i in range(len(config_files))}

                    with ui.tab_panel('Bootstrapping'):

                        ui.label('Bootstrapping Information')
                    
                    with ui.tab_panel('Coarse Tuning'):
                        
                        ui.label('Coarse Tuning Information')   

                    with ui.tab_panel('Virtual Gating'):
                        
                        ui.label('Virtual Gating Information')

                    with ui.tab_panel('Charge State Tuning'):
                        
                        ui.label('Charge State Tuning')

                    with ui.tab_panel('Fine Tuning'):
                        
                        ui.label('Fine Tuning Information')

            with splitter1.after:

                with ui.splitter(horizontal = True) as splitter2:
                    
                    with splitter2.before:
                        
                        self.live_plot_window()

                    with splitter2.after:

                        ui.label('Logger Information').classes('ml-2')


                ui.timer(0.05, self.update_liveplot)
                self.n = 0

        self.footer()

        ui.run(port = 8081)

    def header(self):
        
        """
        The method that defines the header of the gui.

        params: 
            self:
        """

        with ui.header().classes(replace='row items-center') as header:
            ui.label('Header')

    def footer(self):
        
        """
        The method that intialises the footer of the gui.

        params: 
            self:
        """

        with ui.footer(value=True) as footer:
            ui.label('Footer')
            ui.button('ABORT', on_click = self.on_abort, color='red')

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
   
    def live_plot_window(self):

        """
        The method that defines the live plot window, which streams the measurement of our readout instrument.

        params:
            self:
        
        """

        self.liveplot = ui.matplotlib(figsize = (30,20))

        fig = self.liveplot.figure
        self.ax = fig.subplots(1,1)
        xs = np.linspace(-1, 1)
        self.line = self.ax.plot(xs, np.sin(xs))
        self.liveplot.update()

    def update_liveplot(self):
        
        """
        The method that updates the live plot window. This method gets values from the readout buffer, and adds them to the 
        live plot window.

        params:
            self:
        
        """

        retval = self.readout.get_buffer()
        if retval is None:
            return
        else:
            data, times = retval
            self.line[0].set_ydata(data)
            self.line[0].set_xdata(times)
            self.ax.set_xlim(min(times), max(times))
            self.ax.set_ylim(-0.5, 1.5)

            #self.ax.relim()
            #self.ax.autoscale_view()
            #for l in self.ax.lines:
                #l.remove()
            #self.ax.clear()
            #self.ax.plot(times, data)

            #self.liveplot.figure.canvas.draw()
            self.liveplot.figure.tight_layout()
            self.liveplot.update()
            ui.update()

    def on_abort(self):
        ui.notify('Aborting...')
    
    def on_shutdown(self):
        self.readout.join()

    def experiment_progress_bar():

        pb = ui.linear_progress()

@app.on_shutdown
def shutdown():
    global _gui_instances
    for inst in _gui_instances:
        inst.on_shutdown()
