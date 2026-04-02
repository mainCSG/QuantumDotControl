'''
File: gui.py
Authors: Mason Daub (mjdaub@uwaterloo.ca), Benjamin Van Osch (bvanosch@uwaterloo.ca)

This file contains the gui class that runs the auto tuner.
As of now, we are using the nicegui web server as the user interface for the auto tuner.
'''


import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
import os
import threading
import time
from buffered_readout import buffered_readout


class tuner_gui:
    def __init__(self):
        '''
        Creates an instance of the tuner gui

        :param self:
        '''
        print(threading.current_thread().name)
        self.lipsum_text = 'Lorem ipsum dolor sit, amet consectetur adipisicing elit. Quis praesentium cumque magnam odio iure quidem, quod illum numquam possimus obcaecati commodi minima assumenda consectetur culpa fuga nulla ullam. In, libero.'


        self.readout = buffered_readout()

        self.readout.run()



    def plotting_panel(self):
        with ui.matplotlib().figure as fig:
            #fig = plt.gcf()
            axs = fig.subplots(1, 2)
            xs = np.linspace(-np.pi, np.pi, 100)
            axs[0].plot(xs, np.sin(xs))
            axs[1].plot(xs, np.cos(xs))
            fig.tight_layout()

    def right_panel(self):
        with ui.splitter(horizontal = True) as splitter:
            with splitter.before:
                self.plotting_panel()
            with splitter.after:
                ui.label('Status')
                ui.label('Autotuner: idle')
                ui.label('Instrmuents...')
                ui.label(self.lipsum_text)

    def split_view(self, page):
        with ui.splitter() as splitter:
            with splitter.after:
                self.right_panel()
            with splitter.before:
                page()

    def home_page(self):
        ui.label('Home Page stuff')
        ui.button('Connect to instruments')
        ui.button('Autotune')
        ui.label(self.lipsum_text)
        
        config_files = os.listdir('../configs')
        config_dict = {i : config_files[i] for i in range(len(config_files))}

        ui.button('Print Thread', on_click= lambda : print(threading.current_thread().name))

        ui.button('sleep', on_click = lambda : time.sleep(10))
 
        self.liveplot = ui.matplotlib(figsize = (3,2))

        fig = self.liveplot.figure
        self.ax = fig.subplots(1,1)
        xs = np.linspace(-1, 1)
        self.line = self.ax.plot(xs, np.sin(xs))
        self.liveplot.update()

        ui.timer(0.05, self.update_liveplot)


        self.n = 0

    def update_liveplot(self):
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

    def start(self):
        with ui.header().classes(replace='row items-center') as header:
            with ui.dropdown_button('', icon = 'menu', auto_close=True):
                ui.item('Export', on_click=lambda : ui.notify("You clicked export"))
        #ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
            with ui.tabs() as tabs:
                ui.tab('Home')
                ui.tab('Turn-on')
                ui.tab('Pinch-offs')

        with ui.footer(value=True) as footer:
            ui.label('Footer')
            ui.button('ABORT', on_click = self.on_abort, color='red')

        with ui.tab_panels(tabs, value='Home').classes('w-full'):
            with ui.tab_panel('Home'):
                self.split_view(self.home_page)
            with ui.tab_panel('Turn-on'):
                #self.split_view(self.home_page)
                ui.label('Content of B')
            with ui.tab_panel('Pinch-offs'):
                ui.label('Content of C')

        ui.run(port = 8081)
