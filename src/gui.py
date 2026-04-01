'''
File: gui.py
Authors: Mason Daub (mjdaub@uwaterloo.ca), Benjamin Van Osch (bvanosch@uwaterloo.ca)

This file contains the gui class that runs the auto tuner.
As of now, we are using the nicegui web server as the user interface for the auto tuner.
'''


import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui


class tuner_gui:
    def __init__(self):
        '''
        Creates an instance of the tuner gui

        :param self:
        '''
        self.lipsum_text = 'Lorem ipsum dolor sit, amet consectetur adipisicing elit. Quis praesentium cumque magnam odio iure quidem, quod illum numquam possimus obcaecati commodi minima assumenda consectetur culpa fuga nulla ullam. In, libero.'

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
                self.split_view(self.home_page)
            with ui.tab_panel('Pinch-offs'):
                ui.label('Content of C')

        ui.run()
