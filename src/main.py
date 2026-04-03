'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner. This

'''

from nicegui import app, ui
from gui import tuner_gui

gui : tuner_gui

@app.on_startup
def start_tuner_gui():
    print("Starting the GUI...")
    global gui
    gui = tuner_gui()

@app.on_shutdown
def stop_tuner_gui():
    print("Stopping the gui")
    global gui
    gui.on_shutdown()

@ui.page('/')
def tuner_gui_root_page():
    global gui
    gui.root_page()

ui.run(port = 8081)
