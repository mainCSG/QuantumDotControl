'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner.
'''

import datetime
import os

protocol_folder = None
datafolder = None

# Defines the protocol folder to put the log file and data in for that specific protocol run
if protocol_folder is None:
        
    current_dir = os.getcwd()
    os.chdir("..")
    protocol_folder = f"Protocol_Run_{datetime.datetime.now().strftime('%m-%d-%Y')}"
    os.makedirs(protocol_folder, exist_ok=True)
    os.chdir(current_dir)

# Defines the data folder to store all the data inside of in the protocol folder
if datafolder is None:
    current_dir = os.getcwd()
    os.chdir("..")
    datafolder = f"Protocol_Run_{datetime.datetime.now().strftime('%m-%d-%Y')}"
    datafolder = os.path.join(datafolder, "Data")
    os.makedirs(datafolder, exist_ok=True)
    os.chdir(current_dir)

from nicegui import app, ui
from gui import tuner_gui
from tunerlog import TunerLog
from gui_bridge import tuning_bridge

gui = None
logger = None

@app.on_startup
def start_tuner_gui():

    '''
    Description
    -----------
    This method is called when the program starts up. It will initialize the logger, and start the GUI.
    '''

    global gui, logger, datafolder

    print("Starting Program")

    logger = TunerLog("main")
    logger.info("Starting GUI...")

    gui = tuner_gui(tuning_bridge)

    print("Gui Startup Complete! Welcome to the QAT!")

@app.on_shutdown
def stop_tuner_gui():

    '''
    Description
    -----------
    This method is called when the program is shutting down. It will stop the GUI, and shuts down the logger.
    '''

    global gui, logger, datafolder

    if logger is not None:
        logger.warning("Stopping the GUI...")

    if gui is not None:
        gui.on_shutdown()

@ui.page('/')
def tuner_gui_root_page():

    '''
    Description
    -----------
    This method is called when the user navigates to the root page of the GUI. It will display a message to the user that the GUI is still starting, and to refresh shortly.
    '''

    global gui, logger, datafolder

    if gui is None:
        ui.label("GUI is still starting... please refresh shortly")
        return

    if logger is not None:
        logger.debug("Defining the server root page")

    gui.root_page()

# Runs the GUI
ui.run(port = 8081)
