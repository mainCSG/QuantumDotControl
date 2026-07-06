'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner.

'''

import os
from nicegui import app, ui
from gui import tuner_gui
from tunerlog import TunerLog
import datetime

gui = None
logger = None
protocol_folder = None
datafolder = None

@app.on_startup
def start_tuner_gui():
    global gui, logger, datafolder

    print("Starting Program")

    logger = TunerLog("main")
    logger.info("Starting GUI...")

    if protocol_folder is None:
        protocol_folder = f"../Protocol_Run_{datetime.datetime.now().strftime('%m-%d-%Y')}"
        os.makedirs(protocol_folder, exist_ok=True)

    if datafolder is None:
        datafolder = f"../Protocol_Run_{datetime.datetime.now().strftime('%m-%d-%Y')}/Data"
        os.makedirs(datafolder, exist_ok=True)

    gui = tuner_gui()

    print("Gui Startup Complete! Welcome to QAT!")

@app.on_shutdown
def stop_tuner_gui():
    global gui, logger, datafolder

    if logger is not None:
        logger.warning("Stopping the GUI...")

    if gui is not None:
        gui.on_shutdown()

@ui.page('/')
def tuner_gui_root_page():
    global gui, logger, datafolder

    if gui is None:
        ui.label("GUI is still starting... please refresh shortly")
        return

    if logger is not None:
        logger.debug("Defining the server root page")

    gui.root_page()

ui.run(port = 8081)
