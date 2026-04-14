'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner. This

'''

from nicegui import app, ui
from gui import tuner_gui
from tunerlog import TunerLog
gui : tuner_gui
logger : TunerLog

@app.on_startup
def start_tuner_gui():
    global gui, logger
    logger = TunerLog("main")
    logger.info("Starting the GUI...")
    gui = tuner_gui()

@app.on_shutdown
def stop_tuner_gui():
    logger.warning("Stopping the GUI...")
    global gui
    gui.on_shutdown()

@ui.page('/')
def tuner_gui_root_page():
    global gui
    logger.debug("Defining the server root page")
    gui.root_page()

ui.run(port = 8081)
