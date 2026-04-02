'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner.

'''
from gui import tuner_gui

if __name__ in {"__main__", "__mp_main__"}:
    print("Creating")
    gui = tuner_gui()
    print("Starting")
    gui.start()
