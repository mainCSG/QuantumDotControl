'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner.

'''
#from logger import Logger
#from instrument_control import InstrumentControl
#from data_analysis import DataAnalysis
from gui import tuner_gui


#log = Logger()
#ic = InstrumentControl(log, config = config, tuner_config = tuner_config, station_config = station_config)
#om = DataAnalysis(log, config = config, tuner_config = tuner_config, station_config = station_config)
#gui = tuner_gui(log, config = config, tuner_config = tuner_config, station_config = station_config)
gui = tuner_gui()

gui.start()
