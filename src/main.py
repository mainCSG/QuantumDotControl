'''
File: main.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

Entry point to the auto tuner. This

'''

import yaml

from logger import Logger
from src.write_control import InstrumentControl
from data_analysis import DataAnalysis
from gui import tuner_gui


"""
Currently, the config files are defined in the Configs folder, and read here to instantiate each of the classes.
The user would have to create new config files in order to load a separate protocol into the Autotuner.

TODO We can probably generalize the station and tuner config files since they don't really need to be changed much, so we could
     consider having the config file loading from the gui section and the user can pick which one after the application is opened.
     The most general version would be to create/edit this config file within the gui but that can be added later.

"""

with open('configs/Intel_Config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('configs/Intel_Tuner_Config.yaml', 'r') as f:
    tuner_config = yaml.safe_load(f)

with open('configs/Intel_Station_Config.yaml', 'r') as f:
    station_config = yaml.safe_load(f)

log = Logger()
ic = InstrumentControl(log, config = config, tuner_config = tuner_config, station_config = station_config)
om = DataAnalysis(log, config = config, tuner_config = tuner_config, station_config = station_config)
gui = tuner_gui(log, config = config, tuner_config = tuner_config, station_config = station_config)

gui.start()
