'''
File: autotuning_protocol.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

This file details the autotuning protocol in its entirety. This file uses the functions defined in write_control and 
data_analysis to implement experiments for automatic tuning of quantum dot devices.

Each stage of the autotuning code is a class, with the methods of the class corresponding to the experiments of the

'''

from instrument_control import InstrumentControl
from data_analysis import DataAnalysis

class Bootstrapping(InstrumentControl, DataAnalysis):

    def __init__(self):
        
        return None

    def extract_turn_on_voltage(self,
                gate):
        return None
    
    def extract_gate_ranges(self,
                  gate):
        return None
    
    def extract_working_points(self,
                             gates):
        return None
    
    def extract_coulomb_blockade_peaks(self,
                     gate):
        
        return None
    
    def extract_coulomb_diamonds(self,
                         gates):
        
        return None
    
    def extract_lead_dot_tunneling(self,
                           gate):
        
        return None

class CoarseTuning(InstrumentControl, DataAnalysis):

    def __init__(self):

        return None
    
    def extract_charge_stability_diagram():

        return None
    
class VirtualGating:

    def __init__(self):
        
        return None
    
    def construct_lever_arm_matrix():

        return None
    
    def extract_virtual_CSD():

        return None
    
class ChargeStateTuning:

    def extract_charge_states():

        return None
    
class FineTuning:

    def find_PSB():

        return None

    def extract_rabi_frequency():

        return None 