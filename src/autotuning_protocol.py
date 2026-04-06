# Import modules

import yaml, datetime, sys, time, os, shutil, json,re
from pathlib import Path

import pandas as pd

import numpy as np

import scipy as sp
from scipy.ndimage import convolve

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from typing import List, Dict

import qcodes as qc
from qcodes.dataset import AbstractSweep, Measurement
from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase
import numpy.typing as npt

import skimage
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import diamond, rectangle  # noqa

import logging
from colorlog import ColoredFormatter
import sys

from nicegui import ui
import threading

class Bootstrapping:

    def ground_device():
        pass

    def turn_on():
        pass

    def pinch_off():
        pass

    def barrier_barrier_sweep():
        pass

    def set_plunger_sweep():
        pass

    def coulomb_diamonds():
        pass

    def tune_lead_dot_tunneling():
        pass

class CoarseTuning:

    def plunger_plunger_sweep():
        pass

class VirtualGating:

    def lever_arm_matrix():
        pass

class ChargeStateTuning:

    def determine_charge_states():
        pass

class FineTuning:

    def rabi_oscilations():
        pass

