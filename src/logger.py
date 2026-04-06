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


class Logger:

    def initialise_logger(self):

        """
        This method creates new logging categories which the user can see while the autotuner runs, as well as outputs
        a log of everything that occured during the experiment. This method is when an InstrumentControl object is initialised.
        """

        ATTEMPT, COMPLETE, IN_PROGRESS, RESULTS = logging.INFO - 3, logging.INFO - 2, logging.INFO - 1, logging.INFO

        logging.addLevelName(ATTEMPT, 'ATTEMPT')
        logging.addLevelName(COMPLETE, 'COMPLETE')
        logging.addLevelName(IN_PROGRESS, 'IN PROGRESS')
        logging.addLevelName(RESULTS, 'RESULTS')

        def attempt(self, message, *args, **kwargs):
            if self.isEnabledFor(ATTEMPT):
                self._log(ATTEMPT, message, args, **kwargs)
        
        def complete(self, message, *args, **kwargs):
            if self.isEnabledFor(COMPLETE):
                self._log(COMPLETE, message, args, **kwargs)

        def in_progress(self, message, *args, **kwargs):
            if self.isEnabledFor(IN_PROGRESS):
                self._log(IN_PROGRESS, message, args, **kwargs)

        def results(self, message, *args, **kwargs):
            if self.isEnabledFor(RESULTS):
                self._log(RESULTS, message, args, **kwargs)

        logging.Logger.attempt = attempt
        logging.Logger.complete = complete
        logging.Logger.in_progress = in_progress
        logging.Logger.results = results

        console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s %(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    'ATTEMPT': 'yellow',
                    'COMPLETE': 'green',
                    'RESULTS': 'white',
                    'INFO': 'white',
                    'IN PROGRESS': 'white',
                    'WARNING': 'red',
                    'ERROR': 'bold_red',
                    'CRITICAL': 'bold_red'
                }
        )

        # Then, we set the format of the above messages displayed to the user

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s %(message)s"
        )

        # Now, we define a handler, which writes the messages from the Logger

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        # Now we create an info log for the Logger

        file_handler = logging.FileHandler(
            os.path.join(self.db_folder, 'run_info.log')
        )

        file_handler.setFormatter(file_formatter)

        # Finally, we define the logger

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        self.logger.setLevel(min(self.logger.getEffectiveLevel(), ATTEMPT))

        return None