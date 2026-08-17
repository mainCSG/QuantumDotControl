# Imports
import logging
import os
import colorlog
from typing import Literal, List
import sys
import datetime
from qcodes.instrument import Instrument
from nicegui import ui

# Initialize global variables
logfile = None
consoleHandler = None
fileHandler = None
uiHandler = None
history = None
loggers : dict[str, logging.Logger] = {}

# Define formats
formatstr = '[%(levelname)-5s %(asctime)s] %(name)s: %(message)s'
formatstr_colored = '%(log_color)s[%(levelname)-5s %(asctime)s] %(name)s:%(reset)s %(message)s'
datefmt = '%m-%d-%Y %H:%M:%S'

class StorageHandler(logging.Handler):

    '''
    Description
    -----------
    A logging handler that stores all log records in memory. This is useful for sending the log history to a UI element when it is created.
    '''

    def __init__(self, level : int):
        self.history : List[logging.LogRecord] = []
        super().__init__(level)
    def emit(self, record : logging.LogRecord) -> None:
        self.history.append(record)

class LogElementHandler(logging.Handler):

    """
    Description
    -----------
    A logging handler that emits messages to a log element.
    """

    def __init__(self, element: ui.log, level: int = logging.NOTSET) -> None:
        self.element = element
        super().__init__(level)

        global formatstr, datefmt
        formatter = logging.Formatter(formatstr, datefmt=datefmt)
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelno
            colors = {logging.DEBUG: "text-blue text-sm italic", logging.INFO: "text-sm", logging.WARNING: "text-yellow-600 text-sm", logging.ERROR: "text-red", logging.CRITICAL: "text-red font-bold underline"}
            msg = self.format(record)

            client = self.element.client
            self.element.push(msg, classes=colors[level])
        except Exception as e:
            print("UI handler failed:", repr(e))
            self.handleError(record)

class TunerLog(logging.Logger):

    def __init__(self, name : str, level : Literal['debug', 'info', 'warning', 'error'] = 'debug'):

        '''
        Description
        -----------
        A custom logger that logs to a file, the console, and a UI element. It also stores the log history in memory for later use.
        '''

        global logfile, consoleHandler, fileHandler, loggers, formatstr, formatstr_colored, datefmt, history

        try:
            level_num = getattr(logging, level.upper())
        except:
            ...
        else:
            super().__init__(name, level_num)

            if name in loggers:
                return

            if consoleHandler is None:
                consoleHandler = colorlog.StreamHandler(sys.stdout)
                consoleHandler.setLevel(level_num)
                colorFormatter = colorlog.ColoredFormatter(formatstr_colored, datefmt=datefmt,log_colors={
                        'DEBUG': 'blue',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'bold_red',
                    })
                consoleHandler.setFormatter(colorFormatter)

            original_dir = os.getcwd()

            # Creates a log file
            if logfile is None:

                os.chdir("..")
                logfile_dir = f"Protocol_Run_{datetime.datetime.now().strftime('%m-%d-%Y')}"
                logfile = f"QDot_tuner_{datetime.datetime.now().strftime('%m-%d-%Y')}.log"
                logfile = os.path.join(logfile_dir, logfile)

            # Creates the log file directory if it does not exist
            if fileHandler is None:
                fileHandler = logging.FileHandler(logfile)
                fileHandler.setLevel(level_num)
                formatter = logging.Formatter(formatstr, datefmt=datefmt)
                fileHandler.setFormatter(formatter)
            if history is None:
                history = StorageHandler(level_num)

            self.addHandler(fileHandler)
            self.addHandler(consoleHandler)
            self.addHandler(history)

            self.propagate = False
            self.info("Initalizing logger %s with log file '%s'", name, logfile)
            
            loggers[name] = self # add to list of loggers
    
            os.chdir(original_dir)

    def add_ui_handler(self, element: ui.log, level: Literal['debug', 'info', 'warning', 'error'] = 'info'):
        
        """
        Description
        -----------
        Add the UI handler to all loggers, and make sure all the previous history gets sent to the ui logger.

        Parameters
        ----------
        element : ui.log
            The UI element to which log messages will be emitted.
        level : Literal['debug', 'info', 'warning', 'error'], optional
            The logging level for the UI handler. Defaults to 'info'.
        """
        
        level_num = getattr(logging, level.upper())

        global loggers, uiHandler, history

        if uiHandler is not None:
            for logger in loggers.values():
                logger.removeHandler(uiHandler)
        
        uiHandler = LogElementHandler(element, level_num)
        
        for logger in loggers.values():
            if not uiHandler in logger.handlers:
                logger.addHandler(uiHandler)
                if history is not None:
                    logger.removeHandler(history)
        
        if history is not None:
            for record in history.history:
                uiHandler.emit(record)

    def instrument_snapshot(self, instrument : Instrument, level : Literal['debug', 'info', 'warning', 'error'] = 'info'):

        '''
        Description
        -----------
        Print a snapshot of the instrument's parameters and their values to the log.

        Parameters
        ----------
        instrument : Instrument
            The instrument to snapshot.
        level : Literal['debug', 'info', 'warning', 'error'], optional
            The logging level for the snapshot. Defaults to 'info'.
        '''
        
        params = []
        maxlenl = len("Parameter") + 3
        maxlenr = len("Value") + 3
        for key, item in instrument.parameters.items():
            try:
                value = getattr(item ,"get")()
                unit = getattr(item, "unit")
            except:
                value = None
                unit = ""
            else:
                if isinstance(value, (int,float)):
                    valuestr = f"   {value:.3g} {unit}"
                else:
                    valuestr = "   " + str(value)
                if len(valuestr) > 30:
                    valuestr = valuestr[:30]+'...'
                params.append((key, valuestr))
                if len(valuestr) > maxlenr:
                    maxlenr = len(valuestr)
                if len(key) > maxlenl:
                    maxlenl = len(key)
        total_len = maxlenl + maxlenr + 3
        output = f"Instrument '{instrument}':\n{'   Parameter':<{maxlenl}} | {'   Value':<{maxlenr}}\n{'-'*total_len}\n"
        for key, valuestr in params:
            output += f"{key:<{maxlenl}} : {valuestr:<{maxlenr}}\n"
        
        try:
            getattr(self, level)(output)
        except:
            self.exception("Error printing instrument snapshot")