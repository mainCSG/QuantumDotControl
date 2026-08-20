'''
File: experiment_thread.py
Authors: Benjamin Van Osch (bvanosch@uwaterloo.ca), Mason Daub (mjdaub@uwaterloo.ca)

This file contains classes related to running experiments from the autotuning code. Experiments are put into a queue, which
keeps track of which experiments to in which order. The queue can be stached and wait for user input to continue, or can be cleared
with an Abort call from the user. 
'''

# Imports


import threading
from queue import PriorityQueue
from collections.abc import Callable
from dataclasses import dataclass
from instrument_handler import TunerFuture, AbortException
from enum import Enum
from typing import Tuple, Dict, Any, Literal, Protocol, Optional, Deque
from qcodes.instrument import Instrument
from tunerlog import TunerLog

_ExperimentThreadInstance = None
_ExperimentHandlerInstance = None

logger = TunerLog('Expt. Control')

def create_experiment_thread():

    '''
    Description
    -----------
    Creates a new experiment thread if one does not already exist. If one does exist, it returns the existing instance.

    Returns
    -------
    ExperimentThread
        The instance of the experiment thread.
    '''

    global _ExperimentThreadInstance

    if _ExperimentThreadInstance is None:
        _ExperimentThreadInstance = ExperimentThread()
        _ExperimentThreadInstance.run()

    return _ExperimentThreadInstance

def get_experiment_handler():

    '''
    Description
    -----------
    Creates a new experiment handler if one does not already exist. If one does exist, it returns the existing instance.

    Returns
    -------
    experiment_handler
        The instance of the experiment handler.
    '''

    global _ExperimentHandlerInstance

    if _ExperimentHandlerInstance is None:
        thread = create_experiment_thread()
        _ExperimentHandlerInstance = experiment_handler(thread)

    return _ExperimentHandlerInstance

class Expt_status(Enum):
            queued = "Queued"
            running = "Running"
            failed = "Failed"
            invalid = "Invalid"

class ExperimentCallback(Protocol):
    def __call__(self, instrument: Instrument, *args: Any) -> Any:
        ...

@dataclass
class experiment_job:
    future : TunerFuture
    when : float
    type : str

class experiment_callback_job(experiment_job):
    def __init__(self, future : TunerFuture, callback : ExperimentCallback, *args, when : float = -1):
        self.callback : Callable[[Instrument], Any] = lambda inst: callback(inst, args)
        super().__init__(future, when, "instrument_callback")

class ExperimentThread:

    def __init__(self):

        '''
        Description
        -----------
        Initializes the thread by setting events and queues. 
        '''

        self.job_event = threading.Event()
        self.abort_event =  threading.Event()
        self.shutdown_event = threading.Event()
        self.job_queue = PriorityQueue()
        self.THREAD_NAME = "ExperimentThread"
        self.thread = threading.Thread(target = self.__thread_loop__, name = self.THREAD_NAME)
    
    def run(self):

        '''
        Description
        -----------
        Starts the experiment thread
        '''

        self.thread.start()
    
    def join(self):

        '''
        Description
        -----------
        Stops the experiment thread by shutting down the code.
        '''

        print("Stopping the experiment thread...")
        self.shutdown_event.set()
        self.job_event.set()
        self.thread.join()
    
    def __assert_correct_thread__(self):

        '''
        Description
        -----------
        Asserts that the thread name is correct for the experiment thread.
        '''

        assert threading.current_thread().name == self.THREAD_NAME, f"The current thread, {threading.current_thread().name}, is not the Experiment Thread." 

    def add_job(self,
                f: Callable,
                args: tuple = (),
                priority: int = 1,
                wait: bool = True,
                timeout: float = None
                ):

        '''
        Description
        -----------
        Adds a job to the thread, where it'll be processed in the loop.

        Parameters
        ----------
        f : Callable
            job to be adding to queue
        args : tuple, optional
            arguments of the job, defaults to empty tuple
        priority : int, optional
            priority of the job in queue, defaults to 1 (highest priority)
        wait : bool, optional
            sets a delay before the job is added to queue, defaults to True
        timeout : float
            sets a period of time that the job is delayed for, default is None

        Returns
        -------
        future : Future Object
            returns the object that has been added to queue saying that the job is added
        '''

        future = TunerFuture()
        self.job_queue.put((priority, (f, args, future)))

        self.job_event.set()

        if wait:
            return future.result(timeout)

        return future

    def abort(self):

        '''
        Description
        -----------
        Aborts the experiment thread by clearing it.
        '''

        self.abort_event.set()
        self.job_event.set()

    def _drain_queue(self):

        '''
        Description
        -----------
        Empties the experiment job queue.
        '''

        while True:
            try:
                priority, (f, args, future) = self.job_queue.get_nowait()
            except Exception:
                break
            future.set_exception(AbortException("Experiment aborted"))
            self.job_queue.task_done()

        self.abort_event.clear()

    def __thread_loop__(self):

        '''
        Description
        -----------
        Loops through the thread queue and processes jobs or events as they come in. Logs any information about events or jobs.
        '''

        print("Starting worker")

        while not self.shutdown_event.is_set():

            self.job_event.wait()
            # safely clears all jobs in queue
            if self.abort_event.is_set():
                self._drain_queue()    
                self.job_event.clear()
                continue

            while self.job_queue.qsize() > 0:

                if self.abort_event.is_set():
                    self._drain_queue()   
                    break

                priority, (f, args, future) = self.job_queue.get()

                try:
                    result = f(*args, self.abort_event)

                except Exception as e:
                    future.set_exception(e)

                else:
                    future.set_result(result)

                self.job_queue.task_done()

            self.job_event.clear()

class experiment_handler:

    def __init__(self, experiment_thread):

        '''
        Description
        -----------
        Makes the experiment thread accessible to all the functions.

        Paramaters
        ----------
        experiment_thread : instance of thread
            instance of experiment thread that the following functions during the protocol can add jobs to
        '''

        self.experiment_thread = experiment_thread

    def do_sweep(self,
                 sweep,
                 instrument_handler,
                 filename,
                 current_setpoints = {},
                 wait: bool = True,
                 timeout: float = 600000):

        '''
        Description
        -----------
        Conducts a sweep on intruments and saves it.

        Parameters
        ----------
        sweep : SweepFunction
            runs the sweep designed
        instrument_handler : HandlerInstance
            instance of the instrument handler contianing all instruments
        filename : str
            name of the file to save this sweep data to
        current_setpoints : dict, optional
            dictionary containing the voltages the gates need to be set to, default is an empty dictionary
        wait : bool, optional
            delay for the sweep to be added to job queue, defaults to True
        timeout : float, optional
            time period to delay the addition to queue for, defaults to 60000s

        Returns
        -------
        self.experiment_thread.add_job() : add_job function
            Adds the job to the queue to highest priority
        '''

        logger.info("Sweep Start!")

        def sweep_fn(abort_event):
            result = sweep.run(instrument_handler, abort_event, filename, current_setpoints)

            return result

        return self.experiment_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )
        
    def set_voltage_configuration(self,
                                  sweep,
                                  instrument_handler,
                                  current_setpoints = {},
                                  wait: bool = True,
                                  timeout: float = 600000):

        '''
        Description
        -----------
        Sets the voltage configuration of the gates.

        Parameters
        ----------
        sweep : SweepFunction
            creates the voltage config
        instrument_handler : HandlerInstance
            instance of the instrument handler contianing all instruments
        current_setpoints : dict, optional
            dictionary containing the voltages the gates need to be set to, default is an empty dictionary
        wait : bool, optional
            delay for the sweep to be added to job queue, defaults to True
        timeout : float, optional
            time period to delay the addition to queue for, defaults to 60000s

        Returns
        -------
        self.experiment_thread.add_job() : add_job function
            Adds the job to the queue with highest priority
        '''

        def sweep_fn(abort_event):

            '''
            Description
            -----------
            Creates the voltage configuration for the sweep.

            Parameters
            ----------
            abort_event : Event
                event trigger that wil lterminate this job if set

            Returns
            -------
            result : dict
                Contains the voltage configuration
            '''

            result = sweep.set_voltage_configuration(instrument_handler, abort_event, current_setpoints)
            return result

        return self.experiment_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )