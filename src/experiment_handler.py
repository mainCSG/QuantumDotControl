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
    global _ExperimentThreadInstance

    if _ExperimentThreadInstance is None:
        _ExperimentThreadInstance = ExperimentThread()
        _ExperimentThreadInstance.run()

    return _ExperimentThreadInstance

def get_experiment_handler():
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

        self.job_event = threading.Event()
        self.abort_event =  threading.Event()
        self.shutdown_event = threading.Event()
        self.job_queue = PriorityQueue()
        self.THREAD_NAME = "ExperimentThread"
        self.thread = threading.Thread(target = self.__thread_loop__, name = self.THREAD_NAME)
    
    def run(self):

        self.thread.start()
    
    def join(self):

        print("Stopping the experiment thread...")
        self.shutdown_event.set()
        self.job_event.set()
        self.thread.join()
    
    def __assert_correct_thread__(self):

        assert threading.current_thread().name == self.THREAD_NAME, f"The current thread, {threading.current_thread().name}, is not the Experiment Thread." 

    def add_job(self,
            f: Callable,
            args: tuple = (),
            priority: int = 1,
            wait: bool = True,
            timeout: float = None):

        future = TunerFuture()
        self.job_queue.put((priority, (f, args, future)))

        self.job_event.set()

        if wait:
            return future.result(timeout)

        return future

    def abort(self):

        self.abort_event.set()
        self.job_event.set()

    def _drain_queue(self):

        while True:
            try:
                priority, (f, args, future) = self.job_queue.get_nowait()
            except Exception:
                break
            future.set_exception(AbortException("Experiment aborted"))
            self.job_queue.task_done()

        self.abort_event.clear()

    def __thread_loop__(self):

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
        self.experiment_thread = experiment_thread

    def do_sweep(self,
                sweep,
                instrument_handler,
                filename,
                current_setpoints = {},
                wait: bool = True,
                timeout: float = 60000):

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
                timeout: float = 60000):

        def sweep_fn(abort_event):
            result = sweep.set_voltage_configuration(instrument_handler, abort_event, current_setpoints)
            return result

        return self.experiment_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )