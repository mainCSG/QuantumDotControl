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
from instrument_handler import TunerFuture
from enum import Enum
from typing import Tuple, Dict, Any, Literal, Protocol, Optional, Deque
from qcodes.instrument import Instrument
from tunerlog import TunerLog

from autotuning_protocol import Bootstrapping, GlobalChargeTuning, VirtualGating, ChargeStateTuning, QubitTuning

_AutotuningThreadInstance = None
_AutotuningHandlerInstance = None

logger = TunerLog('Autotuning Handler')

def create_autotuning_thread():
    global _AutotuningThreadInstance

    if _AutotuningThreadInstance is None:
        _AutotuningThreadInstance = AutotuningThread()
        _AutotuningThreadInstance.run()

    return _AutotuningThreadInstance

def get_autotuning_handler():
    global _AutotuningHandlerInstance

    if _AutotuningHandlerInstance is None:
        thread = create_autotuning_thread()
        _AutotuningHandlerInstance = autotuning_handler(thread)

    return _AutotuningHandlerInstance

class Auto_status(Enum):
            queued = "Queued"
            running = "Running"
            failed = "Failed"
            invalid = "Invalid"

class AutotuningCallback(Protocol):
    def __call__(self, instrument: Instrument, *args: Any) -> Any:
        ...

@dataclass
class autotuning_job:
    future : TunerFuture
    when : float
    type : str

class autotuning_callback_job(autotuning_job):
    def __init__(self, future : TunerFuture, callback : AutotuningCallback, *args, when : float = -1):
        self.callback : Callable[[Instrument], Any] = lambda inst: callback(inst, args)
        super().__init__(future, when, "instrument_callback")

class AutotuningThread:

    def __init__(self):

        self.job_event = threading.Event()
        self.abort_event =  threading.Event()
        self.shutdown_event = threading.Event()
        self.job_queue = PriorityQueue()
        self.THREAD_NAME = "AutotuningThread"
        self.thread = threading.Thread(target = self.__thread_loop__, name = self.THREAD_NAME)
    
    def run(self):

        self.thread.start()
    
    def join(self):

        print("Stopping the autotuning thread...")
        self.shutdown_event.set()
        self.thread.join()
    
    def __assert_correct_thread__(self):

        assert threading.current_thread().name == self.THREAD_NAME, f"The current thread, {threading.current_thread().name}, is not the Autotuning Thread." 

    def add_job(self,
            f: Callable,
            args: tuple = (),
            priority: int = 1,
            wait: bool = True,
            timeout: float = None):
    
        future = TunerFuture()

        job = (priority, (f, args, future))
        self.job_queue.put(job)

        # Wake the thread
        self.job_event.set()

        if wait:
            return future.result(timeout)
        return future
    
    def abort(self):

        self.abort_event.set()
    
    def __thread_loop__(self):

        print("Starting the Autotuning Thread Worker")

        while not self.shutdown_event.is_set():

            self.job_event.wait()

            while self.job_queue.qsize() > 0:

                if self.abort_event.is_set(): # Clear remaining jobs safely 
                    while not self.job_queue.empty(): 
                        try: 
                            _, (_, _, future) = self.job_queue.get_nowait() 
                            future.set_exception(RuntimeError("Autotuning Stage aborted")) 
                            self.job_queue.task_done() 
                        except: 
                            break 
                        self.abort_event.clear() 
                        continue

                priority, data = self.job_queue.get()
                f, args, future = data

                try:
                    result = f(*args, self.abort_event)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)

                self.job_queue.task_done()

            # reset event once queue is empty
            self.job_event.clear()

        logger.info("Exiting Thread!")

class autotuning_handler:

    def __init__(self, autotuning_thread):
        self.autotuning_thread = autotuning_thread

    def run_bootstrapping(self,
                device_config,
                instrument_handler,
                experiment_handler,
                wait: bool = True,
                timeout: float = 6000):

        def sweep_fn(abort_event):
            result = Bootstrapping(device_config = device_config,
                                   instrument_handler = instrument_handler,
                                   experiment_handler = experiment_handler
                                  )
            return result

        return self.autotuning_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )
        
    def run_global_charge_tuning(self,
                sweep,
                instrument_handler,
                current_setpoints = {},
                wait: bool = True,
                timeout: float = 60):

        def sweep_fn(abort_event):
            result = GlobalChargeTuning()
            return result

        return self.autotuning_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )

    def run_virtual_gating(self,
                sweep,
                instrument_handler,
                current_setpoints = {},
                wait: bool = True,
                timeout: float = 60):

        def sweep_fn(abort_event):
            result = VirtualGating()
            return result

        return self.autotuning_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )
    
    def run_charge_state_tuning(self,
                sweep,
                instrument_handler,
                current_setpoints = {},
                wait: bool = True,
                timeout: float = 60):

        def sweep_fn(abort_event):
            result = ChargeStateTuning()
            return result

        return self.autotuning_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )
    
    def run_qubit_tuning(self,
                sweep,
                instrument_handler,
                current_setpoints = {},
                wait: bool = True,
                timeout: float = 60):

        def sweep_fn(abort_event):
            result = QubitTuning()
            return result

        return self.autotuning_thread.add_job(
                                              sweep_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )