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

from autotuning_protocol import Protocol, Bootstrapping, GlobalChargeTuning, VirtualGating, ChargeStateTuning, QubitTuning

_AutotuningThreadInstance = None
_AutotuningHandlerInstance = None

logger = TunerLog('Autotuning Handler')

def create_autotuning_thread():

    '''
    Description
    -----------
    Creates an instance of the autotuning thread that the autotuning protocol runs on. All sweeps and protocol jobs run on this thread.
    '''

    global _AutotuningThreadInstance

    if _AutotuningThreadInstance is None:
        _AutotuningThreadInstance = AutotuningThread()
        _AutotuningThreadInstance.run()

    return _AutotuningThreadInstance

def get_autotuning_handler():

    '''
    Description
    -----------
    Creates an instance of the autotuning handler that has all the functions to run each stage of the autotuning protocol.
    Also creates a autotuning thread if not created already.
    '''

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

        '''
        Description
        -----------
        Initializes the thread by setting events and queues. 
        '''

        self.job_event = threading.Event() # Triggers when a job come sin through the thread
        self.abort_event =  threading.Event() # Triggers when the thread is getting aborted, when the abort button is pressed
        self.shutdown_event = threading.Event() # Triggers during the shutdown of the GUI
        self.job_queue = PriorityQueue() # Creates a priority queue for the jobs that come into the thread
        self.THREAD_NAME = "AutotuningThread" 
        # Starts a loop for the thread so it continuously checks the job queue or any events
        self.thread = threading.Thread(target = self.__thread_loop__, name = self.THREAD_NAME)
    
    def run(self):

        '''
        Description
        -----------
        Starts the autotuning thread
        '''

        self.thread.start()
    
    def join(self):

        '''
        Description
        -----------
        Stops the autotuning thread by shutting down the code.
        '''

        print("Stopping the autotuning thread...")
        self.shutdown_event.set()
        self.job_event.set()
        self.thread.join()
    
    def __assert_correct_thread__(self):

        '''
        Description
        -----------
        Asserts that the thread name is correct for the autotuning thread.
        '''

        assert threading.current_thread().name == self.THREAD_NAME, f"The current thread, {threading.current_thread().name}, is not the Autotuning Thread." 

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

        job = (priority, (f, args, future))
        self.job_queue.put(job)

        # Wake the thread
        self.job_event.set()

        if wait:
            return future.result(timeout)
        return future
    
    def abort(self):

        '''
        Description
        -----------
        Aborts the autotuning thread by clearing it.
        '''

        self.abort_event.set()
        self.job_event.set()

    def _drain_queue(self):

        '''
        Description
        -----------
        Empties the autotuning job queue.
        '''

        while True:
            try:
                priority, (f, args, future) = self.job_queue.get_nowait()
            except Exception:
                break
            future.set_exception(AbortException("Autotuning job aborted"))
            self.job_queue.task_done()

        self.abort_event.clear()

    def __thread_loop__(self):

        '''
        Description
        -----------
        Loops through the thread queue and processes jobs or events as they come in. Logs any information about events or jobs.
        '''

        print("Starting the Autotuning Thread Worker")

        while not self.shutdown_event.is_set():

            self.job_event.wait()
            # safely removes all jobs in queue
            if self.abort_event.is_set():
                self._drain_queue()        
                self.job_event.clear()
                continue

            while self.job_queue.qsize() > 0:

                if self.abort_event.is_set():
                    self._drain_queue()   
                    break

                priority, data = self.job_queue.get()
                f, args, future = data

                try:
                    result = f(*args, self.abort_event)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)

                self.job_queue.task_done()

            self.job_event.clear()

        logger.info("Exiting Thread!")

class autotuning_handler:

    def __init__(self, autotuning_thread):

        '''
        Description
        -----------
        Makes the autotuning thread accessible to all the functions.

        Paramaters
        ----------
        autotuning_thread : instance of thread
            instance of autotuning thread that the following functions during the protocol can add jobs to
        '''

        self.autotuning_thread = autotuning_thread

    def run_snapshot(self,
                     device_config,
                     instrument_handler,
                     experiment_handler,
                     wait: bool = True,
                     timeout: float = 60000):

        def snapshot_fn(abort_event):
        
            result = Protocol(device_config = device_config,
                              instr_handler = instrument_handler,
                              exp_handler = experiment_handler
                             )

            result.parameter_snapshot(name = "Device")

            return result
        
        return self.autotuning_thread.add_job(snapshot_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )

    def run_bootstrapping(self,
                          device_config,
                          instrument_handler,
                          experiment_handler,
                          wait: bool = True,
                          timeout: float = 6000
                          ):

        '''
        Description
        -----------
        Runs the sweep functions to conduct bootstrapping.

        Parameters
        ----------
        device_config : yaml file
            file containing the device information and setup
        instrument_handler : instance of handler
            instance of instruemnt handler to process and control instruments
        experiment_handler : instance of handler
            instance of experiment handler to process the autotuning steps in bootstrapping stage
        wait : bool, optional
            sets a delay to every job, defaults to True
        timeout : float, optional
            sets a time period to delay each job for, defaults to 6000s
        
        Returns
        -------
        The sweep job for the bootstrapping stage.
        '''

        def autotuning_fn(abort_event):

            '''
            Description
            -----------
            Creates a sweep function to run the bootstrapping steps.

            Parameters
            ----------
            abort_event : Event
                triggers an abort for the sweep

            Returns
            -------
            result :
                the results from the bootstrapping stage
            '''

            result = Bootstrapping(device_config = device_config,
                                   instr_handler = instrument_handler,
                                   exp_handler = experiment_handler
                                  )

            result.autotune(instr_handler = instrument_handler,
                            exp_handler = experiment_handler,
                            num_points_bootstrapping = [150, 150, 200, 200, 200],
                            dev_mode = False
                           )

            return result

        return self.autotuning_thread.add_job(
                                              autotuning_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )

    def run_global_charge_tuning(self,
                                 device_config,
                                 instrument_handler,
                                 experiment_handler,
                                 wait: bool = True,
                                 timeout: float = 6000
                                 ):

        '''
        Description
        -----------
        Runs the sweep functions to conduct global charge tuning.

        Parameters
        ----------
        device_config : yaml file
            file containing the device information and setup
        instrument_handler : instance of handler
            instance of instruemnt handler to process and control instruments
        experiment_handler : instance of handler
            instance of experiment handler to process the autotuning steps in global charge tuning stage
        wait : bool, optional
            sets a delay to every job, defaults to True
        timeout : float, optional
            sets a time period to delay each job for, defaults to 6000s
        
        Returns
        -------
        The sweep job for the global charge tuning stage.
        '''

        def autotuning_fn(abort_event):

            '''
            Description
            -----------
            Creates a sweep function to run the global charge tuning steps.

            Parameters
            ----------
            abort_event : Event
                triggers an abort for the sweep

            Returns
            -------
            result :
                the results from the global charge tuning stage
            '''
            
            result = GlobalChargeTuning(device_config = device_config,
                                        instr_handler = instrument_handler,
                                        exp_handler = experiment_handler
                                       )

            result.autotune(instr_handler = instrument_handler,
                            exp_handler = experiment_handler,
                            num_points_bootstrapping = [150, 150, 200, 200, 200],
                            num_points_global_charge_tuning = [150, 400, 400],
                            dev_mode = False
                           )

            return result

        return self.autotuning_thread.add_job(
                                              autotuning_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )

    def run_gbt_from_cp(self,
                        device_config,
                        instrument_handler,
                        experiment_handler,
                        checkpoint,
                        wait: bool = True,
                        timeout: float = 6000,
                        ):
    
            '''
            Description
            -----------
            Runs the sweep functions to conduct global charge tuning.
    
            Parameters
            ----------
            device_config : yaml file
                file containing the device information and setup
            instrument_handler : instance of handler
                instance of instruemnt handler to process and control instruments
            experiment_handler : instance of handler
                instance of experiment handler to process the autotuning steps in global charge tuning stage
            wait : bool, optional
                sets a delay to every job, defaults to True
            timeout : float, optional
                sets a time period to delay each job for, defaults to 6000s
            
            Returns
            -------
            The sweep job for the global charge tuning stage.
            '''

            def autotuning_fn(abort_event):
    
                '''
                Description
                -----------
                Creates a sweep function to run the global charge tuning steps.
    
                Parameters
                ----------
                abort_event : Event
                    triggers an abort for the sweep
    
                Returns
                -------
                result :
                    the results from the global charge tuning stage
                '''
                
                result = GlobalChargeTuning(device_config = device_config,
                                            instr_handler = instrument_handler,
                                            exp_handler = experiment_handler
                                           )

                result.load_ckpt(checkpoint)

                result.autotune_global_charge_tuning(num_points = [150, 400, 400])
                return result
    
            return self.autotuning_thread.add_job(autotuning_fn,
                                                  args=(),
                                                  wait=wait,
                                                  timeout=timeout
                                                 )

    def run_virtual_gating(self,
                           device_config,
                           instrument_handler,
                           experiment_handler,
                           wait: bool = True,
                           timeout: float = 6000
                           ):

        '''
        Description
        -----------
        Runs the sweep functions to conduct virtual gating.

        Parameters
        ----------
        device_config : yaml file
            file containing the device information and setup
        instrument_handler : instance of handler
            instance of instruemnt handler to process and control instruments
        experiment_handler : instance of handler
            instance of experiment handler to process the autotuning steps in virtual gating stage
        wait : bool, optional
            sets a delay to every job, defaults to True
        timeout : float, optional
            sets a time period to delay each job for, defaults to 6000s
        
        Returns
        -------
        The sweep job for the virtual gating stage.
        '''

        def autotuning_fn(abort_event):

            '''
            Description
            -----------
            Creates a sweep function to run the virtual gating steps.

            Parameters
            ----------
            abort_event : Event
                triggers an abort for the sweep

            Returns
            -------
            result :
                the results from the virtual gating stage
            '''
            
            result = VirtualGating(device_config = device_config,
                                   instr_handler = instrument_handler,
                                   exp_handler = experiment_handler
                                  )

            result.autotune(instr_handler = instrument_handler,
                            exp_handler = experiment_handler,
                            num_points_bootstrapping = [100, 100, 50, 50, 50],
                            num_points_global_charge_tuning = [100, 100, 50],
                            num_points_virtual_gating = [],
                            dev_mode = True
                           )

            return result

        return self.autotuning_thread.add_job(
                                              autotuning_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )
    
    def run_charge_state_tuning(self,
                                sweep,
                                instrument_handler,
                                current_setpoints = {},
                                wait: bool = True,
                                timeout: float = 60
                                ):

        '''
        Description
        -----------
        Runs the sweep functions to conduct charge state tuning.

        Parameters
        ----------
        device_config : yaml file
            file containing the device information and setup
        instrument_handler : instance of handler
            instance of instruemnt handler to process and control instruments
        experiment_handler : instance of handler
            instance of experiment handler to process the autotuning steps in charge state tuning stage
        wait : bool, optional
            sets a delay to every job, defaults to True
        timeout : float, optional
            sets a time period to delay each job for, defaults to 60s
        
        Returns
        -------
        The sweep job for the charge state tuning stage.
        '''

        def autotuning_fn(abort_event):

            '''
            Description
            -----------
            Creates a sweep function to run the charge state tuning steps.

            Parameters
            ----------
            abort_event : Event
                triggers an abort for the sweep

            Returns
            -------
            result :
                the results from the charge state tuning stage
            '''

            result = ChargeStateTuning()
            return result

        return self.autotuning_thread.add_job(
                                              autotuning_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )
    
    def run_qubit_tuning(self,
                         sweep,
                         instrument_handler,
                         current_setpoints = {},
                         wait: bool = True,
                         timeout: float = 60
                         ):

        '''
        Description
        -----------
        Runs the sweep functions to conduct qubit tuning.

        Parameters
        ----------
        device_config : yaml file
            file containing the device information and setup
        instrument_handler : instance of handler
            instance of instruemnt handler to process and control instruments
        experiment_handler : instance of handler
            instance of experiment handler to process the autotuning steps in qubit tuning stage
        wait : bool, optional
            sets a delay to every job, defaults to True
        timeout : float, optional
            sets a time period to delay each job for, defaults to 60s
        
        Returns
        -------
        The sweep job for the qubit tuning stage.
        '''

        def autotuning_fn(abort_event):

            '''
            Description
            -----------
            Creates a sweep function to run the qubit tuning steps.

            Parameters
            ----------
            abort_event : Event
                triggers an abort for the sweep

            Returns
            -------
            result :
                the results from the qubit tuning stage
            '''

            result = QubitTuning()
            return result

        return self.autotuning_thread.add_job(
                                              autotuning_fn,
                                              args=(),
                                              wait=wait,
                                              timeout=timeout
                                             )