'''
File: buffered_readout.py

Author: Mason Daub (mjdaub@uwaterloo.ca)

Provides an asynchronous readout buffer for a arbitrary number of qcodes instruments
and parameters.
'''
import threading
import time
import numpy as np
from typing import List, Tuple
import random
from qcodes.station import Station
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from collections.abc import Callable
from typing import Tuple, Dict, Any, Literal, Protocol, Optional, Deque
from queue import Queue
from collections import deque
from enum import Enum
import re
import sys, os

__BufferExists__ = False
__Instance__ = None


def create_buffer_instance(station : Station, station_lock : threading.Lock):
    global __Instance__
    if __Instance__ is None:
        __Instance__ = buffered_readout(station, station_lock)

    return __Instance__

def make_list(strings : str | List[str] | None) -> List[str] | None:
    if isinstance(strings, str):
        return [strings]
    else:
        return strings

class inst_status(Enum):
            queued = "Queued"
            connected = "Connected"
            failed = "Failed"
            invalid = "Invalid"

class InstrumentCallback(Protocol):
    def __call__(self, instrument: Instrument, *args: Any) -> None:
        ...

class buffer_thread:
    def __init__(self, thread_name : str,\
                    instrument_name : str,\
                    station : Station,\
                    station_lock : threading.Lock,\
                    global_shutdown: threading.Event,\
                    init_func : Optional[InstrumentCallback] = None,
                    *init_args : Any):
        
        self.parameters_private : List[str] = []
        self.parameters_public : List[str] = []
        self.parameters_lock = threading.Lock()
        self.param_queue = Queue()

        self.BUFFER_SIZE = 1000
        self.buffer : Dict[str, Deque[Tuple[float, float]]] = {}
        self.buffer_lock = threading.Lock()
        self.measure_time = 0.01

        self.thread_name = thread_name
        self.thread = threading.Thread(target = self._worker,\
                                        name = self.thread_name,\
                                        args = (instrument_name, station, station_lock, init_func, init_args))
        
        self.instrument : Instrument = None # DO NOT ACCESS EXTERNALLY
        
        self.shutdown_signal = threading.Event()
        self.global_shutdown = global_shutdown

        # heartbeat signal for the watchdog timer
        self.heartbeat : float = -1.0
        self.heartbeat_lock = threading.Lock()

        self.timefunc : Callable[[],float] = time.monotonic

        self.status : str = "Not started"
        self.status_lock  = threading.Lock()

    def start(self):
        self.thread.start()

    def stop(self):
        status = self.get_status()
        if status == "Running":
            # log
            print(f"Stopping the instrument thread {self.thread_name}")
            self.shutdown_signal.set()
            self.thread.join()

    def _update_status(self, status_string : str):
        '''
        Update the status of the thread. Private, do not call (thread safe).
        '''
        with self.status_lock:
            self.status = status_string
    def get_status(self) -> str:
        '''
        Get current status of the instrument buffer thread (thread safe).
        '''
        with self.status_lock:
            return self.status
    def get_heartbeat(self) -> float:
        with self.heartbeat_lock:
            return self.heartbeat
    
    def _worker(self, instrument_name : str, station : Station,\
                station_lock : threading.Lock,\
                init_func : Optional[InstrumentCallback] = None,\
                *init_args : Any):
        '''
        The worker function that controls the asnychonous buffering of a single instrument. (Not thread safe)

        Parameters
        ----------
        instrument_name : str
            Name of the instrument in the station configuration file. This is the instrument that the worker
            will attempt to load onto this thread.
        station : Station
            The station to load the instument from. This will cause the config file to get reloaded, so it 
            must have a lock passed with it
        station_lock : threading.Lock
            The lock to protect the station.
        init_func : Optional[InstrumentCallback] = None
            An InstrumentCallback function that is called after loading the instrument. Allows for settings to be
            set from the worker thread. No init function by default
        *init_args : Any
            An args list that is passed to the initialization callback.
        '''
        # Start the first heartbeat for the watchdog
        with self.heartbeat_lock:
            self.heartbeat = self.timefunc()
        self._update_status("Initializing")
        # Try to load the instrument
        try:
            with station_lock:
                self.instrument = station.load_instrument(instrument_name)
            if not init_func is None:
                # if there is an incorrect number of arguments, init_func will raise a type error
                # There may be additional exceptions raised within init_func if they are not handled
                init_func(self.instrument, *init_args)
        except Exception as e:
            # log the error
            print(f"Worker exception for {instrument_name} with initialization function {init_func} and args {init_args}.\n {e}")
            self.shutdown_signal.set() # Signal to the watchdog that the thread shut down instead of hung
            self._update_status("Failed to initialize")
            return # kill the thread.
        
        # Start the readout loop
        self._update_status("Running")
        tprev = self.timefunc()
        while not self.shutdown_signal.is_set() and not self.global_shutdown.is_set():
            
            self._process_queue()
            self._read_parameters()
            tnow : float
            with self.heartbeat_lock:
                tnow = self.timefunc()
                self.heartbeat = tnow
            delta = tnow - tprev
            sleep_time = self.measure_time - delta
            if sleep_time > 0.001:
                time.sleep(self.measure_time)
            tprev = self.timefunc()

        # Clear the monitored parameters
        with self.parameters_lock:
            self.parameters_private = []
            self.parameters_public = []

        # Shutdown the thread. Set status to stopped now in case of exception on close
        self._update_status("Stopped")
        if self.instrument is not None:
            self.instrument.close_all()

    def _read_parameters(self):
        for param_name in self.parameters_private:
            param : Parameter
            try:
                param = getattr(self.instrument, param_name)
            except Exception as e:
                print(f"Exception while reading parameter: {e}")
            else:
                value = param()
                timestamp = self.timefunc()

                self.buffer[param_name].append((value, timestamp))
        
    def _process_queue(self):
        if not self.param_queue.empty():
            op, name = self.param_queue.get()
            name_in_params = name in self.parameters_private
            if op == "add" and not name_in_params:
                try:
                    param = getattr(self.instrument, name)
                    if not param.gettable:
                        raise Exception(f"Parameter {self.instrument.name}.{name} is not gettable!")
                except KeyError as e:
                    print(f"Key error for parameter {self.instrument.name}.{name}! {e}")
                except Exception as e:
                    # log the exceptions (keyError from getattr, or the manual exception)
                    print(e)
                else:
                    self.parameters_private.append(name) # Add the name to the list of monitored params
                    # add a new buffer entry if there is not already one.
                    with self.buffer_lock:
                        if not name in self.buffer:
                            self.buffer[name] = deque(maxlen =self.BUFFER_SIZE)

            elif op == 'add' and name_in_params:
                # log
                print(f"Parameter {self.instrument.name}.{name} is already being monitored.")
            elif op == 'remove' and name_in_params:
                self.parameters_private.remove(name)
                print(f"Parameter {self.instrument.name}.{name} is no longer being monitored.")
            elif op == 'callback':
                # in this case name is actually a lambda function.
                try:
                    name(self.instrument)
                except Exception as e:
                    print(f"Exception with callback function {name}: {e}")
            
            # Update the copy
            with self.parameters_lock:
                self.parameters_public = self.parameters_private.copy()
            


    def queue_parameters(self, param_names : List[str], operation : Literal["add", "remove"] = "add") -> int:
        '''
        Communicates to the thread that you would like to try to add or remove a parameter from the list
        of measured parameters.

        Parameters
        ---------
        param_names : str | List[str]
            A name or list of parameter names belonging to this instrument to add.

        operation: "add" or "remove"
            Specifies whether to add or remove the requested parameter from the list of measured parameters

        Returns
        -------
        Returns the number of parameters that are attempting to be added.
        '''
        n = 0
        with self.parameters_lock:
            for param_name in param_names:
                if not param_name in self.parameters_public:
                    self.param_queue.put((operation, param_name))
                    n += 1
        return n
    
    def queue_instrument_callback(self, callback : InstrumentCallback, *args):
        self.param_queue.put(("callback", lambda inst: callback(inst, *args)))

class buffered_readout:
    def __init__(self, station : Station, station_lock : threading.Lock):
        '''
        A class to handle the asynchronous buffered readout of the SET current for
        autotuning devices. Instruments can be added from the staton by calling the 
        method add_readout_instrument, where you specify the instrument, the parameters
        you want to monitor, and an initialization callback (if desired)

        Parameters
        ----------
        station : Station
            The qcodes station with the config file loaded. Instruments will be loaded using this
        
        station_lock : threading.Lock
            qcodes unfortunately writes to the station class during load_instrument, so we
            need a lock to protect the station and make calls to it thread safe.

        time_func : Callable[[], Any]
            A function with no arguments that returns some sort of time class. For now this is set to
            time.monotonic to guarantee a monotonic increase in time.
        '''
        global __BufferExists__

        assert not __BufferExists__, "Error: Readout buffer already exists!!"

        __BufferExists__ = True

        self.instrument_threads : Dict[str, buffer_thread] = {}

        self.heartbeats : Dict[str, float] = {}

        self.station = station
        self.station_lock = station_lock

        self.global_shutdown = threading.Event() # A global shutdown signal for all child threads.

        self.monitored_parameters : List[str] = []
        
    def read_buffer(self, var_name : str | List[str], t_avg : float = 0.0, t_stop : float = -1) -> Dict[str, float]:
        '''
        Sample one or more of the asynchronous instrument buffers, and average over a specified amount of time

        Parameters
        ----------
        var_name: str | List[str]
            A string or list of strings of the format '{inst_name}.{param_name}', specifying
            the parameter you would like to read
        t_avg: float = 0
            A float specifying the total amount of averaging time to return
        t_stop: float = -1
            The absolute stop time of when the averging should stop at. The averageing will sample
            the times t in [t_stop - t_avg, t_stop]. If t_stop is less than zero, the current time
            is used.

        Returns
        -------
        retval : Dict[str, float]
            Returns a dictonary of the var_name with its associated value

        Exceptions
        ----------
        KeyError:
            If the specified parameters are not in the dictionary of buffers, it will throw a KeyError exception
        Exception:
            If there are no data points in the perscribed averaging range
        '''
        if t_stop <= 0.0:
            t_stop = self.time_func()

        t_start = t_stop - t_avg
        
        buffer_copies = self.get_buffer(var_name, blocking = True)

        retval : Dict[str, float] = {}

        for name, data in buffer_copies.items():
            # If no averaging is specified, return last data point
            if t_avg == 0.0:
                val, timestamp = data[-1]
                retval[name] = val
            # otherwise, get all of the data points for the correct times
            else:
                sum = 0.0
                n : int = 0
                for i in range(len(data) - 1, 0, -1):
                    value, timestamp = data[i]
                    if timestamp >= t_start and timestamp <= t_stop:
                        sum += value
                        n += 1
                if n == 0:
                    raise Exception(f"Failed to read {name} over the interval {t_start}:{t_stop} (no samples)")
                else:
                    retval[name] = sum / n
        
        return retval

    def get_buffer(self, param_names : str | List[str] | None = None, blocking : bool= False, timeout : float = 0.1) -> Dict[str, List[Tuple[float, float]]] | None:
        '''
        Try to copy a buffer for a parameter with optional blocking

        Parameters
        ----------
        param_names: str | List[str] | None = None
            A list of the parameter names that you wish to get a copy of the buffer for. 
            If None, then it will return a dictionary of all of the buffers.

        Return
        ------
        retval : Dict[str, List[Tuple[float, float]]] | None
            Will return a dictionary of the requested parameters with a copy of the buffer. If it fails to 
            obtain any buffers, it will return None. The buffer is a list of tuples of the value (0) and the timestamp (1)

        Exceptions
        ----------
        KeyError
            May throw a KeyError exception if a bad name is given
        ValueError
            This will get thrown if param_names are not able to be parsed
        '''
        buffer_copies = {}
        if param_names is None:
            param_names = self.monitored_parameters
        param_names = make_list(param_names)
        for param_name in param_names:
            match = re.match(r'^(\w+)\.(\w+)$', param_name)
            inst_name : str
            param_name : str
            if match:
                inst_name, param_short = match.groups()
            else:
                # log
                raise ValueError(f"Read_Buffer: Failed to parse parameter {param_name}")
            
            thread = self.instrument_threads.get(inst_name)
            if not thread is None:
                if thread.buffer_lock.acquire(blocking = blocking):
                    # copy the buffer
                    # May throw a KeyError exception
                    try:
                        buffer_copies[param_name] = list(thread.buffer[param_short])
                    finally:
                        thread.buffer_lock.release()

        return buffer_copies if len(buffer_copies) > 0 else None

    def shutdown_instruments(self):
        self.global_shutdown.set()
        for inst_thread in self.instrument_threads.values():
            inst_thread.stop()


    def add_instrument(self, name : str, param_names : str | List[str] | None = None,\
                                init_func : Optional[InstrumentCallback] = None,\
                                *init_args : Any) -> None:
        '''
        Add a readout instrument to the asynchonous buffer. Adds these parameters to a queue,
        and the readout buffer thread will attempt to add the instrument in its control loop.
        If the readout thread cannot load the instrument, it will get logged. Trying to access
        an instrument that failed to add, or has not yet been added, will throw a KeyError exception.

        Parameters
        ----------
        name: str
            String for the name of the instrument in the qcodes station

        param_names: str | List[str] | None = None
            A string or list of strings with the name of the qcodes parameter 
            to measure from this instrument

        init_func: Callable[[Instrument, Tuple], None] | None = None
            A callback function of the form (Instrument, Tuple) -> None, called 
            after the instrument is loaded on the readout thread. The tuple is meant
            to be used to pass any arguments required for initalization.

        init_args : Tuple = ()
            The arguments to get passed to the init function. By default, it is an empty tuple.
        '''
        param_names = make_list(param_names) # Make it iterable

        # Check to see if the instrument already exists
        instr_thread = self.instrument_threads.get(name)
        if instr_thread is None:
            self.heartbeats[name] = time.monotonic() # Create the first heartbeat

            self.instrument_threads[name] = buffer_thread(f"{name}Thread", name,\
                                                        self.station,\
                                                        self.station_lock,\
                                                        self.global_shutdown,\
                                                        init_func, init_args)
            if param_names is not None:
                self.instrument_threads[name].queue_parameters(param_names)

            self.instrument_threads[name].start()
    
    def remove_instrument(self, name : str):
        '''
        Stop a specific instrument and close it. This will not delete any of the data that is still buffered.
        '''
        if name in self.instrument_threads:
            thread = self.instrument_threads[name]
            thread.stop()
    
    def add_parameter(self, inst_name : str, params : str | List[str]):
        inst = self.instrument_threads.get(inst_name)
        if not inst is None:
            params = make_list(params)
            inst.queue_parameters(params)
        else:
            # log 
            print(f"Cannot add parameter to instrument {inst_name}: it does not exist!")
        
    def remove_parameter(self, inst_name : str, params : str | List[str]):
        inst = self.instrument_threads.get(inst_name)
        if not inst is None:
            params = make_list(params)
            inst.queue_parameters(params, 'remove')
        else:
            # log 
            print(f"Cannot remove parameter from instrument {inst_name}: it does not exist!")

    def get_instrument_status(self, instrument_name : str) -> str:
        '''
        Query the readout buffer for the status of an instrument. The buffer automatically 
        updates the status of each instrument.

        Paramters
        ---------
        instrument_name : str
            The name of the instrument in the config file
        
        Returns
        -------
        retval : inst_status
            Returns a inst_status enumeration (which is just a string) to describe the 
            current status of the instrument. 
        '''
        inst_thread : buffer_thread = self.instrument_threads.get(instrument_name)
        if not inst_thread is None:
            return inst_thread.get_status()
        else:
            return "DNE"
        
    def watchdog(self) -> bool:
        '''
        This is the watchdog timer callback for the buffered_readout object. It 
        keeps track of the status of all of the threads and checks if they still have a heartbeat.

        Returns
        -------
        If the readout buffer is healthy, it returns true. If the readout buffer is not healthy, it will return
        false to trigger a reset.

        '''
        WATCHDOG_TIME = 60 # time between heartbeats before death is declared.
        self.monitored_parameters.clear()
        
        for name, thread in self.instrument_threads.items():
            heartbeat = thread.get_heartbeat()
            current_time = time.monotonic()
            thread_status = thread.get_status()

            # if the thread takes longer than the watchdog time, it's dead.
            alive = True
            if current_time - heartbeat > WATCHDOG_TIME:
                alive = False

            if thread_status == 'Running' and alive:
                with thread.parameters_lock:
                    for param in thread.parameters_public:
                        self.monitored_parameters.append(f"{name}.{param}")
            
            elif not alive and (thread_status == 'Running' or thread_status == 'Initializing'):
                print(f"Instrument thread {name} is dead with status {thread_status}")
                #return False
        return True
