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
from typing import Tuple, Dict, Any
from queue import Queue
from collections import deque
from enum import Enum

__BufferExists__ = False
__Instance__ = None


def create_buffer_instance(station : Station, station_lock : threading.Lock, time_func : Callable[[],Any] = time.monotonic):
    global __Instance__
    if __Instance__ is None:
        __Instance__ = buffered_readout(station, station_lock, time_func)

    return __Instance__

class inst_status(Enum):
            queued = "Queued"
            connected = "Connected"
            failed = "Failed"
            invalid = "Invalid"

class buffered_readout:
    def __init__(self, station : Station, station_lock : threading.Lock, time_func : Callable[[],Any] = time.monotonic):
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

        self.time_func = time_func

        self.BUFFER_SIZE = 1000
        
        self.READING_TIME = 0.01
        
        '''
        Define the circular buffer for storing the stream of data
        It has a lock for multi-threaded operation which protects the
        buffer and its current index.
        '''
        self.buffers : Dict[str, deque] = {}
        self.buffer_lock = threading.Lock()
    
        self.THREAD_NAME = "BufferThread"
        self.thread = threading.Thread(target = self._thread_loop, name = self.THREAD_NAME)
        self.running = False
        self.shutdown_event = threading.Event()

        self.station = station
        self.station_lock = station_lock

        self.instrument_load_queue = Queue() # no lock required, queue is thread safe

        # A list storing a tuple of the instrument name and a list of the
        # montitored paramters
        self.reading_parameters : List[Tuple[str, List[str]]] = []
        self.instruments : Dict[str, Instrument] = {}

        # Store the status of instruments. 
        self.instrument_status_lock = threading.Lock()
        self.instrument_status : Dict[str, inst_status] = {}

    def _assert_correct_thread(self):
        #return True
        assert threading.current_thread().name == self.THREAD_NAME, f"Error, buffer is being run in the thread '{threading.current_thread().name}'!"

    def _thread_loop(self):

        self._assert_correct_thread()


        print(f"Starting the readout buffer loop in thread {self.THREAD_NAME}...")

        while not self.shutdown_event.is_set():

            time.sleep(self.READING_TIME)
            self._add_queued_instruments()
            self._read_instruments()

        self._close_instruments()

        print("Readout buffer thread stopping...")

        return
    def _add_queued_instruments(self):
        '''
        Add all the instruments that may have been added to the queue
        '''
        self._assert_correct_thread()

        # Add all of the insturments in the queue
        while not self.instrument_load_queue.empty():
            inst : Instrument = None
            # Try to acquire a lock on the station
            if self.station_lock.acquire(timeout = 0.1):
                try:
                    name, param_names, init_func, init_args = self.instrument_load_queue.get()
                    
                    # Make sure that we report if loading the instrument fails
                    try:
                        # log
                        print(f"Attempting to load instrument {name}")
                        inst = self.station.load_instrument(name)

                        # Will throw an AttributeError if the params are bad
                        self._test_inst_has_parameters(inst, param_names) 
                    except Exception as e:
                        print(e) # log
                        if inst is not None:
                            inst.close_all()
                        inst = None
                        self._set_instrument_status(name, inst_status.failed)
                    else:
                        # log
                        print(f"Instrument loaded, idn: {inst.get_idn()}")
                        self._set_instrument_status(name, inst_status.connected)
                finally:
                    self.station_lock.release()
            else:
                break # Break from the loop if there is a timout or we cannot acquier the lock
            
            # Store the parameters and the instrument
            # This is not done earlier so that we can release the lock on the station sooner.
            if inst is not None:
                if init_func is not None:
                    init_func(inst, init_args)

                # init the buffer for the parameter
                if isinstance(param_names, str):
                    param_names = [param_names]

                self.reading_parameters.append((name, param_names))
                self.instruments[name] = inst
                # Create a buffer for the parameter
                with self.buffer_lock:
                    for param_name in param_names:
                        long_name = f"{name}.{param_name}"
                        self.buffers[long_name] = deque(maxlen = self.BUFFER_SIZE)

            self.instrument_load_queue.task_done()
    
    def _test_inst_has_parameters(self, inst : Instrument, param_names : str | List[str]) -> bool:
        '''
        Test if the instrument has the specified parameters. We just iterate through the parameters
        and try to access is
        '''
        if isinstance(param_names, str):
            param_names = [param_names]
        for param_name in param_names:
            param : Parameter = getattr(inst, param_name)
            if not param.gettable:
                raise AttributeError(f"The specified parameter {param_name} of instrument {inst.name} is not getable.")
    
    def _close_instruments(self):
        '''
        Only to be called by the buffer thread on shutdown. Acquires a lock on the station and
        then closes all of the loaded instruments.
        '''
        self._assert_correct_thread()
        # I am not certain if closing the instrument modifies the station,
        # but I think that it is likely. So we lock it to be safe
        with self.station_lock: 
            for name, inst in self.instruments.items():
                try:
                    # Log
                    print(f"Closing instrument {name}")
                    inst.close_all()
                except Exception as e:
                    print(e) # log
                
    def _read_instruments(self):
        value = random.Random(self.time_func()).random() # read the instrument!
        curr_time = self.time_func()
        
        # Acquire a lock on the circular buffer and push it to the current index
        with self.buffer_lock:
            for inst_name, param_names in self.reading_parameters:
                for param_name in param_names:
                    param : Parameter = getattr(self.instruments[inst_name], param_name)
                    value = param()
                    curr_time = self.time_func()
                    self.buffers[f"{inst_name}.{param_name}"].append((value, curr_time))
            
        
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
        # Make var_name iterable if it is not already
        if isinstance(var_name, str):
            var_name = [var_name]

        buffer_copies = {}
        # acquire a lock on the buffer for Readout, and then copy it
        with self.buffer_lock:
            for name in var_name:
                buffer_copies[name] = List[self.buffers[name]]

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

    def get_buffer(self, param_names : str | List[str] | None = None, timeout : float = 0.1) -> Dict[str, List[Tuple[float, float]]] | None:
        '''
        Try to copy the buffer without blocking. If it fails to acquire the lock,
        it will return None.

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
        '''
        buffer_copies = {}
        if self.buffer_lock.acquire(blocking = False):#, timeout = timeout):
            try:
                if param_names is None:
                    param_names = self.buffers.keys()
                elif isinstance(param_names, str):
                    param_names = [param_names]
                for name in param_names:
                    buffer = list(self.buffers[name])
                    buffer_copies[name] = buffer
            finally:
                self.buffer_lock.release()
        else:
            return None

        return buffer_copies if len(buffer_copies) > 0 else None

    def run(self):
        if not self.running:
            self.thread.start()
            self.running = True

    
    def join(self):
        self.shutdown_event.set()
        self.thread.join()


    def add_readout_instrument(self, name : str, param_names : str | List[str],\
                                init_func : Callable[[Instrument, Tuple], None] | None = None,\
                                init_args : Tuple = ()) -> None:
        '''
        Add a readout instrument to the asynchonous buffer. Adds these parameters to a queue,
        and the readout buffer thread will attempt to add the instrument in its control loop.
        If the readout thread cannot load the instrument, it will get logged. Trying to access
        an instrument that failed to add, or has not yet been added, will throw a KeyError exception.

        Parameters
        ----------
        name: str
            String for the name of the instrument in the qcodes station

        param_names: str | List[str]
            A string or list of strings with the name of the qcodes parameter 
            to measure from this instrument

        init_func: Callable[[Instrument, Tuple], None] | None = None
            A callback function of the form (Instrument, Tuple) -> None, called 
            after the instrument is loaded on the readout thread. The tuple is meant
            to be used to pass any arguments required for initalization.

        init_args : Tuple = ()
            The arguments to get passed to the init function. By default, it is an empty tuple.
        '''
        self.instrument_load_queue.put((name, param_names, init_func, init_args))
        self._set_instrument_status(name, inst_status.queued)

    def _set_instrument_status(self, instrument_name : str, status : inst_status):
        with self.instrument_status_lock:
            self.instrument_status[instrument_name] = status

    def get_instrument_status(self, instrument_name : str) -> inst_status:
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
        with self.instrument_status_lock:
            retval = self.instrument_status.get(instrument_name)
            if retval is None:
                return inst_status.invalid
            return retval
