'''
File: instrument_handler.py

Author: Mason Daub (mjdaub@uwaterloo.ca)

Provides an asynchronous readout buffer for a arbitrary number of qcodes instruments
and parameters.
'''
import threading
import time
import numpy as np
from typing import List, Tuple
from qcodes.station import Station
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from collections.abc import Callable
from typing import Tuple, Dict, Any, Literal, Protocol, Optional, Deque
from queue import Queue
from collections import deque
from dataclasses import dataclass
from enum import Enum
from tunerlog import TunerLog
import re

_BufferExists = False
_Instance = None

logger = TunerLog("Instr. Control")

def create_buffer_instance(station : Station, station_lock : threading.Lock):
    global _Instance, logger
    if _Instance is None:
        _Instance = instrument_handler(station, station_lock)

    return _Instance

def make_list(strings : str | List[str] | Literal['all']) -> List[str]:
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
    def __call__(self, instrument: Instrument, *args: Any) -> Any:
        ...

class TunerFuture:
    def __init__(self):
        self._done_event = threading.Event()
        self._result : Any = None
        self._exception : Exception | None = None

    def set_result(self, result : Any):
        self._result = result
        self._done_event.set()

    def set_exception(self, e : Exception):
        self._exception = e
        self._done_event.set()

    def result(self, timeout : float | None = None):
        if self._done_event.wait(timeout):
            if self._exception is not None:
                raise self._exception
            return self._result
        else:
            raise TimeoutError("Future timed out while waiting for result")
    
@dataclass
class instrument_job:
    future : TunerFuture
    when : float
    type : str

class instrument_callback_job(instrument_job):
    def __init__(self, future : TunerFuture, callback : InstrumentCallback, *args, when : float = -1):
        self.callback : Callable[[Instrument], Any] = lambda inst: callback(inst, *args)
        super().__init__(future, when, "instrument_callback")

class get_parameter_job(instrument_job):
    def __init__(self, future : TunerFuture, params : List[str], when : float = -1):
        super().__init__(future, when, "get parameter job")
        self.parameters = params

class set_parameter_job(instrument_job):
    def __init__(self, future : TunerFuture, set_vals : Dict[str, Any], when : float = -1):
        super().__init__(future, when, "set parameter job")
        self.set_vals = set_vals

class change_monitor_status_job(instrument_job):
    def __init__(self, future : TunerFuture, params : List[str], add : bool = True, when : float = -1):
        super().__init__(future, when, "modify monitor status job")
        self.parameters = params
        self.add_or_remove = add

class instrument_thread:
    def __init__(self, thread_name : str,\
                    instrument_name: str,\
                    station : Station,\
                    station_lock : threading.Lock,\
                    global_shutdown: threading.Event,\
                    init_func : Optional[InstrumentCallback] = None,\
                    *init_args : Any):
        
        self.parameters_private : List[str] = []
        self.parameters_public : List[str] = []
        self.parameters_lock = threading.Lock()
        self.job_queue : Queue[instrument_job] = Queue()

        self.BUFFER_SIZE = 1000
        self.buffer : Dict[str, Deque[Tuple[float, float]]] = {}
        self.buffer_lock = threading.Lock()
        self.measure_time = 0.01

        self.thread_name = thread_name
        self.thread = threading.Thread(target = self._worker,\
                                        name = self.thread_name,\
                                        args = (instrument_name, station, station_lock, init_func, *init_args))
        
        self.instrument : Instrument # DO NOT ACCESS EXTERNALLY
        
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
            logger.debug(f"Joining the instrument thread {self.thread_name}")
            self.shutdown_signal.set()
            try:
                self.thread.join(10)
            except TimeoutError as e:
                logger.exception("Joining thread '%s' timed out!", self.thread_name)
                raise e # propagate the exception
            else:
                logger.debug("Thread '%s' joined successfully", self.thread_name)

    def _update_status(self, status_string : str):
        '''
        Update the status of the thread. Private, do not call (thread safe).
        '''
        with self.status_lock:
            self.status = status_string
            logger.debug("Thread '%s' updating status to '%s'", self.thread_name, status_string)
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
            logger.info("Attemting to load instrument '%s'.", instrument_name)
            with station_lock:
                self.instrument = station.load_instrument(instrument_name)
            if not init_func is None:
                # if there is an incorrect number of arguments, init_func will raise a type error
                # There may be additional exceptions raised within init_func if they are not handled
                init_func(self.instrument, *init_args)
        except:
            # log the error
            logger.exception(f"Initialization exception for {instrument_name} with initialization function {init_func} and args {init_args}.")
            self.shutdown_signal.set() # Signal to the watchdog that the thread shut down instead of hung
            self._update_status("Failed to initialize")
            return # kill the thread.
        
        logger.info("Thread started for instrument '%s': %r", instrument_name, self.instrument)
        logger.instrument_snapshot(self.instrument)
        # Start the readout loop
        self._update_status("Running")
        
        loop_times = deque(maxlen = 500) # A deque for tracking the average loop time
        tprev = self.timefunc()
        while not self.shutdown_signal.is_set() and not self.global_shutdown.is_set():
            self._process_queue()
            self._read_parameters()

            tnow = self.timefunc()
            delta = tnow - tprev
            sleep_time = self.measure_time - delta
            if sleep_time > 0.001:
                time.sleep(self.measure_time)

            with self.heartbeat_lock:
                tprev = self.timefunc()
                loop_times.append(tprev - self.heartbeat)
                self.heartbeat = tprev
            
        loop_times = list(loop_times)
        average_dt = np.average(loop_times) * 1e3
        times_stdev = np.std(loop_times) * 1e3
        logger.debug("'%s' average loop time: %f +- %f ms (target %f ms)", self.thread_name, average_dt, times_stdev, self.measure_time * 1e3)
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
                param = self.getattr_recursive(self.instrument, param_name)
            except:
                logger.exception("Exception occured while reading parameter '%s.%s.'", self.instrument.name, param_name)
            else:
                value = param()
                timestamp = self.timefunc()

                self.buffer[param_name].append((value, timestamp))

    def _update_public_parameters(self):
        with self.parameters_lock:
            self.parameters_public = self.parameters_private.copy()

    def _process_queue(self):
        count = 2 # to prevent infinite loops with the when parameter of a job
        while self.job_queue.qsize() > 0 and count > 0:
            curr_time = time.monotonic()
            job = self.job_queue.get()
            logger.debug("Instrument '%s', processing job %r", self.instrument.name, job)

            if job.when > curr_time and (job.when > 0):
                self.job_queue.put(job)
                self.job_queue.task_done()
                count -= 1
                continue

            if isinstance(job, instrument_callback_job):
                try:
                    retval = job.callback(self.instrument)
                except Exception as e:
                    job.future.set_exception(e)
                else:
                    job.future.set_result(retval)
                    
            elif isinstance(job, get_parameter_job):
                params = job.parameters
                retval = {}
                for param in params:
                    try:
                        value = self.getattr_recursive(self.instrument, param)()
                    except Exception as e:
                        job.future.set_exception(e)
                    else:
                        retval[param] = value
                job.future.set_result(retval)

            elif isinstance(job, set_parameter_job):
                for param, setval in job.set_vals.items():
                    try:
                        self.getattr_recursive(self.instrument, param)(setval)
                    except Exception as e:
                        job.future.set_exception(e)
                job.future.set_result(None)
            
            elif isinstance(job, change_monitor_status_job):
                self._handle_monitor_status_job(job)

            return
    def getattr_recursive(self, obj, param : str):
        splitted = param.split('.', maxsplit = 1)
        attr = getattr(obj, splitted[0])
        if len(splitted) == 1:
            return attr
        else:
            return self.getattr_recursive(attr, splitted[1])
        
    def _handle_monitor_status_job(self, job : change_monitor_status_job) -> bool:
        if job.add_or_remove: # adding a monitored param
            for param in job.parameters:
                already_monitored = param in self.parameters_private
                if already_monitored:
                    continue
                try:
                    # Test to make sure the requested parameter exists
                    qparam = self.getattr_recursive(self.instrument, param)
                except Exception as e:
                    job.future.set_exception(e)
                    self._update_public_parameters()
                    return False
                else:
                    # Add a new deque to the buffer dictionary if required
                    self.parameters_private.append(param)
                    logger.info("Now monitoring parameter %s.%s", self.instrument.name, param)
                    with self.buffer_lock:
                        if not param in self.buffer:
                            self.buffer[param] = deque(maxlen = self.BUFFER_SIZE)

        else: # Remove a monitored param
            for param in job.parameters:
                already_monitored = param in self.parameters_private
                if not already_monitored:
                    continue
                try:
                    self.parameters_private.remove(param)
                except Exception as e:
                    job.future.set_exception(e)
                    self._update_public_parameters()
                    return False
                else:
                    logger.info("No longer monitoring parameter %s.%s", self.instrument.name, param)
                
        self._update_public_parameters()
        job.future.set_result(None)
        return True               
            

class instrument_handler:
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
        global _BufferExists

        assert not _BufferExists, "Error: Readout buffer already exists!!"

        _BufferExists = True

        self.instrument_threads : Dict[str, instrument_thread] = {}

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
            t_stop = time.monotonic()

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

    def get_buffer(self, param_names : str | List[str] | Literal['all'] = 'all', blocking : bool= False, timeout : float = 0.1) -> Dict[str, List[Tuple[float, float]]]:
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
        if param_names == 'all':
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

        return buffer_copies

    def shutdown_instruments(self):
        self.global_shutdown.set()
        for inst_thread in self.instrument_threads.values():
            inst_thread.stop()


    def add_instrument(self, name : str,\
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
        # Check to see if the instrument already exists
        instr_thread = self.instrument_threads.get(name)
        if instr_thread is None:
            self.heartbeats[name] = time.monotonic() # Create the first heartbeat

            self.instrument_threads[name] = instrument_thread(f"{name}Thread", name,\
                                                        self.station,\
                                                        self.station_lock,\
                                                        self.global_shutdown,\
                                                        init_func, *init_args)

            self.instrument_threads[name].start()

    def add_callback(self, instrument : str,
                     callback : InstrumentCallback,
                     *args,
                     wait : bool = True,
                     timeout : float = 60,
                     when : float = -1) -> Any:
        """Add a callback function to an instruments job queue.

        Args:
            instrument (str): _description_
            callback (InstrumentCallback): _description_
            wait (bool, optional): _description_. Defaults to True.
            timeout (float, optional): _description_. Defaults to 60.
            when (float, optional): When (in absolute time with time.monotonic) to do the callback. 
                By default, it will execute as soon as it gets to the front of the queue.
            
        Returns:
            Any: If the instrument does not exist, it will return None. If wait is true,
                it will return the result of the callback. If wait is false, it will return a future.
        """
        inst = self.instrument_threads.get(instrument)
        if inst is not None:
            future = TunerFuture()
            job = instrument_callback_job(future, callback, *args, when = when)
            inst.job_queue.put(job)

            if wait:
                return future.result(timeout)
            else:
                return future
        else:
            return None

    def get_parameter(self, instrument : str, 
                      params : str | List[str],
                      wait : bool= True,
                      timeout : float = 60,
                      when = -1) -> Any:
        """Get parameters from an instrument

        Args:
            instrument (str): The name of the instrument to read from
            params (str | List[str]): The name of the parameter(s) to read
            wait (bool, optional): Whether or not to wait for the get command to complete. Defaults to True
            timeout (float | None, optional): Timeout for waiting, defaults to 60s

        Returns:
            Any: If wait is True, then it will return a dictionary of the gotten parameters.
                If wait is False, it will return the Future for the job. If the instrument name is
                invalid, it will return None.
        """
        params = make_list(params)

        inst = self.instrument_threads.get(instrument)
        if inst is not None:
            future = TunerFuture()
            job = get_parameter_job(future, params, when)
            inst.job_queue.put(job)

            if wait:
                return future.result(timeout)
            else:
                return future
        return None
    
    def set_parameter(self, instrument : str,
                      set_vals : Dict[str, Any],
                      wait : bool = True,
                      timeout : float = 60,
                      when : float = -1) -> bool | TunerFuture:
        """Set one or more parameters of an instrument.

        Args:
            instrument (str): _description_
            set_vals (Dict[str, Any]): A dictionary of the parameter names and the value you want to set it to.
                For example {'dac1': 1.0, 'dac2': 2.0}.
            wait (bool, optional): Wait for operation to complete if True. Defaults to True.
            timeout (float, optional): Timeout for waiting. Defaults to 60s.

        Returns:
            bool | Future: Returns false on error and true on success. If wait is False, it will return the
                future.
        """
        inst = self.instrument_threads.get(instrument)
        if inst is not None:

            future = TunerFuture()
            job = set_parameter_job(future = future, set_vals = set_vals, when=when)
            inst.job_queue.put(job)
            if wait:
                return future.result(timeout)
            else:
                return future
        return False

    def remove_instrument(self, name : str, finish_jobs : bool = True):
        '''
        Stop a specific instrument and close it. This will not delete any of the data that is still buffered.
        '''
        if name in self.instrument_threads:
            thread = self.instrument_threads[name]
            if finish_jobs:
                thread.job_queue.join()
            thread.stop()
    
    def monitor_parameter(self, inst_name : str, 
                          params : str | List[str],
                          remove = False,
                          wait : bool = True, 
                          timeout : float = 60,
                          when : float = -1) -> bool | TunerFuture:
        inst = self.instrument_threads.get(inst_name)
        if not inst is None:
            params = make_list(params)
            future = TunerFuture()
            job =change_monitor_status_job(future, params, not remove, when)

            inst.job_queue.put(job)

            if wait:
                return future.result(timeout)
            else:
                return future
        else:
            logger.warning("Cannot add parameters to instrument '%s': It does not exist!", inst_name)
        return False
        
    def stop_monitoring_parameter(self, inst_name : str, 
                          params : str | List[str], 
                          wait : bool = True, 
                          timeout : float = 60,
                          when : float = -1) -> bool | TunerFuture:
        return self.monitor_parameter(inst_name, params, True, wait, timeout, when)

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
        inst_thread : instrument_thread | None = self.instrument_threads.get(instrument_name)
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
                logger.warning(f"Instrument thread {name} is dead with status {thread_status}")
        return True