import threading
import time
import numpy as np
from typing import List, Tuple
import random

__BufferExists__ = False
__Instance__ = None


def create_buffer_instance():
    global __Instance__
    if __Instance__ is None:
        __Instance__ = buffered_readout()

    return __Instance__


class buffered_readout:
    def __init__(self):
        '''
        A class to handle the asynchronous buffered readout of the SET current for
        autotuning devices.
        '''
        global __BufferExists__

        assert not __BufferExists__, "Error: Readout buffer already exists!!"

        __BufferExists__ = True

        self.time_func = time.time

        self.BUFFER_SIZE = 1000
        
        self.READING_TIME = 0.01
        
        '''
        Define the circular buffer for storing the stream of data
        It has a lock for multi-threaded operation which protects the
        buffer and its current index.
        '''
        self.buffer = [(float(0), float(-1.0))] * self.BUFFER_SIZE
        self.buffer_index = 0
        self.buffer_lock = threading.Lock()
    
        self.THREAD_NAME = "BufferThread"
        self.thread = threading.Thread(target = self.__thread_loop__, name = self.THREAD_NAME)
        self.running = False
        self.shutdown_event = threading.Event()

    def __assert_correct_thread__(self):
        #return True
        assert threading.current_thread().name == self.THREAD_NAME, f"Error, buffer is being run in the thread '{threading.current_thread().name}'!"

    def __open_instruments__(self):

        self.__assert_correct_thread__()

        return
    def __thread_loop__(self):

        self.__assert_correct_thread__()

        self.__open_instruments__()

        print(f"Starting the readout buffer loop in thread {self.THREAD_NAME}...")

        while not self.shutdown_event.is_set():

            time.sleep(self.READING_TIME)
            self.__read_instruments__()

        print(f"Stopping the readout buffer thread...")
        return

    def __read_instruments__(self):
        value = random.Random(self.time_func()).random() # read the instrument!
        curr_time = self.time_func()
        

        # Acquire a lock on the circular buffer and push it to the current index
        with self.buffer_lock:
            self.buffer[self.buffer_index] = (value, curr_time)
            self.buffer_index = (self.buffer_index + 1) % self.BUFFER_SIZE
        
    def read_buffer(self, t_avg : float = 0.0, t_start : float = 0.0) -> float:
        '''

        '''
        if t_start <= 0.0:
            t_start = time.time()

        # acquire a lock on the buffer for Readout, and then copy it
        with self.buffer_lock:
            buffer_copy = self.buffer.copy()

        # Sort according to the time stamp
        buffer_copy.sort(key = lambda e: e[1])
        
        i = self.BUFFER_SIZE - 1
        values : List[float] = [] 
        t_stop = t_start - t_avg
        while buffer_copy[i][1] >= t_stop:
            timestamp = buffer_copy[i][1]

            if timestamp <= t_start:
                values.append(buffer_copy[i][0])
        return float(np.average(values))
    def get_buffer(self) -> Tuple | None:
        '''
        Try to copy the buffer without blocking. If it fails to acquire the lock,
        it will return None.
        '''
        
        if self.buffer_lock.acquire(blocking = False):
            try:
                copy = self.buffer.copy()
            finally:
                self.buffer_lock.release()
        else:
            return None

        # Next return only valid time stamps
        copy.sort(key = lambda e : e[1])
        retval : List[float] = []
        timestamps : List[float] = []
        for value, timestamp in copy:
            if timestamp >= 0.0:
                retval.append(value)
                timestamps.append(timestamp)
        assert len(retval) == len(timestamps)
        return (retval, timestamps)

    def run(self):
        if not self.running:
            self.thread.start()
            self.running = True
    def join(self):
        self.shutdown_event.set()
        self.thread.join()
