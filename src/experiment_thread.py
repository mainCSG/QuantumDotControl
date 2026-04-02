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

class ExperimentThread:


    def __init__(self):


        self.job_event = threading.Event()
        self.abort_event =  threading.Event()
        self.shutdown_event = threading.Event()
        self.job_queue = PriorityQueue()
        self.THREAD_NAME = "experimental_thread"
        self.thread = threading.Thread(target = self.__thread_loop__, name = self.THREAD_NAME)

    def run(self):

        self.thread.start()
    
    def join(self):

        self.thread.join()
    
    def __assert_correct_thread__(self):

        assert threading.current_thread().name == self.THREAD_NAME, f"The current thread, {threading.current_thread().name}, is not the Experiment Thread." 

    def add_job(self,
                f: callable,
                args,
                priority: int = 1):

        self.job_queue.put((priority,(f, args)))
    
    def abort(self):

        self.abort_event.set()

    
    def __thread_loop__(self, job):

        while not self.shutdown_event.set():

            self.job_event.wait(timeout = 1)

            if self.job_queue.qsize() > 0:
                priority, data = self.job_queue.get()

                f, args = data

                f(*args, self.abort_event)

                self.job_queue.task_done()

                while self.abort_event.is_set():
                    self.job_queue.get()

