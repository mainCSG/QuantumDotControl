import queue

class TuningBridge:
    def __init__(self):
        self.plot_queue = queue.Queue()

tuning_bridge = TuningBridge()