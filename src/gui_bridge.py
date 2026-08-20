import queue

class TuningBridge:

    '''
    Description
    -----------
    This class is used to bridge the GUI and the tuning protocol. It will be used to pass figure data between the two, and to allow the GUI to add plots from the tuning protocol to the given tab.
    '''

    def __init__(self):
        self.plot_queue = queue.Queue()

# Creates an instance that will be imported
tuning_bridge = TuningBridge()