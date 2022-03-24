import time


class TimeElapsed(object):
    def __init__(self, header=''):
        self.header = header
        self.counter = 1
        self.start = time.time()

    def restart(self):
        self.start = time.time()
        self.counter = 1

    def elapsed_str(self):
        end = time.time()
        time_str = f'{self.counter:3d}> {end - self.start :.5f}'
        self.start = end
        self.counter += 1
        return time_str
