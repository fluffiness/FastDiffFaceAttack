import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.calls = 0
        self.start_time = 0.
        self.times = []


    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, hms=True):
        diff = time.time() - self.start_time
        self.times.append(diff)
        if hms:
            return sec2hms(diff)
        else:
            return diff
    
    def average(self, hms=True):
        if hms:
            return sec2hms(sum(self.times) / len(self.times))
        else:
            return sum(self.times) / len(self.times)

    def total(self, hms=True):
        if hms:
            return sec2hms(sum(self.times))
        else:
            return sum(self.times)

    def clear(self):
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.times = []


def sec2hms(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60
    hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return hms