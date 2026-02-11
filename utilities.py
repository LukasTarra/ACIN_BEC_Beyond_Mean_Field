import time

class Timer:
    """
    Simple class structure for runtime checking of code.

    Use it like this:
    timer = Timer()
    timer.start()
    sim_time = timer.stop()
    """
    
    def __init__(self):
        """Initialize a new Timer instance with start_time and end_time set to None."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer by recording the current time using time.perf_counter()."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Stop the timer by recording the current time using time.perf_counter() and print+return the elapsed time."""
        self.end_time = time.perf_counter()
        print(f"Timer stopped after {self.elapsed()} seconds.")
        
        return self.elapsed()

    def elapsed(self):
        """Calculate the elapsed time since the timer was started.

        Returns:
            float: The elapsed time in seconds.
        Raises:
            ValueError: If the timer has not been started yet.
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time



