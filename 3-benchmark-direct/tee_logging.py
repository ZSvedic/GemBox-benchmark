import sys, os, datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr

class Tee:
    def __init__(self, log_file):
        self.log = log_file
    def write(self, data):
        sys.__stdout__.write(data)
        sys.__stdout__.flush()
        self.log.write(data)
        self.log.flush()
    def flush(self):
        sys.__stdout__.flush()
        self.log.flush()

@contextmanager
def logging_context(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(log_dir, f"run_{ts}.log")
    log = open(path, "w")
    tee = Tee(log)
    with redirect_stdout(tee), redirect_stderr(tee):
        print(f"Logging started: {ts} → {path}")
        try:
            yield
        finally:
            print(f"Logging finished: {path}")
            log.close()
