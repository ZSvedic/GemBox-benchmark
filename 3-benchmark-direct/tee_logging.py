# Python stdlib.
import contextlib as cl
import datetime
import os
import sys
import tempfile

# Tee class and its context manager.

class Tee:
    '''Mirrors stdout/stderr to a log file while still printing to terminal.'''

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

@cl.contextmanager
def logging_context(prefix: str = "noname", log_dir: str = "logs"):
    '''Creates timestamped log file and tee all output inside the context.'''
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(log_dir, f"{prefix}_{ts}.log")
    log = open(path, "w")
    tee = Tee(log)
    with cl.redirect_stdout(tee), cl.redirect_stderr(tee):
        print(f"Logging started: {ts} → {path}")
        try:
            yield
        finally:
            print(f"Logging finished: {path}")
            log.close()

# Main test.

def main_test():
    print("\n===== tee_logging.main_test() =====")

    with tempfile.TemporaryDirectory() as d:
        with logging_context("tee-test", d):
            print("hello tee")

        files = [f for f in os.listdir(d) if f.endswith(".log")]
        assert len(files)==1, "FAIL: Wrong number of log files."
        
        p = os.path.join(d, files[0])
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
            assert ("Logging started:" in txt and 
                    "hello tee" in txt and 
                    "Logging finished:" in txt), "FAIL: Log file content incorrect."
        
        print("PASS: Tee logging works correctly.")

if __name__ == "__main__":
    main_test()
