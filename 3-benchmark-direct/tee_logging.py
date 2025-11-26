import sys
import os
import datetime
import tempfile

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
def logging_context(prefix: str, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(log_dir, f"{prefix}_{ts}.log")
    log = open(path, "w")
    tee = Tee(log)
    with redirect_stdout(tee), redirect_stderr(tee):
        print(f"Logging started: {ts} → {path}")
        try:
            yield
        finally:
            print(f"Logging finished: {path}")
            log.close()

def main_test():
    print("\n===== tee_logging.main_test() =====")

    with tempfile.TemporaryDirectory() as d:
        with logging_context(d):
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
