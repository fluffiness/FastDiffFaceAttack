from datetime import datetime
from pathlib import Path
import os
import argparse

class DatetimeLogger:
    """
    Given a log directory, creates a subdirectory named after the current datetime, and stores the log file within.
    Can log argparse.Namespace objects and strings
    """

    def __init__(self, log_dir):
        current_datetime = datetime.now()
        formatted_datetime_string = current_datetime.strftime("%m-%d_%H:%M")

        self.log_dir = os.path.join(log_dir, formatted_datetime_string)
        log_file = os.path.join(self.log_dir, f"logs_{formatted_datetime_string}.txt")
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True, parents=True)

        with open(self.log_file, 'w') as file:
            file.write(current_datetime.strftime("%Y/%m/%d %H:%M:%S"))
    
    def log_args(self, args: argparse.Namespace):
        """used for logging an argparse.Namespace object"""
        with open(self.log_file, 'a') as file:
            for k in vars(args):
                line = f"{k:<20}: {getattr(args, k)}\n"
                file.write(line)
    
    def log(self, line: str='\n', out=True):
        if out:
            print(line)
        with open(self.log_file, 'a') as file:
            file.write(line + '\n')
    

if __name__ == "__main__":
    log_dir = "./logs"
    logger = DatetimeLogger(log_dir)
        
        
        