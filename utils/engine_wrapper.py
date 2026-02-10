import os
from matlab import engine
from pathlib import Path

def start_matlab(working_dir=None):
    """Start MATLAB engine with optional working directory."""
    with TempWD(working_dir or Path.cwd()):
        eng = engine.start_matlab()
    return eng

class EngineWrapper():
    """Wraps a matlab.engine() instance to do lazy loading"""

    def __init__(self):
        self.eng=None
        self.is_started=False

    def __getattr__(self, item):
        if item=='is_started':
            return self.is_started
        # if engine not running:
        if self.is_started is False:
            self.eng=start_matlab()
            self.is_started = True
        return getattr(self.eng, item)
    
    def shutdown_engine(self):
        if self.is_started:
            self.quit()
            self.is_started = False

    def __del__(self):
        if self.is_started:
            self.quit()

class TempWD():
    """context manager to temporarily switch cwd to `dir_path` 
    (https://gist.github.com/ganileni/c32a2fe0df8ffcd02b0c451c55e63c95#file-wrapping-ipynb)"""
    
    def __init__(self, dir_path):
        self.cwd = Path.cwd()
        self.dir_path = Path(dir_path)
        os.chdir(self.dir_path.resolve())
        
    def __enter__(self):
        return None
    
    def __exit__(self, type, value, traceback):
        os.chdir(self.cwd.resolve())


