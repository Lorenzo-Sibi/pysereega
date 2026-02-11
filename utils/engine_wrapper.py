import os
from functools import wraps
from typing import Optional, Union, Callable
from pathlib import Path
from matlab import engine


def start_matlab(working_dir=None):
    """Start MATLAB engine with optional working directory."""
    with TempWD(working_dir or Path.cwd()):
        eng = engine.start_matlab()
    return eng

class EngineWrapper():
    """
    Wraps a matlab.engine instance to do lazy loading.
    
    The engine is only started when first accessed, avoiding ~1 second
    startup time during module import.
    
    Parameters
    ----------
    working_dir : Path or str, optional
        Working directory for MATLAB engine
        
    Examples
    --------
    >>> wrapper = EngineWrapper()
    >>> # Engine not yet started
    >>> result = wrapper.eval("2+2")  # Engine starts here
    >>> # Subsequent calls use the same engine instance
    """

    def __init__(self, working_dir : Optional[Union[Path, str]] = None):
        self.eng=None
        self.is_started=False
        self.working_dir = working_dir

    def __getattr__(self, item):
        if item in ('eng', 'is_started', 'working_dir'):
            return object.__getattribute__(self, item)
        # if engine not running:
        if self.is_started is False:
            self.eng=start_matlab(working_dir=self.working_dir)
            self.is_started = True
        return getattr(self.eng, item)
    
    def shutdown_engine(self):
        if self.is_started:
            self.quit()
            self.is_started = False

    def __del__(self):
        if self.is_started:
            try: 
                self.quit()
            except:
                pass

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


_default_engine = EngineWrapper() # This is the global (default) engine wrapper instance


def with_matlab_engine(func : Callable = None, engine_wrapper : EngineWrapper = None):
    """
    Decorator that injects a MATLAB engine into functions.
    
    Parameters
    ----------
    func : callable, optional
        Function to decorate (when used without arguments)
    engine_wrapper : EngineWrapper
        The engine wrapper to inject
        
    Returns
    -------
    decorator : callable
        Decorator function
        
    Examples
    --------
    >>> _default_engine = EngineWrapper()
    >>> 
    >>> @with_matlab_engine(_default_engine)
    >>> def my_function(x, eng):
    >>>     return eng.eval(f"sqrt({x})")
    >>> 
    >>> # 'eng' is automatically provided
    >>> result = my_function(16)
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'eng' not in kwargs:
                kwargs['eng'] = engine_wrapper or _default_engine
            return f(*args, **kwargs)
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator