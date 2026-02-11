from typing import Optional, Union, List
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .leadfield import LeadField, ChannelLocation
from utils import start_matlab, EngineWrapper, with_matlab_engine

@with_matlab_engine
def lf_generate_from_pha(
    atlas : str,
    layout : str,
    montage : Optional[str] = None,
    labels: Optional[List[str]] = None,
    normalize : bool = False,
    verbose : bool = False,
    eng: EngineWrapper = None
):
    eng = start_matlab()
    try:
        pass
    finally:
        if verbose:
            print("Quitting Matlab engine...")
        eng.quit()