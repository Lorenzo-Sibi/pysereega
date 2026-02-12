"""
Lead field management for SEREEGA-py.

This module provides functions to load, generate, and manipulate lead fields
(forward models) for EEG source simulation.

Available lead field sources:
- New York Head (ICBM-NY): Pre-computed adult head model
- MNE-Python TODO
- Pediatric Head Atlas TODO
- HArtMuT TODO
- Custom MATLAB files TODO

Quick Start
-----------
>>> from sereega_py.leadfield import generate_from_nyhead
>>> 
>>> # Generate from New York Head (requires download)
>>> lf = generate_from_nyhead(montage='S64')
>>> 
>>> # Inspect lead field
>>> print(f"Channels: {lf.n_channels}, Sources: {lf.n_sources}")
>>> print(f"Channel labels: {lf.channel_labels[:5]}...")
"""

from .leadfield import LeadField, ChannelLocation

# MATLAB loaders (require pre-downloaded data)
from .nyhead import (
    lf_generate_from_nyhead,
    NYHEAD_LEADFIELD_PATH,
    NYHEAD_WORKING_DIR
)


__all__ = [
    # Core classes
    'LeadField',
    'ChannelLocation',
    
    # MATLAB loaders
    'lf_generate_from_nyhead',
    
]

# Version info
__version__ = '0.1.0'
__author__ = 'SEREEGA-py Lorenzo Sibi'