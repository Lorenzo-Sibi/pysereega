"""
Base classes for signal generation.

This module defines the abstract base classes and interfaces for all signal types
in the pysereega library. All concrete signal implementations must inherit from
this base class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np


class Signal(ABC):
    """
    Abstract base class for all signal types (Noise, ERPSignal, ...).
    
    This class defines the interface that all signal types must implement.
    Signals represent different types of neural activity that can be simulated
    and projected through a lead field to generate scalp EEG data.
    
    All concrete signal classes must implement:
    - generate(): Generate signal epochs with variability
    - get_amplitude_at(): Get expected amplitude at a specific time
    - to_dict(): Serialize signal configuration to dictionary
    - from_dict(): Deserialize signal from dictionary
    - signal_type: Property returning signal type identifier
    - plot_signal(): Optional method to visualize the signal shape - TODO
    
    Parameters are stored as immutable dataclasses to ensure thread safety
    and prevent accidental modification.
    """
    
    @abstractmethod
    def generate(
        self, 
        n_epochs: int,
        srate: float,
        onset: float = 0.0,
        baseonly: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> np.ndarray:
        """
        Generate signal epochs with trial-to-trial variability. Main method.
        n_samples is determined by the epoch length and sampling rate by the following
        formula: n_samples = int(np.round(epoch_length / 1000 * srate))
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs/trials to generate
        n_samples : int
            Number of samples per epoch
        srate : float
            Sampling rate in Hz
        onset : float, optional
            Signal onset time in seconds relative to epoch start (default: 0.0)
        random_state : int or np.random.RandomState, optional
            Random state for reproducibility
            
        Returns
        -------
        signal : np.ndarray
            Generated signal data, shape (n_epochs, n_samples)
            
        Notes
        -----
        Each epoch may vary according to the signal's variability parameters
        (e.g., latency jitter, amplitude variability, phase variability).
        """
        pass
    
    @abstractmethod
    def get_amplitude_at(self, time: float, **kwargs) -> float:
        """
        Get expected amplitude at a specific time point.
        
        This returns the deterministic amplitude (without variability) that
        the signal would have at the given time point.
        
        Parameters
        ----------
        time : float
            Time in seconds
        **kwargs
            Additional signal-specific parameters
            
        Returns
        -------
        amplitude : float
            Expected amplitude at the given time
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize signal configuration to dictionary.
        
        This method should produce a dictionary that can be serialized to JSON
        and contains all information needed to reconstruct the signal.
        
        Returns
        -------
        config : dict
            Dictionary containing signal type and parameters
            
        Examples
        --------
        >>> signal = ERPSignal(params)
        >>> config = signal.to_dict()
        >>> # Save to JSON
        >>> import json
        >>> with open('signal.json', 'w') as f:
        ...     json.dump(config, f)
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """
        Deserialize signal from dictionary.
        
        This method should reconstruct a signal instance from a dictionary
        produced by to_dict().
        
        Parameters
        ----------
        data : dict
            Dictionary containing signal configuration
            
        Returns
        -------
        signal : Signal
            Reconstructed signal instance
            
        Examples
        --------
        >>> # Load from JSON
        >>> import json
        >>> with open('signal.json', 'r') as f:
        ...     config = json.load(f)
        >>> signal = ERPSignal.from_dict(config)
        """
        pass
    
    @abstractmethod
    def plot_signal(self, srate: int, epoch_length: int, **kwargs):
        """
        Plot signal activation pattern with variability.
        
        Abstract method - implemented by subclasses.
        See ERPSignal.plot_signal() and NoiseSignal.plot_signal() for details.
        """
        pass
    
    @property
    @abstractmethod
    def signal_type(self) -> str:
        """
        Return signal type identifier.
        
        This should be a unique string identifying the signal type,
        e.g., 'erp', 'ersp', 'noise', 'arm', 'data'.
        
        Returns
        -------
        type_id : str
            Signal type identifier
        """
        pass
    
    @abstractmethod
    def _validate(self) -> None:
        """Validate signal parameters."""
        pass
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(type='{self.signal_type}')"



"""
MATLAB structure for Noise signal definition.

    .type:                 class type (must be 'noise')
*   .color:                noise color, 'white', 'pink', 'brown', 
                           'blue' or 'purple' for gaussian noise. add
                           '-unif' (e.g. 'brown-unif' for uniform
                           noise.
*   .peakAmplitude:        1-by-n matrix of peak amplitudes
    .peakAmplitudeDv:      1-by-n matrix of peak amplitude deviations
    .peakAmplitudeSlope:   1-by-n matrix of peak amplitude slopes
    .probability:          0-1 scalar indicating probability of
                           appearance
    .probabilitySlope:     scalar, slope of the probability
    
----

MATLAB structure for ERP signal definition.

    .type:                 class type (must be 'erp')
*   .peakLatency:          1-by-n matrix of peak latencies
    .peakLatencyDv:        1-by-n matrix of peak latency deviations
    .peakLatencySlope:     1-by-n matrix of peak latency slopes
    .peakLatencyShift:     value indicating maximum absolute peak 
                           latency deviation applied equally to all
                           peaks
*   .peakWidth:            1-by-n matrix of peak widths
    .peakWidthDv:          1-by-n matrix of peak width deviations
    .peakWidthSlope:       1-by-n matrix of peak width slopes
*   .peakAmplitude:        1-by-n matrix of peak amplitudes
    .peakAmplitudeDv:      1-by-n matrix of peak amplitude deviations
    .peakAmplitudeSlope:   1-by-n matrix of peak amplitude slopes
    .probability:          0-1 scalar indicating probability of
                           appearance
    .probabilitySlope:     scalar, slope of the probability
    
----

MATLAB structure for autoregressive signal definition.

    .type:                 class type (must be 'arm')
*   .order:                order of the autoregressive model, i.e. 
                           the number of lags
*   .amplitude:            the maximum absolute amplitude of the signal
    .amplitudeDv:          maximum amplitude deviation
    .amplitudeSlope:       amplitude slope
    .arm:                  the coefficient tensor of the 
                           autoregressive model
    .probability:          0-1 scalar indicating probability of
                           appearance
    .probabilitySlope:     scalar, slope of the probability
"""