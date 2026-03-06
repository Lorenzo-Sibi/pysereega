"""
Event-Related Potential (ERP) signal generation.

This module implements ERP signals matching the MATLAB SEREEGA implementation.
ERPs are time-locked transient responses characterized by one or more peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional
from .signal import Signal

class ERPSignal(Signal):
    """
    Event-Related Potential signal.
    
    Generates transient, time-locked responses with configurable peaks.
    Supports all MATLAB SEREEGA ERP features including deviation, slopes,
    probability, and multiple peaks.
    
    Parameters
    ----------
    peak_latency : float or array-like
        Peak latency in ms (can be array for multiple peaks)
    peak_width : float or array-like
        Peak width in ms
    peak_amplitude : float or array-like
        Peak amplitude in µV
    peak_latency_dv : float or array-like, optional
        Peak latency deviation (6-sigma range) in ms
    peak_width_dv : float or array-like, optional
        Peak width deviation in ms
    peak_amplitude_dv : float or array-like, optional
        Peak amplitude deviation in µV
    peak_latency_shift : float, optional
        Global latency shift applied to all peaks (ms)
    peak_latency_slope : float or array-like, optional
        Latency slope over epochs (ms)
    peak_width_slope : float or array-like, optional
        Width slope over epochs (ms)
    peak_amplitude_slope : float or array-like, optional
        Amplitude slope over epochs (µV)
    probability : float, optional
        Probability of occurrence (0-1, default: 1)
    probability_slope : float, optional
        Change in probability over epochs
    type : str, optional
        Peak shape: 'gaussian' (default), 'box', 'triangular'
    prestored : array-like, optional
        Pre-generated signal to use instead of procedural generation
        
    Examples
    --------
    >>> # Simple P300
    >>> p300 = ERPSignal(peak_latency=300, peak_width=200, peak_amplitude=5)
    
    >>> # N170 with variability
    >>> n170 = ERPSignal(
    ...     peak_latency=170,
    ...     peak_width=30,
    ...     peak_amplitude=-3,
    ...     peak_latency_dv=20,
    ...     peak_amplitude_dv=1
    ... )
    
    >>> # Multiple peaks with amplitude slope
    >>> erp = ERPSignal(
    ...     peak_latency=[200, 400],
    ...     peak_width=[100, 150],
    ...     peak_amplitude=[-2, 3],
    ...     peak_amplitude_slope=-1)
    """
    def __init__(self,
                 peak_latency: Union[float, List[float]],
                 peak_width: Union[float, List[float]],
                 peak_amplitude: Union[float, List[float]],
                 peak_latency_dv: Optional[Union[float, List[float]]] = 0,
                 peak_width_dv: Optional[Union[float, List[float]]] = 0,
                 peak_amplitude_dv: Optional[Union[float, List[float]]] = 0,
                 peak_latency_shift: Optional[float] = 0,
                 peak_latency_slope: Optional[Union[float, List[float]]] = 0,
                 peak_width_slope: Optional[Union[float, List[float]]] = 0,
                 peak_amplitude_slope: Optional[Union[float, List[float]]] = 0,
                 probability: Optional[float] = 1,
                 probability_slope: Optional[float] = 0,
                 type: str = 'gaussian',
                 prestored: Optional[np.ndarray] = None):
        
        # Convert to arrays for consistent handling
        self.peak_latency = np.atleast_1d(peak_latency)
        self.peak_width = np.atleast_1d(peak_width)
        self.peak_amplitude = np.atleast_1d(peak_amplitude)
        
        # Deviations
        self.peak_latency_dv = np.atleast_1d(peak_latency_dv)
        self.peak_width_dv = np.atleast_1d(peak_width_dv)
        self.peak_amplitude_dv = np.atleast_1d(peak_amplitude_dv)
        
        # Shift and slopes
        self.peak_latency_shift = peak_latency_shift
        self.peak_latency_slope = np.atleast_1d(peak_latency_slope)
        self.peak_width_slope = np.atleast_1d(peak_width_slope)
        self.peak_amplitude_slope = np.atleast_1d(peak_amplitude_slope)
        
        # Probability
        self.probability = probability
        self.probability_slope = probability_slope
        
        # Shape and prestored
        self.type = type
        self._signal_type = 'erp'
        self.prestored = prestored
        
        # Validate
        self._validate()
        self.n_peaks = len(self.peak_latency)
        
        # Precompute masks for non-zero variability (this is for optimization)
        self.has_lat_dv = (self.peak_latency_dv != 0).any()
        self.has_wid_dv = (self.peak_width_dv != 0).any()
        self.has_amp_dv = (self.peak_amplitude_dv != 0).any()
        self.has_lat_slope = (self.peak_latency_slope != 0).any()
        self.has_wid_slope = (self.peak_width_slope != 0).any()
        self.has_amp_slope = (self.peak_amplitude_slope != 0).any()
    
    def _validate(self):
        """Validate parameters."""
        n_peaks = len(self.peak_latency)
        
        # Broadcast scalar deviations/slopes to match number of peaks
        if len(self.peak_latency_dv) == 1:
            self.peak_latency_dv = np.repeat(self.peak_latency_dv, n_peaks)
        if len(self.peak_width_dv) == 1:
            self.peak_width_dv = np.repeat(self.peak_width_dv, n_peaks)
        if len(self.peak_amplitude_dv) == 1:
            self.peak_amplitude_dv = np.repeat(self.peak_amplitude_dv, n_peaks)
        if len(self.peak_latency_slope) == 1:
            self.peak_latency_slope = np.repeat(self.peak_latency_slope, n_peaks)
        if len(self.peak_width_slope) == 1:
            self.peak_width_slope = np.repeat(self.peak_width_slope, n_peaks)
        if len(self.peak_amplitude_slope) == 1:
            self.peak_amplitude_slope = np.repeat(self.peak_amplitude_slope, n_peaks)
        
        # Check all arrays match number of peaks
        arrays = [
            self.peak_width, self.peak_amplitude,
            self.peak_latency_dv, self.peak_width_dv, self.peak_amplitude_dv,
            self.peak_latency_slope, self.peak_width_slope, self.peak_amplitude_slope
        ]
        for arr in arrays:
            if len(arr) != n_peaks:
                raise ValueError(f"All peak parameters must have same length ({n_peaks} peaks)")
    
    def generate(self, 
                 n_epochs: int, 
                 srate: float, 
                 epoch_length: float, 
                 baseonly: bool = False, 
                 random_state: Optional[Union[int, np.random.RandomState]] = None) -> np.ndarray:
        """
        Generate ERP signal.
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs
        srate : float
            Sampling rate in Hz
        epoch_length : float
            Epoch lenght in ms
            
        Returns
        -------
        signal : np.ndarray
            Generated signal (n_epochs, n_samples)
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
        else:
            rng = np.random
        
        if self.prestored is not None:
            return self._generate_prestored(n_epochs, epoch_length, srate)
        
        n_samples = int(np.round(epoch_length / 1000 * srate))
        
        if baseonly:
            return self._generate_base(n_epochs, n_samples, srate)
        
        random_probs = rng.rand(n_epochs)
        
        if self.has_lat_dv or self.has_wid_dv or self.has_amp_dv:
            # Shape: (n_epochs, n_peaks, 3) for lat/wid/amp
            random_devs = rng.randn(n_epochs, self.n_peaks, 3) / 6.0
        else:
            random_devs = None
        
        if n_epochs > 1:
            epoch_prog = np.arange(n_epochs, dtype=np.float64) / (n_epochs - 1)
        else:
            epoch_prog = np.zeros(1, dtype=np.float64)
        
        probs = np.clip(self.probability + self.probability_slope * epoch_prog, 0, 1)
        time = np.arange(n_samples, dtype=np.float64) / srate * 1000 # ms
        signal = np.zeros((n_epochs, n_samples), dtype=np.float64)
        
        # hybrid loop: epoch loop + vectorized peak generation (this is more memory efficient than fully vectorized)
        for epoch in range(n_epochs):
            if random_probs[epoch] > probs[epoch]:
                continue
            
            ep = epoch_prog[epoch]
            lats = self.peak_latency + self.peak_latency_shift
            if self.has_lat_slope:
                lats = lats + self.peak_latency_slope * ep
            if self.has_lat_dv:
                lats = lats + random_devs[epoch, :, 0] * self.peak_latency_dv
            # Width
            wids = self.peak_width.copy()
            if self.has_wid_slope:
                wids = wids + self.peak_width_slope * ep
            if self.has_wid_dv:
                wids = wids + random_devs[epoch, :, 1] * self.peak_width_dv
            wids = np.maximum(wids, 1.0)
            
            # Amplitude
            amps = self.peak_amplitude.copy()
            if self.has_amp_slope:
                amps = amps + self.peak_amplitude_slope * ep
            if self.has_amp_dv:
                amps = amps + random_devs[epoch, :, 2] * self.peak_amplitude_dv
            
            # Generate all peaks for this epoch (vectorized)
            signal[epoch] = self._generate_peaks(time, lats, wids, amps)
        
        return signal
    
    def _generate_peaks(self,
                                    time: np.ndarray,
                                    latencies: np.ndarray,
                                    widths: np.ndarray,
                                    amplitudes: np.ndarray) -> np.ndarray:
        """
        Generate all peaks vectorized over peaks and time.
        
        Parameters
        ----------
        time : ndarray, shape (n_samples,)
        latencies : ndarray, shape (n_peaks,)
        widths : ndarray, shape (n_peaks,)
        amplitudes : ndarray, shape (n_peaks,)
        
        Returns
        -------
        signal : ndarray, shape (n_samples,)
            Sum of all peaks
        """
        # Broadcasting: (n_peaks, 1) with (n_samples,) → (n_peaks, n_samples)
        lat = latencies[:, np.newaxis]
        wid = widths[:, np.newaxis]
        amp = amplitudes[:, np.newaxis]
        t = time[np.newaxis, :]
        
        if self.type == 'gaussian':
            sigma = wid / 6.0
            peaks = amp * np.exp(-0.5 * ((t - lat) / sigma) ** 2)
        elif self.type == 'box':
            peaks = np.where(np.abs(t - lat) <= wid / 2, amp, 0.0)
        elif self.type == 'triangular':
            dist = np.abs(t - lat)
            peaks = np.where(dist <= wid, amp * (1 - dist / wid), 0.0)
        
        # Sum over peaks: (n_peaks, n_samples) → (n_samples,)
        return peaks.sum(axis=0)
    
    def _generate_base(self, n_epochs: int, n_samples: int, srate: float) -> np.ndarray:
        """Generate base signal without variability."""
        time = np.arange(n_samples, dtype=np.float64) / srate * 1000

        # Base parameters
        lats = self.peak_latency + self.peak_latency_shift
        wids = self.peak_width
        amps = self.peak_amplitude

        # Generate once and tile
        base = self._generate_peaks(time, lats, wids, amps)
        return np.tile(base, (n_epochs, 1))
    
    def _generate_prestored(self, n_epochs: int, epoch_length: float, srate: float) -> np.ndarray:
        """Generate from prestored signal."""
        n_samples = int(np.round(epoch_length * srate / 1000))
        
        if self.prestored.shape[-1] != n_samples:
            raise ValueError(f"Prestored signal size mismatch")
        
        if self.prestored.ndim == 1:
            return np.tile(self.prestored, (n_epochs, 1))
        else:
            if self.prestored.shape[0] == n_epochs:
                return self.prestored.copy()
            elif self.prestored.shape[0] == 1:
                return np.tile(self.prestored, (n_epochs, 1))
            else:
                raise ValueError(f"Prestored epoch count mismatch")
    
    def get_amplitude_at(self, time: float, with_variation: bool = False, **kwargs) -> float:
        """Get amplitude at a specific time point."""
        raise NotImplementedError(
            "get_amplitude_at is not implemented yet for ERPSignal"
            "(still deciding how to handle variability in this method)")
    
    def plot_signal(self, srate: float, epoch_length: float, baseonly: bool = False, show_deviations=True, show_slopes=True,
                prestim=0, ax=None, colors=None):
        """
        Plot ERP signal with variability and slopes for a single epoch (base) and final epoch (slopes)
        
        MATLAB equivalent: erp_plot_signal_fromclass(class, epochs)
        
        Parameters
        ----------
        show_deviations : bool
            Show deviation envelope (dotted/dashed lines)
        show_slopes : bool
            Show final epoch with slopes (second color)
        prestim : float
            Prestimulus period in ms (shifts time axis)
        ax : Axes
            Matplotlib axes (None = create new figure)
        colors : list
            [initial_color, final_color] (default: blue, red)
            
        Returns
        -------
        fig, ax : Figure and Axes objects
        
        Usage
        -----
        >>> erp = ERPSignal(peak_latency=300, peak_width=200, peak_amplitude=5,
        ...                 peak_latency_dv=50, peak_amplitude_slope=-2)
        >>> fig, ax = erp.plot_signal(show_deviations=True, show_slopes=True)
        >>> plt.show()
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        if colors is None:
            colors = ['#1f77b4', '#d62728']
        
        # Time vector
        n_samples = int(np.round(epoch_length / 1000 * srate))
        time = np.arange(n_samples) / srate * 1000
        if prestim > 0:
            time = time - prestim
        
        # Base signal (solid, first color)
        base = self._generate_base(1, n_samples, srate)[0]
        ax.plot(time, base, '-', color=colors[0], linewidth=2, label='Base', zorder=5)
        
        if not baseonly and show_deviations:
            # Deviation envelope (dotted/dashed, first color)
            lats_neg = self.peak_latency - self.peak_latency_dv + self.peak_latency_shift
            wids_neg = np.maximum(self.peak_width - self.peak_width_dv, 1.0)
            amps_neg = self.peak_amplitude - self.peak_amplitude_dv
            sig_neg = self._generate_peaks(time, lats_neg, wids_neg, amps_neg)
            
            lats_pos = self.peak_latency + self.peak_latency_dv + self.peak_latency_shift
            wids_pos = self.peak_width + self.peak_width_dv
            amps_pos = self.peak_amplitude + self.peak_amplitude_dv
            sig_pos = self._generate_peaks(time, lats_pos, wids_pos, amps_pos)
            
            ax.plot(time, sig_neg, ':', color=colors[0], linewidth=1.5, alpha=0.7, zorder=3)
            ax.plot(time, sig_pos, '--', color=colors[0], linewidth=1.5, alpha=0.7, zorder=3)
        
        if not baseonly and show_slopes and (self.has_lat_slope or self.has_wid_slope or self.has_amp_slope):
            # Final epoch with slopes (solid, second color)
            lats_slope = self.peak_latency + self.peak_latency_slope
            wids_slope = np.maximum(self.peak_width + self.peak_width_slope, 1.0)
            amps_slope = self.peak_amplitude + self.peak_amplitude_slope
            sig_slope = self._generate_peaks(time, lats_slope, wids_slope, amps_slope)
            
            ax.plot(time, sig_slope, '-', color=colors[1], linewidth=2, 
                    label='Final epoch', zorder=4)
            
            if show_deviations:
                # Slopes with deviations (dotted/dashed, second color)
                sig_slope_neg = self._generate_peaks(time, 
                    lats_slope - self.peak_latency_dv,
                    np.maximum(wids_slope - self.peak_width_dv, 1.0),
                    amps_slope - self.peak_amplitude_dv)
                
                sig_slope_pos = self._generate_peaks(time,
                    lats_slope + self.peak_latency_dv,
                    wids_slope + self.peak_width_dv,
                    amps_slope + self.peak_amplitude_dv)
                
                ax.plot(time, sig_slope_neg, ':', color=colors[1], linewidth=1.5, alpha=0.7, zorder=2)
                ax.plot(time, sig_slope_pos, '--', color=colors[1], linewidth=1.5, alpha=0.7, zorder=2)
        
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'ERP Signal: {self}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig, ax
    
    def to_dict(self) -> dict:
        return {
            'peak_latency': self.peak_latency.tolist(),
            'peak_width': self.peak_width.tolist(),
            'peak_amplitude': self.peak_amplitude.tolist(),
            'peak_latency_dv': self.peak_latency_dv.tolist(),
            'peak_width_dv': self.peak_width_dv.tolist(),
            'peak_amplitude_dv': self.peak_amplitude_dv.tolist(),
            'peak_latency_shift': self.peak_latency_shift,
            'peak_latency_slope': self.peak_latency_slope.tolist(),
            'peak_width_slope': self.peak_width_slope.tolist(),
            'peak_amplitude_slope': self.peak_amplitude_slope.tolist(),
            'probability': self.probability,
            'probability_slope': self.probability_slope,
            'type': self.type,
        }
    
    def from_dict(self, params: dict) -> Signal:
        return ERPSignal(**params)
    
    def signal_type(self):
        return self._signal_type
    
    def __repr__(self):
        if self.n_peaks == 1:
            return (
                f"ERPSignal(latency={self.peak_latency[0]:.0f}ms, "
                f"width={self.peak_width[0]:.0f}ms, "
                f"amplitude={self.peak_amplitude[0]:.2f}µV)"
            )
        else:
            return f"ERPSignal({self.n_peaks} peaks)"
    
    def get_base_waveform(self, srate: float, epoch_length: float) -> np.ndarray:
        """Get base waveform without variability."""
        n_samples = int(np.round(epoch_length * srate / 1000))
        return self._generate_base(1, n_samples, srate)[0]
