"""
Base classes for signal generation.

This module defines the abstract base classes and interfaces for all signal types
in the pysereega library. All concrete signal implementations must inherit from
this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt


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


class SignalComposition(Signal):
    """
    Composite signal that combines multiple signals together.
    
    This class allows you to combine multiple Signal instances into a single
    composite signal. The generate() method will generate each component signal
    and sum them together to produce the final output.
    
    Parameters
    ----------
    signals : List[Signal]
        List of Signal instances to combine
        
    Notes
    -----
    The composite signal will have the same variability as its components, but
    the overall amplitude will be the sum of the component amplitudes.
    """
    def __init__(self, signals: List[Signal]):
        super().__init__()
        self.signals = signals
        self._validate()
        
    def _validate(self):
        for comp in self.signals:
            if not isinstance(comp, Signal):
                raise ValueError(f"All components must be instances of Signal, got {type(comp)}")
        for comp in self.signals:
            comp._validate()
    
    def generate(self, n_epochs, srate, onset = 0, baseonly = False, random_state = None):
        """Generate signal epochs by summing component signals."""
        generated_components = [comp.generate(n_epochs, srate, onset, baseonly, random_state) for comp in self.signals]
        return np.sum(generated_components, axis=0)
    
    def get_amplitude_at(self, time: float, **kwargs) -> float:
        """Get expected amplitude at a specific time by summing component amplitudes."""
        return sum(comp.get_amplitude_at(time, **kwargs) for comp in self.signals)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize composite signal to dictionary."""
        return {
            'type': 'composite',
            'components': [comp.to_dict() for comp in self.signals]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalComposition':
        """Deserialize composite signal from dictionary."""
        if data['type'] != 'composite':
            raise ValueError(f"Invalid type for SignalComposition: {data['type']}")
        components = [Signal.from_dict(comp_data) for comp_data in data['components']]
        return cls(components)
    
    def plot_signal(self, srate=500, epoch_length=1000, n_epochs=100,
                show_deviations=True, show_slopes=True, baseonly=False,
                prestim=0, ax=None, colors=None, random_state=None, **kwargs):
        """
        Plot composite signal as unified total (sum of all components).
        
        Shows:
        - Base signal (sum, no variability)
        - Deviation envelope (±1 SD from Monte Carlo)
        - Final epoch with slopes (if applicable)
        
        Usage:
            fig, ax = comp.plot_signal(show_deviations=True, show_slopes=True)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
        
        if colors is None:
            colors = ['#2E86AB', '#A23B72']
        
        n_samples = int(np.round(epoch_length / 1000 * srate))
        time = np.arange(n_samples) / srate * 1000
        if prestim > 0:
            time = time - prestim
        
        random_state = random_state if random_state is not None else 0
        rng = np.random.RandomState(random_state)
        
        # Base signal (no variability)
        base = np.zeros(n_samples)
        for signal in self.signals:
            base += signal.generate(1, srate, epoch_length, baseonly=True, random_state=rng)[0]
        
        ax.plot(time, base, '-', color=colors[0], linewidth=2.5, 
                label='Base', zorder=5)
        
        if not baseonly and show_deviations:
            # Monte Carlo: generate realizations to estimate variability 
            # TODO: n_real could be a parameter 
            # TODO: we could opptimize it by generating all components together rather than in a loop
            n_real = 50
            realizations = np.zeros((n_real, n_samples))
            
            for i in range(n_real):
                epoch = np.zeros(n_samples)
                for signal in self.signals:
                    epoch += signal.generate(1, srate, epoch_length, baseonly=False)[0]
                realizations[i] = epoch
            
            mean = realizations.mean(axis=0)
            std = realizations.std(axis=0)
            
            ax.fill_between(time, mean - std, mean + std,
                        color=colors[0], alpha=0.2, zorder=3,
                        label='±1 SD')
        
        if not baseonly and show_slopes:
            # has any component slpes?
            has_slopes = any(
                getattr(s, 'has_amp_slope', False) or 
                getattr(s, 'has_lat_slope', False) or 
                getattr(s, 'has_wid_slope', False)
                for s in self.signals
            )
            
            if has_slopes:
                # Generate final epoch with slopes
                final = np.zeros(n_samples)
                
                for signal in self.signals:
                    if hasattr(signal, '_generate_peaks'):  # ERPSignal
                        lats = signal.peak_latency + signal.peak_latency_slope
                        wids = np.maximum(signal.peak_width + signal.peak_width_slope, 1.0)
                        amps = signal.peak_amplitude + signal.peak_amplitude_slope
                        final += signal._generate_peaks(time, lats, wids, amps)
                        
                    elif hasattr(signal, '_generate_colored_noise'):  # NoiseSignal
                        if signal.has_amp_slope:
                            final_amp = signal.amplitude + signal.amplitude_slope
                            final += signal._generate_colored_noise(n_samples, final_amp, rng)
                        else:
                            final += signal.generate(1, srate, epoch_length, baseonly=True, random_state=rng)[0]
                    else:
                        final += signal.generate(1, srate, epoch_length, baseonly=True, random_state=rng)[0]
                
                ax.plot(time, final, '-', color=colors[1], linewidth=2.5,
                    label='Final epoch', zorder=6)
        
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude (µV)', fontsize=11, fontweight='bold')
        ax.set_title(f'Composite Signal ({len(self.signals)} components)', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.95, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        fig.tight_layout()
        return fig, ax

    def plot_components(self, srate=500, epoch_length=1000, n_epochs=100,
                        show_sum=True, show_deviations=False, prestim=0,
                        figsize=(12, 10), random_state=None):
        """
        Plot each component in separate subplots.
        
        Shows:
        - Each component individually (using their plot_signal method)
        - Optional sum of all components
        
        Usage:
            fig, axes = comp.plot_components(show_sum=True)
        """
        n_components = len(self.signals)
        n_rows = n_components + (1 if show_sum else 0)
        
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
        if n_rows == 1:
            axes = [axes]
        
        n_samples = int(np.round(epoch_length / 1000 * srate))
        time = np.arange(n_samples) / srate * 1000
        if prestim > 0:
            time = time - prestim
        
        rng = np.random.RandomState(random_state if random_state is not None else 0)
        colors = plt.cm.tab10(np.linspace(0, 0.9, n_components))
        
        total = np.zeros(n_samples)
        
        # Plot each component
        for idx, (signal, ax) in enumerate(zip(self.signals, axes[:n_components])):
            if hasattr(signal, 'plot_signal'):
                signal.plot_signal(
                    srate=srate,
                    epoch_length=epoch_length,
                    n_epochs=n_epochs,
                    show_deviations=show_deviations,
                    show_slopes=False,
                    prestim=prestim,
                    ax=ax,
                    colors=[colors[idx], '#666666']
                )
            else:
                component = signal.generate(1, srate, epoch_length, baseonly=True, random_state=rng)
                ax.plot(time, component[0], color=colors[idx], linewidth=2)
                ax.set_ylabel('Amplitude', fontsize=10)
                ax.grid(True, alpha=0.3)
            
            signal_type = getattr(signal, 'signal_type', type(signal).__name__)
            ax.set_title(f'Component {idx+1}: {signal_type}', 
                        fontsize=11, fontweight='bold', loc='left')
            
            component = signal.generate(1, srate, epoch_length, baseonly=True, random_state=rng)
            total += component[0]
        
        # Sum subplot
        if show_sum:
            ax_sum = axes[-1]
            ax_sum.plot(time, total, color='#2E86AB', linewidth=2.5, label='Sum')
            ax_sum.axhline(0, color='k', linewidth=0.8, alpha=0.3)
            ax_sum.set_title('SUM of all components', fontsize=11, fontweight='bold', loc='left')
            ax_sum.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
            ax_sum.set_ylabel('Amplitude (µV)', fontsize=10)
            ax_sum.grid(True, alpha=0.3)
            ax_sum.legend()
        
        axes[-1].set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
        
        fig.suptitle(f'Composite Signal: Component Breakdown ({n_components} signals)', 
                    fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        return fig, axes

    def plot_decomposition(self, srate=500, epoch_length=1000, n_epochs=100,
                        show_individual=True, show_deviations=True, prestim=0,
                        figsize=(14, 6), random_state=None):
        """
        Plot total signal + component decomposition side-by-side.
        
        Shows:
        - Left panel: Total composite with deviations/slopes
        - Right panel: Components stacked (with offset for visibility)
        
        Usage:
            fig, (ax_total, ax_decomp) = comp.plot_decomposition()
        """
        if show_individual:
            fig, (ax_total, ax_decomp) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax_total = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
            ax_decomp = None
        
        # Left panel: Total
        self.plot_signal(
            srate=srate,
            epoch_length=epoch_length,
            n_epochs=n_epochs,
            show_deviations=show_deviations,
            show_slopes=True,
            prestim=prestim,
            ax=ax_total,
            random_state=random_state
        )
        ax_total.set_title('A) Total Composite', fontsize=12, fontweight='bold', loc='left')
        
        # Right panel: Decomposition
        if show_individual and ax_decomp is not None:
            n_samples = int(np.round(epoch_length / 1000 * srate))
            time = np.arange(n_samples) / srate * 1000
            if prestim > 0:
                time = time - prestim
            
            rng = np.random.RandomState(random_state)
            colors = plt.cm.Set2(np.linspace(0, 0.9, len(self.signals)))
            
            components = []
            labels = []
            
            for idx, signal in enumerate(self.signals):
                component = signal.generate(1, srate, epoch_length, baseonly=True, random_state=rng)
                components.append(component[0])
                
                signal_type = getattr(signal, 'signal_type', type(signal).__name__)
                labels.append(f"{signal_type} #{idx+1}")
            
            # Stacked plot with offset
            offset = 0
            max_amp = max(np.abs(c).max() for c in components)
            spacing = max_amp * 1.5
            
            for idx, (component, label) in enumerate(zip(components, labels)):
                y = component + offset
                ax_decomp.plot(time, y, color=colors[idx], linewidth=2, 
                            label=label, alpha=0.8)
                ax_decomp.axhline(offset, color='gray', linestyle=':', 
                                linewidth=0.8, alpha=0.3)
                offset -= spacing
            
            ax_decomp.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
            ax_decomp.set_ylabel('Amplitude (µV)', fontsize=11, fontweight='bold')
            ax_decomp.set_title('B) Components (stacked)', fontsize=12, fontweight='bold', loc='left')
            ax_decomp.legend(loc='upper right', framealpha=0.95, fontsize=9)
            ax_decomp.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_decomp.set_yticks([])  # Stacked plots have arbitrary offset
        
        fig.suptitle(f'Composite Signal Analysis ({len(self.signals)} components)', 
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        if show_individual:
            return fig, (ax_total, ax_decomp)
        else:
            return fig, ax_total
        
    @property
    def signal_type(self) -> str:
        """Return signal type identifier."""
        return 'composite'
    
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