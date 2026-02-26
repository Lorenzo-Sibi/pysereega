"""
Noise signal generation.

This module implements colored noise signals matching the MATLAB SEREEGA
implementation. Supports multiple noise colors (white, pink, brown, blue, purple)
with both Gaussian and uniform distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from .signal import Signal


class NoiseSignal(Signal):
    """
    Colored noise signal.
    
    Generates noise with specific power spectral density characteristics.
    Supports all MATLAB SEREEGA noise features including different colors,
    Gaussian and uniform distributions, and variability parameters.
    
    Noise Colors:
    - white: flat power spectrum (1/f^0)
    - pink: 1/f power spectrum
    - brown: 1/f^2 power spectrum  
    - blue: f power spectrum (1/f^-1)
    - purple: f^2 power spectrum (1/f^-2)
    
    Parameters
    ----------
    color : str
        Noise color: 'white', 'pink', 'brown', 'blue', 'purple'
        Add '-unif' suffix for uniform distribution (e.g., 'pink-unif')
        Default is Gaussian distribution
    amplitude : float
        Maximum absolute amplitude (signal normalized to this value)
    amplitude_dv : float, optional
        Amplitude deviation (6-sigma range, default: 0)
    amplitude_slope : float, optional
        Amplitude slope over epochs (default: 0)
    probability : float, optional
        Probability of occurrence (0-1, default: 1)
    probability_slope : float, optional
        Change in probability over epochs (default: 0)
        
    Examples
    --------
    >>> # White Gaussian noise
    >>> white = NoiseSignal(color='white', amplitude=1.0)
    
    >>> # Pink noise with variability
    >>> pink = NoiseSignal(
    ...     color='pink',
    ...     amplitude=0.5,
    ...     amplitude_dv=0.1,
    ...     amplitude_slope=-0.2
    ... )
    
    >>> # Brown uniform noise
    >>> brown_unif = NoiseSignal(color='brown-unif', amplitude=0.8)
    
    Notes
    -----
    Noise coloring is achieved through FFT-based spectral shaping,
    avoiding the need for external DSP dependencies.
    
    The 6-sigma convention means that amplitude_dv defines the range
    within which 99.7% of values will fall.
    """
    
    # Color to spectral slope mapping
    # For Gaussian (FFT-based): negative slopes for pink/brown (1/f shaping)
    # For Uniform (coefficient-based): positive alphas (as in MATLAB coloreduniform)
    COLOR_SLOPES_GAUSSIAN = {
        'white': 0,
        'pink': -1,   # 1/f power spectrum
        'brown': -2,  # 1/f^2 power spectrum
        'blue': 1,    # f power spectrum
        'purple': 2   # f^2 power spectrum
    }
    
    COLOR_ALPHAS_UNIFORM = {
        'white': 0,
        'pink': 1,    # MATLAB: noise_generate_signal_coloreduniform(n, range, 1)
        'brown': 2,   # MATLAB: noise_generate_signal_coloreduniform(n, range, 2)
        'blue': -1,   # MATLAB: noise_generate_signal_coloreduniform(n, range, -1)
        'purple': -2  # MATLAB: noise_generate_signal_coloreduniform(n, range, -2)
    }
    
    def __init__(self,
                 color: str = 'white',
                 amplitude: float = 1.0,
                 amplitude_dv: float = 0,
                 amplitude_slope: float = 0,
                 probability: float = 1,
                 probability_slope: float = 0):
        
        # Parse color (may have '-unif' suffix)
        self.color = color
        self.process = 'gaussian'  # default
        
        if color.endswith('-unif'):
            self.base_color = color[:-5]  # Remove '-unif'
            self.process = 'uniform'
        else:
            self.base_color = color
        
        # Validate color
        if self.base_color not in self.COLOR_SLOPES_GAUSSIAN:
            raise ValueError(
                f"Unknown color '{self.base_color}'. "
                f"Must be one of: {list(self.COLOR_SLOPES_GAUSSIAN.keys())}")
        
        self.amplitude = float(amplitude)
        self.amplitude_dv = float(amplitude_dv)
        self.amplitude_slope = float(amplitude_slope)
        self.probability = float(probability)
        self.probability_slope = float(probability_slope)
        
        self._signal_type = 'noise'
        
        self._validate()
        
        # mask for optimization
        self.has_amp_dv = (self.amplitude_dv != 0)
        self.has_amp_slope = (self.amplitude_slope != 0)
    
    def _validate(self):
        """Validate parameters."""
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")
        
        if not 0 <= self.probability <= 1:
            raise ValueError(f"probability must be between 0 and 1, got {self.probability}")
    
    def generate(self,
                 n_epochs: int,
                 srate: float,
                 epoch_length: float,
                 baseonly: bool = False,
                 random_state: Optional[Union[int, np.random.RandomState]] = None) -> np.ndarray:
        """
        Generate colored noise signal.
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs
        srate : float
            Sampling rate in Hz
        epoch_length : float
            Epoch length in ms
        baseonly : bool, optional
            If True, generate base signal without variability (default: False)
        random_state : int or RandomState, optional
            Random state for reproducibility
            
        Returns
        -------
        signal : np.ndarray
            Generated signal (n_epochs, n_samples)
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
        else:
            rng = np.random
        
        n_samples = int(np.round(epoch_length / 1000 * srate))
        
        if baseonly:
            return self._generate_base(n_epochs, n_samples, rng)
        
        random_probs = rng.rand(n_epochs)
        
        if self.has_amp_dv:
            random_amp_devs = rng.randn(n_epochs) / 6.0
        else:
            random_amp_devs = None
        
        # Epoch progression for slopes
        if n_epochs > 1:
            epoch_prog = np.arange(n_epochs, dtype=np.float64) / (n_epochs - 1)
        else:
            epoch_prog = np.zeros(1, dtype=np.float64)
        
        probs = np.clip(self.probability + self.probability_slope * epoch_prog,0, 1)
        signal = np.zeros((n_epochs, n_samples), dtype=np.float64)
        
        for epoch in range(n_epochs):
            if random_probs[epoch] > probs[epoch]:
                continue  # Leave as zeros
            amp = self.amplitude          
              
            if self.has_amp_slope:
                amp = amp + self.amplitude_slope * epoch_prog[epoch]
            
            if self.has_amp_dv:
                amp = amp + random_amp_devs[epoch] * self.amplitude_dv
            
            signal[epoch] = self._generate_colored_noise(n_samples, amp, rng)
        
        return signal
    
    def _generate_colored_noise(self,
                                n_samples: int,
                                amplitude: float,
                                rng) -> np.ndarray:
        """
        Generate a single epoch of colored noise.
        
        Chooses between Gaussian and Uniform generation based on self.process.
        """
        if self.process == 'uniform':
            # Use MATLAB coloreduniform algorithm
            if self.base_color == 'white':
                # Simple uniform white noise in range (-amplitude, amplitude)
                noise = rng.uniform(-amplitude, amplitude, n_samples)
            else:
                # Colored uniform noise using MATLAB algorithm
                alpha = self.COLOR_ALPHAS_UNIFORM[self.base_color]
                noise = self._generate_colored_noise_uniform(n_samples, amplitude, alpha, rng)
        else:
            # Gaussian colored noise
            noise = self._generate_colored_noise_gaussian(n_samples, amplitude, rng)
        
        # Center around 0
        noise = noise - noise.mean()
        
        # Normalize to amplitude
        noise = self._normalize_to_amplitude(noise, amplitude)
        
        return noise
    
    def _generate_colored_noise_gaussian(self,
                                         n_samples: int,
                                         amplitude: float,
                                         rng) -> np.ndarray:
        """
        Generate colored Gaussian noise using FFT-based spectral shaping.
        This replaces MATLAB's dsp.ColoredNoise which requires DSP Toolbox.
        """
        # Generate white Gaussian noise
        noise = rng.randn(n_samples)
        
        # Apply spectral coloring
        if self.base_color != 'white':
            slope = self.COLOR_SLOPES_GAUSSIAN[self.base_color]
            noise = self._apply_spectral_slope(noise, slope)
        
        return noise
    
    def _generate_colored_noise_uniform(self,
                                       n_samples: int,
                                       amplitude: float,
                                       alpha: float,
                                       rng) -> np.ndarray:
        """
        Generate colored uniform noise - MATLAB SEREEGA compatible.
        
        This implements the exact algorithm from noise_generate_signal_coloreduniform.m
        (f_alpha_uniform by Miroslav Stoyanov) for maximum compatibility.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        amplitude : float
            Maximum absolute amplitude (used as range/2)
        alpha : float
            Spectral slope (1/f^alpha coloring)
        rng : RandomState
            Random number generator
            
        Returns
        -------
        noise : ndarray
            Colored uniform noise
            
        Notes
        -----
        This is the exact MATLAB algorithm:
        1. Generate coefficients Hk
        2. Generate white uniform noise Wk
        3. FFT both
        4. Multiply in frequency domain
        5. IFFT and extract real part
        """
        n = n_samples
        
        # Generate coefficients Hk
        hfa = np.zeros(2 * n)
        hfa[0] = 1.0
        for i in range(1, n):
            hfa[i] = hfa[i-1] * (0.5 * alpha + (i - 1)) / i
        # hfa[n:2*n] already zeros
        
        # Fill Wk with white uniform noise in range (-amplitude, amplitude)
        wfa = np.concatenate([
            rng.uniform(-amplitude, amplitude, n),
            np.zeros(n)
        ])
        
        # FFT both
        fh = np.fft.fft(hfa)
        fw = np.fft.fft(wfa)
        
        # Multiply (only need first n+1 points)
        fh = fh[:n+1]
        fw = fw[:n+1]
        fw = fh * fw
        
        # Scaling to match Numerical Recipes behavior
        fw[0] = fw[0] / 2
        fw[-1] = fw[-1] / 2
        
        # Reconstruct for IFFT
        fw = np.concatenate([fw, np.zeros(n-1)])
        
        # IFFT
        x = np.fft.ifft(fw)
        
        # Extract real part and scale
        x = 2 * np.real(x[:n])
        
        return x
    
    def _apply_spectral_slope(self, signal: np.ndarray, slope: float) -> np.ndarray:
        """
        Apply spectral slope to signal via FFT shaping.
        
        This implements 1/f^alpha coloring where alpha is the slope parameter:
        - slope = 0: white (no change)
        - slope = -1: pink (1/f)
        - slope = -2: brown (1/f^2)
        - slope = +1: blue (f)
        - slope = +2: purple (f^2)
        
        Parameters
        ----------
        signal : ndarray
            Input white noise
        slope : float
            Spectral slope (alpha in 1/f^alpha)
            
        Returns
        -------
        colored : ndarray
            Spectrally shaped noise
        """
        n = len(signal)
        
        # FFT
        fft_signal = np.fft.rfft(signal)
        
        # Frequency vector (avoid division by zero at DC)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-10  # Small value to avoid div by zero
        
        # Apply power law: multiply by f^(slope/2)
        # (We use slope/2 because power is amplitude squared)
        fft_signal *= freqs ** (slope / 2)
        
        # IFFT
        colored = np.fft.irfft(fft_signal, n=n)
        
        return colored
    
    def _normalize_to_amplitude(self, signal: np.ndarray, amplitude: float) -> np.ndarray:
        """
        Normalize signal so maximum absolute value equals amplitude.
        
        This matches MATLAB's utl_normalise behavior.
        
        Parameters
        ----------
        signal : ndarray
            Input signal
        amplitude : float
            Target maximum absolute amplitude
            
        Returns
        -------
        normalized : ndarray
            Normalized signal
        """
        max_abs = np.abs(signal).max()
        
        if max_abs > 0:
            return signal * (amplitude / max_abs)
        else:
            return signal
    
    def _generate_base(self,
                      n_epochs: int,
                      n_samples: int,
                      rng) -> np.ndarray:
        """
        Generate base signal without variability.
        
        In baseonly mode, we generate noise with base amplitude only,
        no slopes or deviations.
        """
        signal = np.zeros((n_epochs, n_samples), dtype=np.float64)
        
        for epoch in range(n_epochs):
            signal[epoch] = self._generate_colored_noise(
                n_samples,
                self.amplitude,
                rng
            )
        
        return signal
    
    def get_amplitude_at(self, time: float, **kwargs) -> float:
        """
        Get expected amplitude at a specific time point.
        
        For noise, amplitude is constant over time (in expectation),
        so this returns the base amplitude.
        
        Parameters
        ----------
        time : float
            Time in seconds (ignored for noise)
            
        Returns
        -------
        amplitude : float
            Expected amplitude (constant for noise)
        """
        return self.amplitude
    
    def plot_signal(self, srate: int, epoch_length: int,
                show_deviations: bool=True, show_slopes: bool=True, baseonly: bool=False,
                prestim: float=0, ax=None, colors=None, random_state=None):
        """
        Plot noise signal with variability and slopes.
        
        MATLAB equivalent: noise_plot_signal_fromclass(class, epochs)
        
        Parameters
        ----------
        srate : float
            Sampling rate (Hz)
        epoch_length : float
            Epoch length (ms)
        show_deviations : bool
            Show deviation envelope (offset lines)
        show_slopes : bool
            Show final epoch with slopes (second color)
        baseonly : bool
            Show only base signal (no variability)
        prestim : float
            Prestimulus period in ms
        ax : Axes
            Matplotlib axes (None = create new)
        colors : list
            [initial_color, final_color] (default: blue, red)
        random_state : int
            Random seed for reproducible noise
            
        Returns
        -------
        fig, ax : Figure and Axes objects
        
        Usage
        -----
        >>> noise = NoiseSignal(color='pink', amplitude=1.0,
        ...                     amplitude_dv=0.3, amplitude_slope=-0.5)
        >>> fig, ax = noise.plot_signal(show_deviations=True, show_slopes=True)
        >>> plt.show()
        """        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        if colors is None:
            colors = ['#1f77b4', '#d62728']  # Blue, Red
        
        # Time vector
        n_samples = int(np.round(epoch_length / 1000 * srate))
        time = np.arange(n_samples) / srate * 1000
        if prestim > 0:
            time = time - prestim
        
        rng = np.random.RandomState(random_state)
        
        # Base signal (solid, first color)
        base = self._generate_colored_noise(n_samples, self.amplitude, rng)
        ax.plot(time, base, '-', color=colors[0], linewidth=1.5, 
                label='Base', alpha=0.8, zorder=5)
        
        if not baseonly and show_deviations:
            # Deviation envelope (dotted, first color)
            # Note: For noise, deviation is amplitude offset, not new noise generation
            ax.plot(time, base - self.amplitude_dv, ':', color=colors[0], 
                    linewidth=1.5, alpha=0.6, zorder=3)
            ax.plot(time, base + self.amplitude_dv, ':', color=colors[0], 
                    linewidth=1.5, alpha=0.6, zorder=3)
        
        if not baseonly and show_slopes and self.has_amp_slope:
            # Final epoch with slopes (solid, second color)
            final_amp = self.amplitude + self.amplitude_slope
            final = self._generate_colored_noise(n_samples, final_amp, rng)
            
            ax.plot(time, final, '-', color=colors[1], linewidth=1.5,
                    label='Final epoch', alpha=0.8, zorder=4)
            
            if show_deviations:
                # Slopes with deviations (dotted/dashed, second color)
                ax.plot(time, final - self.amplitude_dv, ':', color=colors[1],
                        linewidth=1.5, alpha=0.6, zorder=2)
                ax.plot(time, final + self.amplitude_dv, '--', color=colors[1],
                        linewidth=1.5, alpha=0.6, zorder=2)
        
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Noise Signal: {self}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig, ax
    
    def to_dict(self) -> dict:
        """Serialize noise configuration to dictionary."""
        return {
            'color': self.color,
            'amplitude': self.amplitude,
            'amplitude_dv': self.amplitude_dv,
            'amplitude_slope': self.amplitude_slope,
            'probability': self.probability,
            'probability_slope': self.probability_slope,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'NoiseSignal':
        """Deserialize noise from dictionary."""
        return cls(**data)
    
    @property
    def signal_type(self) -> str:
        """Return signal type identifier."""
        return self._signal_type
    
    def __repr__(self):
        return (
            f"NoiseSignal(color='{self.color}', "
            f"amplitude={self.amplitude:.2f}, "
            f"process='{self.process}')"
        )