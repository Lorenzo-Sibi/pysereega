
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from pathlib import Path

@dataclass
class ChannelLocation:
    """Infomration regrding a single EEG channel/electrode"""
    labels: str # Channels label (e.g. "fz", "C3", ...)
    X: float = 0.0 # X coordinate (typically left-rght in head coordinates)
    Y: float = 0.0 # Y coordinate (typically anterior-posterior)
    Z: float = 0.0 # Z coordinate (typically inferior-superior)
    
    theta : float = 0.0
    radius: float = 0.0
    """Spherical radius"""
    
    sph_theta: float = 0.0
    """Spherical theta in standard coordinates"""
    
    sph_phi: float = 0.0
    """Spherical phi angle"""
    
    sph_radius: float = 0.0
    """Spherical radius"""
    
    type: str = 'EEG'
    """Channel type"""
    
    urchan: int = 0
    """Original channel index"""
    
    ref: str = ''
    """Reference channel"""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (EEGLAB-compatible format)"""
        return {
            'labels': self.labels,
            'X': self.X, 'Y': self.Y, 'Z': self.Z,
            'theta': self.theta, 'radius': self.radius,
            'sph_theta': self.sph_theta,
            'sph_phi': self.sph_phi,
            'sph_radius': self.sph_radius,
            'type': self.type,
            'urchan': self.urchan,
            'ref': self.ref
        }
        
@dataclass
class LeadField:
    """
    Lead field structure containing forward model information.
    
    This represents the relationship between source activity and scalp recordings.
    For each source, it contains the projection pattern to all electrodes for
    three orthogonal dipole orientations (X, Y, Z).
    
    Attributes
    ----------
    leadfield : np.ndarray
        Lead field matrix of shape (n_channels, n_sources, 3).
        The last dimension represents X, Y, Z dipole orientations.
        Units depend on the source (typically µV/(nA·m) or relative).
        
    pos : np.ndarray
        Source positions of shape (n_sources, 3) in MNI coordinates (mm) same as SEREEGA origial.
        Coordinates are typically: X=left-right, Y=posterior-anterior, Z=inferior-superior (see ChannelLocation class).
        
    orientation : np.ndarray
        Default dipole orientations of shape (n_sources, 3).
        Often perpendicular to cortical surface, or zeros if no default available.
        
    chanlocs : List[ChannelLocation]
        Channel information for each electrode.
        
    atlas : Optional[List[str]]
        Anatomical labels for each source (e.g., 'Brain_Right_Visual_Cortex').
        Must start with 'Brain', 'Eye', or 'Muscle' for compatibility.
        
    method : str
        Method used to generate the lead field (e.g., 'nyhead', 'fieldtrip').
        
    source : str
        Source of the lead field data.
        
    unit : str
        Units of the lead field values (e.g., 'µV/(nA·m)', 'relative').
    """
    
    leadfield: np.ndarray
    """Lead field matrix [n_channels x n_sources x 3]"""
    
    pos: np.ndarray
    """Source positions [n_sources x 3] in MNI coordinates (mm)"""
    
    orientation: np.ndarray
    """Default orientations [n_sources x 3]"""
    
    chanlocs: List[ChannelLocation]
    """Channel location information"""
    
    atlas: Optional[List[str]] = None
    """Anatomical atlas labels for each source"""
    
    method: str = 'unknown'
    """Generation method"""
    
    source: str = 'unknown'
    """Data source"""
    
    unit: str = 'relative'
    """Lead field units"""
    
    metadata: Dict = field(default_factory=dict)
    """Additional metadata"""
    
    def __post_init__(self):
        """Validate lead field structure after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate the lead field structure."""
        # Check shapes
        n_channels, n_sources, n_orient = self.leadfield.shape
        
        if n_orient != 3:
            raise ValueError(f"Lead field must have 3 orientations (XYZ), got {n_orient}")
        
        if self.pos.shape != (n_sources, 3):
            raise ValueError(
                f"Position array shape {self.pos.shape} doesn't match "
                f"expected ({n_sources}, 3)"
            )
        
        if self.orientation.shape != (n_sources, 3):
            raise ValueError(
                f"Orientation array shape {self.orientation.shape} doesn't match "
                f"expected ({n_sources}, 3)"
            )
        
        if len(self.chanlocs) != n_channels:
            raise ValueError(
                f"Number of channel locations ({len(self.chanlocs)}) doesn't match "
                f"number of channels ({n_channels})"
            )
        
        if self.atlas is not None and len(self.atlas) != n_sources:
            raise ValueError(
                f"Atlas length ({len(self.atlas)}) doesn't match "
                f"number of sources ({n_sources})"
            )
    
    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.leadfield.shape[0]
    
    @property
    def n_sources(self) -> int:
        """Number of brain sources."""
        return self.leadfield.shape[1]
    
    @property
    def channel_labels(self) -> List[str]:
        """List of channel labels."""
        return [ch.labels for ch in self.chanlocs]
    
    def get_projection(self, 
                      source_idx: int, 
                      orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the projection pattern for a source with given orientation.
        
        Parameters
        ----------
        source_idx : int
            Index of the source (0 to n_sources-1).
        orientation : np.ndarray, optional
            Dipole orientation [x, y, z]. If None, uses default orientation.
            
        Returns
        -------
        np.ndarray
            Projection pattern of shape (n_channels,).
            
        Examples
        --------
        >>> lf = LeadField(...)
        >>> # Get projection with default orientation
        >>> proj = lf.get_projection(100)
        >>> 
        >>> # Get projection with custom orientation (anterior-posterior)
        >>> proj = lf.get_projection(100, orientation=[0, 1, 0])
        """
        if source_idx < 0 or source_idx >= self.n_sources:
            raise ValueError(
                f"Source index {source_idx} out of range [0, {self.n_sources})"
            )
        
        if orientation is None:
            orientation = self.orientation[source_idx]
        else:
            orientation = np.asarray(orientation)
            if orientation.shape != (3,):
                raise ValueError(
                    f"Orientation must be shape (3,), got {orientation.shape}"
                )
        
        # Linear combination of X, Y, Z projections weighted by orientation
        projection = self.leadfield[:, source_idx, :] @ orientation
        
        return projection
    
    def normalize(self, method: str = 'norm') -> 'LeadField':
        """
        Normalize the lead field for comparability across different sources.
        
        Parameters
        ----------
        method : str
            Normalization method:
            - 'norm': Normalize by Frobenius norm of each source
            - 'max': Normalize by maximum absolute value
            
        Returns
        -------
        LeadField
            New lead field with normalized values.
            
        Notes
        -----
        This is useful when lead field units are not well-defined or when
        comparing lead fields from different sources.
        """
        lf_normalized = self.leadfield.copy()
        
        if method == 'norm':
            # Normalize each source by its Frobenius norm
            for i in range(self.n_sources):
                norm = np.linalg.norm(lf_normalized[:, i, :])
                if norm > 0:
                    lf_normalized[:, i, :] /= norm
                    
        elif method == 'max':
            # Normalize by maximum absolute value
            max_val = np.abs(lf_normalized).max()
            if max_val > 0:
                lf_normalized /= max_val
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create new lead field with normalized values
        return LeadField(
            leadfield=lf_normalized,
            pos=self.pos.copy(),
            orientation=self.orientation.copy(),
            chanlocs=self.chanlocs.copy(),
            atlas=self.atlas.copy() if self.atlas else None,
            method=self.method,
            source=self.source,
            unit=f'{self.unit}_normalized_{method}',
            metadata=self.metadata.copy()
        )
    
    def get_sources_in_region(self, region_names: List[str]) -> np.ndarray:
        """
        Get indices of sources in specified anatomical regions.
        
        Parameters
        ----------
        region_names : List[str]
            List of region names to search for (case-insensitive).
            
        Returns
        -------
        np.ndarray
            Array of source indices in the specified regions.
            
        Raises
        ------
        ValueError
            If no atlas is available.
            
        Examples
        --------
        >>> lf = LeadField(...)
        >>> # Get all sources in visual cortex
        >>> sources = lf.get_sources_in_region(['Visual_Cortex'])
        """
        if self.atlas is None:
            raise ValueError("No atlas available for this lead field")
        
        indices = []
        region_names_lower = [r.lower() for r in region_names]
        
        for i, label in enumerate(self.atlas):
            label_lower = label.lower()
            if any(region in label_lower for region in region_names_lower):
                indices.append(i)
        
        return np.array(indices, dtype=int)
    
    def to_dict(self) -> Dict:
        """Convert lead field to dictionary for serialization."""
        return {
            'leadfield': self.leadfield,
            'pos': self.pos,
            'orientation': self.orientation,
            'chanlocs': [ch.to_dict() for ch in self.chanlocs],
            'atlas': self.atlas,
            'method': self.method,
            'source': self.source,
            'unit': self.unit,
            'metadata': self.metadata
        }
    
    def save(self, filepath: str):
        """
        Save lead field to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the lead field. Extension determines format:
            - .npz: NumPy compressed format (recommended)
            - .mat: MATLAB format (requires scipy)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            # Save as NumPy compressed
            data = self.to_dict()
            np.savez_compressed(filepath, **data)
            
        elif filepath.suffix == '.mat':
            # Save as MATLAB format
            from scipy.io import savemat
            data = self.to_dict()
            savemat(filepath, data)
            
        else:
            raise ValueError(
                f"Unsupported file format: {filepath.suffix}. "
                f"Use .npz or .mat"
            )
    
    @classmethod
    def load(cls, filepath: str) -> 'LeadField':
        """
        Load lead field from file.
        
        Parameters
        ----------
        filepath : str
            Path to the lead field file.
            
        Returns
        -------
        LeadField
            Loaded lead field object.
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            data = np.load(filepath, allow_pickle=True)
            chanlocs = [
                ChannelLocation(**ch) for ch in data['chanlocs']
            ]
            atlas = list(data['atlas']) if 'atlas' in data else None
            
            return cls(
                leadfield=data['leadfield'],
                pos=data['pos'],
                orientation=data['orientation'],
                chanlocs=chanlocs,
                atlas=atlas,
                method=str(data.get('method', 'unknown')),
                source=str(data.get('source', 'unknown')),
                unit=str(data.get('unit', 'relative')),
                metadata=dict(data.get('metadata', {}))
            )
            
        elif filepath.suffix == '.mat':
            from scipy.io import loadmat
            data = loadmat(filepath)
            # Implementation for MATLAB format
            raise NotImplementedError("MATLAB loading not yet implemented")
            
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def __repr__(self) -> str:
        """String representation of lead field."""
        return (
            f"LeadField(\n"
            f"  channels={self.n_channels},\n"
            f"  sources={self.n_sources},\n"
            f"  method='{self.method}',\n"
            f"  unit='{self.unit}',\n"
            f"  has_atlas={self.atlas is not None}\n"
            f")"
        )