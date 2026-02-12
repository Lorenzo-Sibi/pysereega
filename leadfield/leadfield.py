
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import re
import warnings
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
    """Original channel index (before removing the channel)"""
    
    ref: str = ''
    """Reference channel (e.g. for re-referencing)"""
    
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
                      source_idx: Union[int, np.ndarray, List[int]], 
                      orientation: Optional[np.ndarray] = None,
                      normalize_leadfield: bool = False,
                      normalize_orientation : bool = False
        ) -> np.ndarray:
        """
        Get the projection pattern for source(s) with given orientation.
        
        Returns the (oriented) projection matrix of given source(s) in the 
        leadfield, using the optionally given orientation and normalization 
        arguments.
        
        Parameters
        ----------
        source_idx : int, np.ndarray, or List[int]
            Single source index or array of source indices (0 to n_sources-1).
            If array, the mean projection of all sources will be returned.
        orientation : np.ndarray, optional
            Dipole orientation(s) [x, y, z]. If None, uses default orientation 
            from the leadfield. Shape can be:
            - (3,) for single orientation applied to all sources
            - (n_sources, 3) for individual orientations per source
        normalize_leadfield : bool, default=False
            Whether to normalize the leadfield before projecting to have the 
            most extreme value be either 1 or -1, depending on its sign.
        normalize_orientation : bool, default=False
            Whether to normalize the orientation vector(s) as above.
            
        Returns
        -------
        np.ndarray
            Projection pattern of shape (n_channels,) representing the 
            topographic distribution on the scalp.
            
        Examples
        --------
        >>> lf = LeadField(...)
        >>> # Get projection with default orientation
        >>> proj = lf.get_projection(100)
        >>> 
        >>> # Get projection with custom orientation (anterior-posterior)
        >>> proj = lf.get_projection(100, orientation=[0, 1, 0])
        >>> 
        >>> # Get mean projection from multiple sources
        >>> proj = lf.get_projection([100, 101, 102])
        >>> 
        >>> # Multiple sources with individual orientations
        >>> orientations = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> proj = lf.get_projection([100, 101, 102], orientation=orientations)
        """
        
        source_idx = np.atleast_1d(source_idx)
        n_sources = source_idx.shape[0]
        
        if np.any(source_idx < 0) or np.any(source_idx >= self.n_sources):
            raise ValueError(f"Source index out of range [0, {self.n_sources}). Got: {source_idx}")
        if source_idx.ndim > 1:
            raise ValueError(f"Source index must be 1D array, list or scalar. Got shape: {source_idx.shape}")
        
        if orientation is None:
            orientation = self.orientation[source_idx] # (n_sources, 3)
        else:
            orientation = np.atleast_2d(orientation)
            
            if orientation.shape[0] == 1 and n_sources > 1:
                warnings.warn(
                    f'Only one orientation indicated for {n_sources} sources; '
                    f'applying that same orientation to all sources')
                orientation = np.repeat(orientation, n_sources, axis=0) # tested: 'repeat' is much faster than 'tile'
            
        if orientation.shape != (n_sources, 3):
            raise ValueError(
            f"Orientation shape {orientation.shape} doesn't match "
            f"expected ({n_sources}, 3)")
                    
        # Extract leadfield for this source: (n_channels, 3)
        lf_sources = self.leadfield[:, source_idx, :].copy()
        
        if normalize_leadfield:
            max_vals = np.abs(lf_sources).max(axis=(0, 2), keepdims=True)
            lf_sources = np.where(max_vals > 0, lf_sources / max_vals, lf_sources)

        if normalize_orientation:
            max_vals = np.abs(orientation).max(axis=1, keepdims=True)
            orientation = np.where(max_vals > 0, orientation / max_vals, orientation)
        
        # Vectorized projection computation using Einstein summations
        # This is the core operation and is highly optimized
        # lf_sources: (n_channels, n_sources, 3) and orientation: (n_sources, 3)
        # -> result: (n_channels, n_sources)
        projections = np.einsum('ijk,jk->ij', lf_sources, orientation)
    
        return projections.mean(axis=1) if n_sources > 1 else projections[:, 0]
    
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
        
    def get_regions(self, cats : List[str] = {'brain', 'eye', 'muscle'}) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Return a tuple of generic region categories present in the leadfield's atlas
        as well as a list of unique regions with the respective counters.
        
        Parameters
        ----------
        cats : List[str]
            List of region categories to search for (case-insensitive).
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict[str, int]] where:
            allregions ()- cell of strings listing all unique regions present in
                        the atlas
            numall - numeric list indicating how many sources are present in
                    each of the regions in the allregions list
            generic_counts 
            Dictionary of counts for each generic category in cats.
            Array of source indices in the specified regions.
            
        Raises
        ------
        ValueError
            If no atlas is available.
            
        Examples
        --------       
        """
        if self.atlas is None:
            raise ValueError("No atlas available for this lead field")
        
        atlas = np.asarray(self.atlas, dtype=np.str_)
        
        allregions, numall = np.unique(atlas, return_counts=True)
        
        lower = np.char.lower(atlas)
        generic_counts = {}
        
        for c in cats:
            mask = np.char.startswith(lower, c)
            generic_counts[c] = np.sum(mask)
        
        return allregions, numall, generic_counts
        
    def get_source_all(self, region : Union[List[str], str] = '.*') ->np.ndarray: # alias of get_sources_in_region
        if isinstance(region, str):
            region = [region]
        return self.get_sources_in_region(region_patterns=region)
        
    def get_sources_in_region(self, region_patterns: Union[List[str], str]) -> np.ndarray:
        """
        Get indices of sources in specified anatomical regions. 
        If multiple region patterns match multiple sources in the atlas, all matchin sources indices are returned.
        If none of the region patterns match any source in the atlas, an empty array is returned. 
        
        Parameters
        ----------
        region_patterns : List[str]
            List of region patterns to search for (case-insensitive).
            
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
        if isinstance(region_patterns, str):
            region_patterns = [region_patterns]
        if region_patterns is None:
            region_patterns = ['.*'] # matchinig all regions
        if self.atlas is None:
            raise ValueError("No atlas available for this lead field")
        
        allregions, _, _, = self.get_regions()
        idx = np.zeros(len(self.atlas), dtype=bool)
        
        for region in region_patterns:
            if region in allregions:
                region = re.escape(region)
                region = f'^{region}$'
                print(f'Assuming exact match: {region}')
                
            matches = np.array([bool(re.search(region, entry, re.IGNORECASE)) for entry in self.atlas])
            idx = idx | matches
        return np.nonzero(idx)[0]
    
    def get_source_random(self, number: int = 1, region_patterns: List[str] = '.*') -> np.ndarray:
        
        region_idxs = self.get_source_all(region=region_patterns)
        if len(region_idxs) == 0:
            return np.array([], dtype=int)
        return np.random.choice(region_idxs, size=min(number, len(region_idxs)), replace=False)
        
    
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