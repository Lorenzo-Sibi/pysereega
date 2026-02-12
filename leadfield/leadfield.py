
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
        
    def get_source_all(self, region : Union[List[str], np.ndarray, str] = '.*') ->np.ndarray: # alias of get_sources_in_region
        if isinstance(region, str):
            region = [region]
        return self.get_sources_in_region(region_patterns=region)
        
    def get_sources_in_region(self, region_patterns: Union[List[str], np.ndarray, str]) -> np.ndarray:
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
        if (not isinstance(region_patterns, str)) and (not isinstance(region_patterns, np.ndarray)) and (not isinstance(region_patterns, list)):
            raise ValueError("'region_patterns' attribute should be string, list or np.ndarray of strings.")
        if isinstance(region_patterns, str):
            region_patterns = [region_patterns]
        if region_patterns is None:
            region_patterns = ['.*'] # matchinig all regions
        if self.atlas is None:
            raise ValueError("No atlas available for this lead field")
        
        allregions, _, _ = self.get_regions()
        idx = np.zeros(len(self.atlas), dtype=bool)
        
        for region in region_patterns:
            if region in allregions:
                region = re.escape(region)
                region = f'^{region}$'
                print(f'Assuming exact match: {region}')
                
            matches = np.array([bool(re.search(region, entry, re.IGNORECASE)) for entry in self.atlas])
            idx = idx | matches
        return np.nonzero(idx)[0]
    
    def get_source_inradius(self, 
                            centre: Union[List[float], np.ndarray], 
                            radius: float, 
                            region: Union[List[str], np.ndarray, str] = '.*'
        ) -> np.ndarray:
        """
        Get sources within a certain radius of an indicated source or coordinate.
        
        Returns the source(s) in the leadfield within a certain radius of
        an indicated source or coordinate.
        
        Parameters
        ----------
        centre : Union[List[float], np.ndarray, int]
            Either:
            - 1D array/list of [x, y, z] coordinates in mm (MNI coordinates)
            - Single integer representing a source index
        radius : float
            The radius in which to search, in mm
        region : Union[List[str], np.ndarray, str], optional
            Region pattern(s) to filter sources. Default '.*' matches all regions.
            Can be a single pattern string, or list of patterns.
            
        Returns
        -------
        np.ndarray
            Array of source indices within the indicated area (radius)
            
        Raises
        ------
        ValueError
            If centre is not a valid source index or 3D coordinate.
            If no atlas is available when region filtering is requested.
            
        Examples
        --------
        >>> lf = lf_generate_from_nyhead()
        >>> # Get sources within 10mm of coordinate [0, 0, 0]
        >>> sources = lf.get_source_inradius([0, 0, 0], 10)
        >>> 
        >>> # Get sources within 15mm of source index 100
        >>> sources = lf.get_source_inradius(100, 15)
        >>> 
        >>> # Get sources in visual cortex within 10mm of a point
        >>> sources = lf.get_source_inradius([0, -50, 0], 10, region='Visual')
        >>> 
        >>> # Multiple region patterns
        >>> sources = lf.get_source_inradius([0, 0, 0], 20, region=['Motor', 'Sensory'])
        
        Notes
        -----
        This function uses Euclidean distance in 3D space to determine which
        sources fall within the specified radius. The coordinate system follows
        MNI conventions (typically X=left-right, Y=posterior-anterior, 
        Z=inferior-superior).
        """
        region_idx = self.get_sources_in_region(region_patterns=region)
        
        if len(region_idx) == 0:
            return np.array([], dtype=int)
    
        centre = np.atleast_1d(centre)
        
        if centre.size == 1:
            # Single index - get coordinates from that source
            source_idx = int(centre[0])
            if source_idx < 0 or source_idx >= self.n_sources:
                raise ValueError(
                    f"One value provided for centre. Interpreting centre as source index"
                    f"Source index {source_idx} out of range [0, {self.n_sources})")
            centre = self.pos[source_idx, :]
        elif centre.size == 3:
            centre = centre.flatten()
        else:
            raise ValueError(
                f"Centre must be either a single source index or [x, y, z] coordinates. "
                f"Got array with {centre.size} elements")
        
        # Calculate Euclidean distances for sources in region
        # Vectorized computation (much faster than MATLAB loop)
        # pos shape: (n_sources, 3), centre shape: (3,)
        positions = self.pos[region_idx, :]  # (n_region_sources, 3)
        
        # Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
        distances = np.sqrt(np.sum((positions - centre)**2, axis=1))
        
        # Filter sources within radius
        within_radius = distances <= radius
        sources_in_radius = region_idx[within_radius]
        
        return sources_in_radius
        
    def get_source_nearest(self, pos: Union[List[float], np.ndarray], 
                      region: Union[List[str], np.ndarray, str] = '.*') -> Tuple[int, float]:
        """
        Get the source nearest to the given position.
        
        Returns the source in the leadfield nearest to the given position,
        and its distance from that position. The returned source can
        optionally be constrained to indicated region(s).
        
        Parameters
        ----------
        pos : Union[List[float], np.ndarray]
            1D array/list of [x, y, z] coordinates in mm (MNI coordinates)
        region : Union[List[str], np.ndarray, str], optional
            Region pattern(s) to filter sources. Default '.*' matches all regions.
            Can be a single pattern string, or list of patterns.
            
        Returns
        -------
        Tuple[int, float]
            source_idx : int
                The nearest source index
            dist : float
                The distance of the found source to the indicated position (in mm)
                
        Raises
        ------
        ValueError
            If pos is not a valid 3D coordinate.
            If no sources found in specified region(s).
            
        Examples
        --------
        >>> lf = lf_generate_from_nyhead()
        >>> # Get source nearest to origin
        >>> source_idx, distance = lf.get_source_nearest([0, 0, 0])
        >>> print(f"Source {source_idx} is {distance:.2f} mm from origin")
        >>> 
        >>> # Get nearest source in motor cortex to a specific point
        >>> source_idx, dist = lf.get_source_nearest([20, -10, 60], region='Motor')
        >>> 
        >>> # Find nearest source in multiple regions
        >>> idx, dist = lf.get_source_nearest([0, 0, 0], region=['Brain.*Visual', 'Brain.*Auditory'])
        
        Notes
        -----
        Uses Euclidean distance in 3D space. Coordinate system follows MNI 
        conventions (typically X=left-right, Y=posterior-anterior, 
        Z=inferior-superior).
        """
        # Validate and convert pos to numpy array
        pos = np.atleast_1d(pos).flatten()
        if pos.size != 3:
            raise ValueError(
                f"Position must be [x, y, z] coordinates. "
                f"Got array with {pos.size} elements"
            )
        
        # Get sources in specified region(s)
        region_idx = self.get_sources_in_region(region_patterns=region)
        
        # Check if any sources found
        if len(region_idx) == 0:
            raise ValueError(
                f"No sources found in region(s): {region}. "
                f"Cannot find nearest source."
            )
        
        # Calculate Euclidean distances (vectorized)
        positions = self.pos[region_idx, :]  # (n_region_sources, 3)
        distances = np.sqrt(np.sum((positions - pos)**2, axis=1))
        
        # Find minimum distance and corresponding index
        min_idx = np.argmin(distances)
        source_idx = region_idx[min_idx]
        dist = distances[min_idx]
        
        return int(source_idx), float(dist)


    def get_source_middle(self, region: Union[List[str], np.ndarray, str] = '.*', 
                        method: str = 'average') -> int:
        """
        Get the source nearest to the middle of indicated region(s).
        
        Returns the source nearest to either the average coordinates of all
        sources in the indicated region(s), or to the average of their
        boundaries (min/max along each axis).
        
        Note: Regardless of method, this may or may not represent an actual
        geometric 'middle', especially for irregularly shaped regions.
        
        Parameters
        ----------
        region : Union[List[str], np.ndarray, str], optional
            Region pattern(s) to search. Default '.*' matches all regions.
            Can be a single pattern string, or list of patterns.
        method : str, optional
            Method to calculate the 'middle'. Options:
            - 'average' (default): Mean of all source coordinates
            - 'minmax': Mean of minimum and maximum values along each axis
            
        Returns
        -------
        int
            The index of the source nearest to the calculated middle point
                
        Raises
        ------
        ValueError
            If fewer than 3 sources found in region (cannot determine middle).
            If unknown method specified.
            
        Examples
        --------
        >>> lf = lf_generate_from_nyhead()
        >>> # Get middle source of entire brain using average method
        >>> middle = lf.get_source_middle(region='Brain.*')
        >>> 
        >>> # Get middle of visual cortex using min-max boundaries
        >>> middle = lf.get_source_middle(region='Visual', method='minmax')
        >>> 
        >>> # Get middle across multiple regions
        >>> middle = lf.get_source_middle(region=['Motor', 'Sensory'], method='average')
        
        Notes
        -----
        The 'average' method computes the centroid (mean position) of all sources
        in the region, then finds the source nearest to that point.
        
        The 'minmax' method computes the center of the bounding box by taking
        the mean of [max(x), max(y), max(z)] and [min(x), min(y), min(z)],
        then finds the source nearest to that point.
        """
        # Get sources in specified region(s)
        source_indices = self.get_sources_in_region(region_patterns=region)
        
        # Check minimum number of sources
        if len(source_indices) < 3:
            raise ValueError(
                f"Cannot determine 'middle' from {len(source_indices)} source(s). "
                f"At least 3 sources required in region: {region}"
            )
        
        # Get positions of sources in region
        positions = self.pos[source_indices, :]  # (n_sources, 3)
        
        # Calculate middle point based on method
        if method == 'average':
            # Mean of all coordinates
            middle_point = np.mean(positions, axis=0)
            
        elif method == 'minmax':
            # Mean of bounding box (min and max along each axis)
            max_coords = np.max(positions, axis=0)  # (3,)
            min_coords = np.min(positions, axis=0)  # (3,)
            middle_point = np.mean([max_coords, min_coords], axis=0)
            
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Valid options are: 'average', 'minmax'"
            )
        
        # Find source nearest to the calculated middle point
        source_idx, _ = self.get_source_nearest(middle_point, region=region)
        
        return int(source_idx)
    
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

            def _unwrap_np_scalar(value):
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    return value.item()
                return value

            chanlocs = [
                ChannelLocation(**_unwrap_np_scalar(ch)) for ch in data['chanlocs']
            ]
            atlas_raw = _unwrap_np_scalar(data['atlas']) if 'atlas' in data else None
            if atlas_raw is None:
                atlas = None
            elif isinstance(atlas_raw, np.ndarray):
                atlas = atlas_raw.tolist()
            else:
                atlas = list(atlas_raw)

            metadata_raw = _unwrap_np_scalar(data['metadata']) if 'metadata' in data else {}
            if metadata_raw is None:
                metadata = {}
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = dict(metadata_raw)

            return cls(
                leadfield=data['leadfield'],
                pos=data['pos'],
                orientation=data['orientation'],
                chanlocs=chanlocs,
                atlas=atlas,
                method=str(_unwrap_np_scalar(data['method']) if 'method' in data else 'unknown'),
                source=str(_unwrap_np_scalar(data['source']) if 'source' in data else 'unknown'),
                unit=str(_unwrap_np_scalar(data['unit']) if 'unit' in data else 'relative'),
                metadata=metadata
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