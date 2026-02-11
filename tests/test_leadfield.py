"""
Tests for lead field loaders.

Run with: pytest test_leadfield.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from leadfield import (
    LeadField,
    ChannelLocation,
    generate_leadfield_mne,
    calculate_pseudoperpendicular_orientation,
    calculate_pseudotangential_orientation,
    sanitize_atlas,
)


class TestChannelLocation:
    """Tests for ChannelLocation dataclass."""
    
    def test_creation(self):
        """Test basic channel location creation."""
        ch = ChannelLocation(
            labels='Fz',
            X=0.0,
            Y=85.0,
            Z=40.0,
            theta=0.0,
            radius=95.0
        )
        
        assert ch.labels == 'Fz'
        assert ch.X == 0.0
        assert ch.Y == 85.0
        assert ch.type == 'EEG'
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        ch = ChannelLocation(labels='Cz')
        d = ch.to_dict()
        
        assert isinstance(d, dict)
        assert d['labels'] == 'Cz'
        assert 'X' in d
        assert 'Y' in d


class TestLeadField:
    """Tests for LeadField class."""
    
    def test_creation(self):
        """Test basic lead field creation."""
        n_channels, n_sources = 3, 5
        
        lf = LeadField(
            leadfield=np.random.randn(n_channels, n_sources, 3),
            pos=np.random.randn(n_sources, 3) * 50,
            orientation=np.random.randn(n_sources, 3),
            chanlocs=[
                ChannelLocation(labels=f'Ch{i+1}')
                for i in range(n_channels)
            ],
            method='test',
            source='test'
        )
        
        assert lf.n_channels == n_channels
        assert lf.n_sources == n_sources
        assert len(lf.channel_labels) == n_channels
    
    def test_validation_wrong_shape(self):
        """Test that validation catches incorrect shapes."""
        with pytest.raises(ValueError, match="must have 3 orientations"):
            LeadField(
                leadfield=np.random.randn(3, 5, 2),  # Wrong last dim
                pos=np.random.randn(5, 3),
                orientation=np.random.randn(5, 3),
                chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(3)]
            )
    
    def test_validation_mismatched_sources(self):
        """Test that validation catches mismatched source counts."""
        with pytest.raises(ValueError, match="doesn't match"):
            LeadField(
                leadfield=np.random.randn(3, 5, 3),
                pos=np.random.randn(4, 3),  # Wrong number of sources
                orientation=np.random.randn(5, 3),
                chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(3)]
            )
    
    def test_get_projection_default(self):
        """Test getting projection with default orientation."""
        lf = LeadField(
            leadfield=np.array([
                [[1, 0, 0], [0, 1, 0]],
                [[0, 1, 0], [1, 0, 0]]
            ]),  # 2 channels, 2 sources, 3 orientations
            pos=np.array([[0, 0, 0], [10, 0, 0]]),
            orientation=np.array([[1, 0, 0], [0, 1, 0]]),
            chanlocs=[ChannelLocation(labels='Ch1'), ChannelLocation(labels='Ch2')]
        )
        
        # Source 0 with default orientation [1, 0, 0]
        proj = lf.get_projection(0)
        expected = lf.leadfield[:, 0, :] @ np.array([1, 0, 0])
        np.testing.assert_array_equal(proj, expected)
    
    def test_get_projection_custom(self):
        """Test getting projection with custom orientation."""
        lf = LeadField(
            leadfield=np.random.randn(3, 2, 3),
            pos=np.random.randn(2, 3) * 50,
            orientation=np.zeros((2, 3)),
            chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(3)]
        )
        
        custom_orient = np.array([0, 1, 0])
        proj = lf.get_projection(0, orientation=custom_orient)
        
        expected = lf.leadfield[:, 0, :] @ custom_orient
        np.testing.assert_array_almost_equal(proj, expected)
    
    def test_normalize(self):
        """Test lead field normalization."""
        lf = LeadField(
            leadfield=np.random.randn(3, 5, 3) * 100,  # Large values
            pos=np.random.randn(5, 3) * 50,
            orientation=np.random.randn(5, 3),
            chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(3)]
        )
        
        lf_norm = lf.normalize(method='norm')
        
        # Check that values are smaller
        assert np.abs(lf_norm.leadfield).max() < np.abs(lf.leadfield).max()
        
        # Check that it's a new object
        assert lf_norm is not lf
    
    def test_save_load_npz(self):
        """Test saving and loading lead field."""
        lf_original = LeadField(
            leadfield=np.random.randn(3, 5, 3),
            pos=np.random.randn(5, 3) * 50,
            orientation=np.random.randn(5, 3),
            chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(3)],
            atlas=['Brain_Region1'] * 5,
            method='test',
            source='test'
        )
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            lf_original.save(temp_path)
            
            # Load
            lf_loaded = LeadField.load(temp_path)
            
            # Check equality
            np.testing.assert_array_equal(
                lf_original.leadfield,
                lf_loaded.leadfield
            )
            np.testing.assert_array_equal(lf_original.pos, lf_loaded.pos)
            assert lf_original.method == lf_loaded.method
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_get_sources_in_region(self):
        """Test region-based source selection."""
        lf = LeadField(
            leadfield=np.random.randn(3, 5, 3),
            pos=np.random.randn(5, 3) * 50,
            orientation=np.random.randn(5, 3),
            chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(3)],
            atlas=[
                'Brain_Visual_Cortex',
                'Brain_Visual_Cortex',
                'Brain_Motor_Cortex',
                'Brain_Frontal_Lobe',
                'Brain_Visual_Area'
            ]
        )
        
        # Find visual sources
        visual_sources = lf.get_sources_in_region(['visual'])
        assert len(visual_sources) == 3  # Indices 0, 1, 4
        
        # Find motor sources
        motor_sources = lf.get_sources_in_region(['motor'])
        assert len(motor_sources) == 1  # Index 2
    
    def test_repr(self):
        """Test string representation."""
        lf = LeadField(
            leadfield=np.random.randn(64, 2000, 3),
            pos=np.random.randn(2000, 3) * 50,
            orientation=np.random.randn(2000, 3),
            chanlocs=[ChannelLocation(labels=f'Ch{i}') for i in range(64)],
            method='test',
            source='test'
        )
        
        repr_str = repr(lf)
        assert 'channels=64' in repr_str
        assert 'sources=2000' in repr_str
        assert 'test' in repr_str


class TestOrientationFunctions:
    """Tests for orientation calculation functions."""
    
    def test_pseudoperpendicular(self):
        """Test pseudo-perpendicular orientation calculation."""
        # Test with point on positive X axis
        pos = np.array([100, 0, 0])
        orient = calculate_pseudoperpendicular_orientation(pos)
        
        # Should point in X direction
        np.testing.assert_array_almost_equal(orient, [1, 0, 0])
        
        # Should be normalized
        assert np.abs(np.linalg.norm(orient) - 1.0) < 1e-10
    
    def test_pseudotangential_horizontal(self):
        """Test pseudo-tangential horizontal orientation."""
        pos = np.array([100, 0, 0])
        orient = calculate_pseudotangential_orientation(pos, 'horizontal')
        
        # Should be perpendicular to radial in XY plane
        # and normalized
        assert np.abs(np.linalg.norm(orient) - 1.0) < 1e-10
        assert np.abs(orient[2]) < 1e-10  # Z component should be zero
    
    def test_pseudotangential_all_directions(self):
        """Test all tangential directions."""
        pos = np.array([20, -75, 0])
        
        for direction in ['horizontal', 'sagittal', 'coronal']:
            orient = calculate_pseudotangential_orientation(pos, direction)
            
            # Should be normalized
            assert np.abs(np.linalg.norm(orient) - 1.0) < 1e-10


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_sanitize_atlas_already_valid(self):
        """Test atlas sanitization with already valid labels."""
        atlas = ['Brain_Visual_Cortex', 'Eye_Left', 'Muscle_Temporal']
        sanitized = sanitize_atlas(atlas)
        
        assert sanitized == atlas
    
    def test_sanitize_atlas_add_prefix(self):
        """Test atlas sanitization adds Brain prefix."""
        atlas = ['Visual_Cortex', 'Motor_Cortex']
        sanitized = sanitize_atlas(atlas)
        
        assert sanitized == ['Brain_Visual_Cortex', 'Brain_Motor_Cortex']
    
    def test_sanitize_atlas_case_insensitive(self):
        """Test atlas sanitization is case-insensitive."""
        atlas = ['BRAIN_Test', 'eye_test', 'MuScLe_Test']
        sanitized = sanitize_atlas(atlas)
        
        # Should keep original case
        assert sanitized == atlas


@pytest.mark.slow
class TestMNELoader:
    """
    Tests for MNE lead field generation.
    
    These tests require MNE-Python and may be slow.
    Mark as slow to skip in quick test runs.
    """
    
    def test_generate_minimal(self):
        """Test generating minimal lead field with MNE."""
        pytest.importorskip('mne')
        
        lf = generate_leadfield_mne(
            montage=['Fz', 'Cz', 'Pz'],
            spacing='oct5',
            normalize=False,
            verbose=False
        )
        
        assert lf.n_channels == 3
        assert lf.n_sources > 0
        assert lf.method == 'mne-python'
    
    def test_generate_with_normalization(self):
        """Test MNE generation with normalization."""
        pytest.importorskip('mne')
        
        lf = generate_leadfield_mne(
            montage=['Cz'],
            spacing='oct5',
            normalize=True,
            verbose=False
        )
        
        assert 'normalized' in lf.unit


def test_module_imports():
    """Test that all main components can be imported."""
    from leadfield import (
        LeadField,
        ChannelLocation,
        generate_leadfield_mne,
        load_matlab_leadfield,
        get_montage_labels,
        sanitize_atlas,
    )
    
    # Just check they're callable/instantiable
    assert callable(generate_leadfield_mne)
    assert callable(load_matlab_leadfield)
    assert callable(get_montage_labels)
    assert callable(sanitize_atlas)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
