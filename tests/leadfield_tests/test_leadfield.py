"""
Tests for lead field classes (LeadField and ChannelLocation).

Run with: pytest test_leadfield.py -v
"""
import unittest
import numpy as np
from pathlib import Path
import tempfile

from leadfield import (
    LeadField,
    ChannelLocation,
)


class TestChannelLocation(unittest.TestCase):
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
        
        self.assertEqual(ch.labels, 'Fz')
        self.assertEqual(ch.X, 0.0)
        self.assertEqual(ch.Y, 85.0)
        self.assertEqual(ch.type, 'EEG')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        ch = ChannelLocation(labels='Cz')
        d = ch.to_dict()
        
        self.assertTrue(isinstance(d, dict))
        self.assertEqual(d['labels'], 'Cz')
        self.assertIn('X', d)
        self.assertIn('Y', d)


class TestLeadField(unittest.TestCase):
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
        
        self.assertEqual(lf.n_channels, n_channels)
        self.assertEqual(lf.n_sources, n_sources)
        self.assertEqual(len(lf.channel_labels), n_channels)
    
    def test_validation_wrong_shape(self):
        """Test that validation catches incorrect shapes."""
        with self.assertRaisesRegex(ValueError, "must have 3 orientations"):
            LeadField(
                leadfield=np.random.randn(3, 5, 2),  # Wrong last dim
                pos=np.random.randn(5, 3),
                orientation=np.random.randn(5, 3),
                chanlocs=[ChannelLocation(labels=f"Ch{i}") for i in range(3)],
            )

    
    def test_validation_mismatched_sources(self):
        """Test that validation catches mismatched source counts."""
        with self.assertRaisesRegex(ValueError, "doesn't match"):
            LeadField(
                leadfield=np.random.randn(3, 5, 3),
                pos=np.random.randn(4, 3),  # Wrong number of sources
                orientation=np.random.randn(5, 3),
                chanlocs=[ChannelLocation(labels=f"Ch{i}") for i in range(3)],
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
        self.assertLess(np.abs(lf_norm.leadfield).max(), np.abs(lf.leadfield).max())
        
        # Check that it's a new object
        self.assertTrue(lf_norm is not lf)
    
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
            np.testing.assert_array_equal(lf_original.leadfield,lf_loaded.leadfield)
            np.testing.assert_array_equal(lf_original.pos, lf_loaded.pos)
            self.assertEqual(lf_original.method, lf_loaded.method)
            
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
        self.assertTrue(len(visual_sources), 3)  # Indices 0, 1, 4
        
        # Find motor sources
        motor_sources = lf.get_sources_in_region(['motor'])
        self.assertTrue(len(motor_sources), 1)  # Index 2
    
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
        self.assertIn('channels=64', repr_str)
        self.assertIn('sources=2000', repr_str)
        self.assertIn('test', repr_str)


if __name__ == '__main__':
    unittest.main([__file__, '-v'])
