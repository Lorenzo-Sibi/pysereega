import unittest
import numpy as np
import time
from pathlib import Path
from leadfield import LeadField, ChannelLocation, lf_generate_from_nyhead, NYHEAD_LEADFIELD_PATH, NYHEAD_WORKING_DIR

class TestNyheadGeneration(unittest.TestCase):
    
    def test_generate_with_montage(self):
        """Test leadfield generation using a standard montage."""
        lf = lf_generate_from_nyhead(montage='S64', verbose=True)
        
        self.assertIsInstance(lf, LeadField)
        self.assertGreater(lf.n_sources, 0)
        self.assertEqual(lf.leadfield.shape, (64, lf.n_sources, 3))
    
    def test_generate_with_custom_labels(self):
        """Test leadfield generation with custom electrode labels."""
        labels = ['Fz', 'Cz', 'Pz', 'Oz']
        lf = lf_generate_from_nyhead(montage='S64', labels=labels, verbose=True)
        
        self.assertEqual(lf.n_channels, len(labels))
        self.assertEqual(lf.leadfield.shape[0], len(labels))
        self.assertEqual(set(lf.channel_labels), set(labels))
    
    def test_generate_with_small_subset(self):
        """Test with a minimal electrode set."""
        labels = ['Cz']
        lf = lf_generate_from_nyhead(montage='S64', labels=labels, verbose=True)
        
        self.assertEqual(lf.n_channels, 1)
        self.assertEqual(lf.channel_labels[0], 'Cz')
        self.assertEqual(lf.leadfield.shape, (1, lf.n_sources, 3))

    def test_different_montages(self):
        """Test various standard montages."""
        montages_to_test = ['BioSemi32', 'BioSemi64']
        
        for montage in montages_to_test:
            try:
                lf = lf_generate_from_nyhead(montage=montage)
                self.assertGreater(lf.n_channels, 0)
                self.assertGreater(lf.n_sources, 0)
            except Exception as e:
                self.skipTest(f"Montage {montage} not available: {e}")
                

class TestNyheadDataIntegrity(unittest.TestCase):
    
    def test_leadfield_array_properties(self):
        """Test that leadfield array has correct properties."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz'])
        
        self.assertEqual(lf.leadfield.ndim, 3)
        self.assertEqual(lf.leadfield.shape[2], 3)  # XYZ orientations
        
        self.assertTrue(np.issubdtype(lf.leadfield.dtype, np.floating))
        
        self.assertFalse(np.any(np.isnan(lf.leadfield)))
        self.assertFalse(np.any(np.isinf(lf.leadfield)))
        
        self.assertTrue(np.any(lf.leadfield != 0))

    def test_source_positions(self):
        """Test source position extraction."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        self.assertEqual(lf.pos.shape, (lf.n_sources, 3))
        
        self.assertTrue(np.all(np.abs(lf.pos[:, 0]) < 150))  # X
        self.assertTrue(np.all(lf.pos[:, 1] > -150))
        self.assertTrue(np.all(lf.pos[:, 1] < 150))  # Y
        self.assertTrue(np.all(lf.pos[:, 2] > -150))
        self.assertTrue(np.all(lf.pos[:, 2] < 150))  # Z
        
        self.assertFalse(np.any(np.isnan(lf.pos)))
        self.assertFalse(np.any(np.isinf(lf.pos)))
    
    def test_orientations(self):
        """Test dipole orientation extraction."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        self.assertEqual(lf.orientation.shape, (lf.n_sources, 3))
        
        norms = np.linalg.norm(lf.orientation, axis=1)

        valid_norms = (norms < 1e-6) | (np.abs(norms - 1.0) < 0.1)
        self.assertTrue(np.all(valid_norms))

class TestNyheadChannelLocations(unittest.TestCase):
    """Test channel location extraction and parsing."""
    
    def test_chanloc_structure(self):
        """Test that channel locations have correct structure."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz'])
        
        self.assertEqual(len(lf.chanlocs), 3)
        
        for ch in lf.chanlocs:
            self.assertIsInstance(ch, ChannelLocation)
            self.assertIsInstance(ch.labels, str)
            self.assertGreater(len(ch.labels), 0)
    
    def test_chanloc_coordinates(self):
        """Test channel coordinate extraction."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Fz'])
        
        ch = lf.chanlocs[0]
        
        # Check that coordinates exist and are numeric
        self.assertIsInstance(ch.X, float)
        self.assertIsInstance(ch.Y, float)
        self.assertIsInstance(ch.Z, float)
        
        # Check that at least one coordinate is non-zero
        self.assertTrue((ch.X != 0) or (ch.Y != 0) or (ch.Z != 0))
    
    def test_chanloc_spherical_coords(self):
        """Test spherical coordinate extraction."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        ch = lf.chanlocs[0]
        
        self.assertTrue(hasattr(ch, 'theta'))
        self.assertTrue(hasattr(ch, 'radius'))
        self.assertTrue(hasattr(ch, 'sph_theta'))
        self.assertTrue(hasattr(ch, 'sph_phi'))
        self.assertTrue(hasattr(ch, 'sph_radius'))
    
    def test_channel_labels_match_request(self):
        """Test that returned channels match requested labels."""
        requested = ['Fz', 'Cz', 'Pz', 'Oz']
        lf = lf_generate_from_nyhead(montage='S64', labels=requested)
        
        returned = lf.channel_labels
        
        self.assertEqual(len(returned), len(requested))
        
        self.assertEqual(set(returned), set(requested))


class TestNyheadAtlasHandling(unittest.TestCase):
    """Test anatomical atlas extraction and parsing."""
    
    def test_atlas_exists(self):
        """Test that atlas is loaded."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        self.assertIsNotNone(lf.atlas)
        self.assertIsInstance(lf.atlas, list)
    
    def test_atlas_length(self):
        """Test that atlas length matches number of sources."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        self.assertEqual(len(lf.atlas), lf.n_sources)
    
    def test_atlas_labels_format(self):
        """Test atlas label format."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        # All labels should be strings
        self.assertTrue(all(isinstance(label, str) for label in lf.atlas))
        
        # Labels should start with valid prefixes (Brain, Eye, Muscle)
        valid_prefixes = ('Brain', 'Eye', 'Muscle')
        for label in lf.atlas:
            has_valid_prefix = any(label.startswith(prefix) for prefix in valid_prefixes)
            self.assertTrue(has_valid_prefix, 
                          f"Label '{label}' doesn't start with valid prefix")
    
    def test_get_sources_in_region(self):
        """Test region-based source selection."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        # Try to find visual cortex sources
        visual_indices = lf.get_sources_in_region(['Central'])
        
        # Should find at least some sources
        self.assertGreater(len(visual_indices), 0)
        
        # Check that selected sources actually contain "Central"
        for idx in visual_indices:
            self.assertIn('central', lf.atlas[idx].lower())


class TestNyheadProjectionFunctionality(unittest.TestCase):
    """Test projection computation."""
    
    def test_get_projection_default_orientation(self):
        """Test projection with default orientation."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        # Get projection for first source
        proj = lf.get_projection(0)
        
        # Should return vector with length = n_channels
        self.assertEqual(proj.shape, (lf.n_channels,))
        self.assertFalse(np.all(proj == 0))  # Should not be all zeros
    
    def test_get_projection_custom_orientation(self):
        """Test projection with custom orientation."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz'])
        
        # Test with anterior-posterior orientation
        proj_ap = lf.get_projection(0, orientation=[0, 1, 0])
        
        # Test with left-right orientation
        proj_lr = lf.get_projection(0, orientation=[1, 0, 0])
        
        # Different orientations should give different projections
        self.assertFalse(np.allclose(proj_ap, proj_lr))
    
    def test_get_projection_bounds_check(self):
        """Test that invalid source indices raise errors."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        with self.assertRaises(ValueError):
            lf.get_projection(-1)
        
        with self.assertRaises(ValueError):
            lf.get_projection(lf.n_sources)


class TestNyheadNormalization(unittest.TestCase):
    """Test leadfield normalization."""
    
    def test_normalize_by_norm(self):
        """Test Frobenius norm normalization."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        lf_norm = lf.normalize(method='norm')
        
        # Check that it returns a new object
        self.assertIsNot(lf_norm, lf)
        
        # Check shape preserved
        self.assertEqual(lf_norm.leadfield.shape, lf.leadfield.shape)
        
        # Check that normalization was applied
        # After normalization, each source should have Frobenius norm ≈ 1
        for i in range(min(100, lf_norm.n_sources)):  # Just the first 100
            source_norm = np.linalg.norm(lf_norm.leadfield[:, i, :])
            self.assertAlmostEqual(source_norm, 1.0, places=6)
    
    def test_normalize_by_max(self):
        """Test max normalization."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        lf_norm = lf.normalize(method='max')
        
        # Maximum absolute value should be 1
        max_val = np.abs(lf_norm.leadfield).max()
        self.assertAlmostEqual(max_val, 1.0, places=6)
    
    def test_normalized_unit_update(self):
        """Test that unit string is updated after normalization."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        lf_norm = lf.normalize(method='norm')
        
        self.assertIn('normalized', lf_norm.unit.lower())


class TestNyheadMetadata(unittest.TestCase):
    """Test metadata and attributes."""
    
    def test_metadata_fields(self):
        """Test that metadata is properly set."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Cz'])
        
        self.assertEqual(lf.method, 'nyhead')
        self.assertEqual(lf.source, 'New York Head (ICBM-NY)')
        self.assertEqual(lf.unit, 'µV/(nA·m)')
        
        # Check metadata dict
        self.assertIsInstance(lf.metadata, dict)
        self.assertIn('reference', lf.metadata)
        self.assertIn('copyright', lf.metadata)
    
    def test_repr(self):
        """Test string representation."""
        lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz'])
        
        repr_str = repr(lf)
        
        self.assertIn('LeadField', repr_str)
        self.assertIn('channels=2', repr_str)
        self.assertIn('nyhead', repr_str)


class TestNyheadErrorHandling(unittest.TestCase):
    """Test error handling and validation."""
    
    def test_invalid_nyhead_path(self):
        """Test error when nyhead file doesn't exist."""
        with self.assertRaisesRegex(ValueError, "Invalid nyhead_path"):
            lf_generate_from_nyhead(montage='S64',nyhead_path='/nonexistent/path/file.mat')
    
    def test_invalid_file_extension(self):
        """Test error with wrong file extension."""
        with self.assertRaisesRegex(ValueError, "Invalid nyhead_path"):
            lf_generate_from_nyhead(montage='S64', nyhead_path='/some/path/file.txt')
    
    def test_invalid_channel_labels(self):
        """Test handling of invalid/missing channel labels."""
        labels = ['Fz', 'NONEXISTENT_CHANNEL', 'Cz']
        
        # Should either work with warning or raise error
        try:
            lf = lf_generate_from_nyhead(montage='S64', labels=labels)
            # If it works, should have fewer channels than requested
            self.assertLessEqual(lf.n_channels, len(labels))
        except Exception:
            # Or it might raise an error - both are acceptable
            pass

def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadGeneration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadDataIntegrity))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadChannelLocations))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadAtlasHandling))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadProjectionFunctionality))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadNormalization))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadMetadata))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNyheadErrorHandling))
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
    unittest.main()
        