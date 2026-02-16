"""test_get_source_inradius.py"""
import unittest
import numpy as np
from leadfield import lf_generate_from_nyhead, LeadField
from utils import EngineWrapper


class TestInRadiusVsMATLAB(unittest.TestCase):
    """Validate Python get_source_inradius against MATLAB implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Load leadfield once for all tests."""
        cls.eng = EngineWrapper()
        cls.lf: LeadField = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'], eng=cls.eng)
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup MATLAB engine."""
        cls.eng.quit()
    
    def _get_matlab_inradius(self, centre, radius, region='.*'):
        """Get sources in radius from MATLAB for comparison."""
        eng = self.eng
        
        # Transfer leadfield to MATLAB
        eng.workspace['lf_mat'] = {
            'leadfield': self.lf.leadfield.astype(np.float64),
            'orientation': self.lf.orientation.astype(np.float64),
            'pos': self.lf.pos.astype(np.float64),
            'atlas': list(self.lf.atlas)
        }
        eng.workspace['lf_atlas'] = self.lf.atlas
        eng.eval("lf_mat.atlas = lf_atlas'", nargout=0) 
        
        centre = np.atleast_1d(centre)
        if centre.size == 1:
            centre_str = str(int(centre[0]) + 1)
        else:
            centre_arr = centre.astype(np.float64)
            eng.workspace['centre'] = centre_arr
            centre_str = 'centre'
        
        if isinstance(region, str):
            region = [region]
        eng.workspace['region_cell'] = region
        
        eng.eval(f"result = lf_get_source_inradius(lf_mat, {centre_str}, {radius}, 'region', region_cell);",nargout=0)
        result = eng.workspace['result']
               
        if result is None:
            return np.array([], dtype=int)
        
        result_np = np.asarray(result)
        if result_np.size == 0:
            return np.array([], dtype=int)
        
        result_array = result_np.flatten().astype(int) - 1
        return result_array
    
    def test_centre_as_coordinates_small_radius(self):
        """Test with coordinate centre and small radius."""
        centre = [0, 0, 0]
        radius = 10
        
        sources_py = self.lf.get_source_inradius(centre, radius)
        sources_mat = self._get_matlab_inradius(centre, radius)
        np.testing.assert_array_equal(np.sort(sources_py), np.sort(sources_mat),
            err_msg="Source indices differ for coordinate centre with small radius")
    
    def test_centre_as_coordinates_large_radius(self):
        """Test with coordinate centre and large radius."""
        centre = [20, -10, 30]
        radius = 50
        
        sources_py = self.lf.get_source_inradius(centre, radius)
        sources_mat = self._get_matlab_inradius(centre, radius)
        
        np.testing.assert_array_equal(np.sort(sources_py), np.sort(sources_mat),
            err_msg="Source indices differ for coordinate centre with large radius")
    
    def test_centre_as_source_index(self):
        """Test with source index as centre."""
        centre_idx = 1000
        radius = 25
        
        sources_py = self.lf.get_source_inradius(centre_idx, radius)
        sources_mat = self._get_matlab_inradius(centre_idx, radius)
        
        np.testing.assert_array_equal(
            np.sort(sources_py), np.sort(sources_mat),
            err_msg="Source indices differ when using source index as centre"
        )
    
    def test_with_region_filter(self):
        """Test with region filtering."""
        centre = [0, 0, 0]
        radius = 30
        region = 'Brain.*'
        
        sources_py = self.lf.get_source_inradius(centre, radius, region=region)
        sources_mat = self._get_matlab_inradius(centre, radius, region=region)
        
        np.testing.assert_array_equal(np.sort(sources_py), np.sort(sources_mat),
            err_msg="Source indices differ with region filter")
    
    def test_zero_radius(self):
        """Test with zero radius (should return only exact match or empty)."""
        centre = self.lf.pos[500]  # Use exact position of a source
        radius = 0
        
        sources_py = self.lf.get_source_inradius(centre, radius)
        sources_mat = self._get_matlab_inradius(centre, radius)
        
        np.testing.assert_array_equal(np.sort(sources_py), np.sort(sources_mat),
            err_msg="Source indices differ with zero radius")
    
    def test_very_large_radius(self):
        """Test with very large radius (should return many sources)."""
        centre = [0, 0, 0]
        radius = 200
        
        sources_py = self.lf.get_source_inradius(centre, radius)
        sources_mat = self._get_matlab_inradius(centre, radius)
        
        np.testing.assert_array_equal(np.sort(sources_py), np.sort(sources_mat),
            err_msg="Source indices differ with large radius")


class TestInRadiusProperties(unittest.TestCase):
    """Test mathematical and logical properties of get_source_inradius."""
    
    @classmethod
    def setUpClass(cls):
        """Load leadfield once for all tests."""
        cls.lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'])
    
    def test_returns_array(self):
        """Test that method returns numpy array."""
        result = self.lf.get_source_inradius([0, 0, 0], 20)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, int)
    
    def test_distances_within_radius(self):
        """Test that all returned sources are actually within radius."""
        centre = np.array([10, -5, 15])
        radius = 30
        
        sources = self.lf.get_source_inradius(centre, radius)
        
        for src_idx in sources:
            src_pos = self.lf.pos[src_idx]
            dist = np.linalg.norm(src_pos - centre)
            self.assertLessEqual(dist, radius,f"Source {src_idx} at distance {dist} exceeds radius {radius}")
    
    def test_no_sources_excluded_within_radius(self):
        """Test that no sources within radius are excluded."""
        centre = np.array([0, 0, 0])
        radius = 40
        
        sources = self.lf.get_source_inradius(centre, radius)
        sources_set = set(sources)
        
        # Check all sources manually
        for idx in range(self.lf.n_sources):
            dist = np.linalg.norm(self.lf.pos[idx] - centre)
            if dist <= radius:
                self.assertIn(idx, sources_set, f"Source {idx} at distance {dist} should be included but isn't")
    
    def test_increasing_radius_includes_more_sources(self):
        """Test that larger radius includes more or equal sources."""
        centre = [0, 0, 0]
        
        sources_10 = self.lf.get_source_inradius(centre, 10)
        sources_20 = self.lf.get_source_inradius(centre, 20)
        sources_30 = self.lf.get_source_inradius(centre, 30)
        
        self.assertLessEqual(len(sources_10), len(sources_20))
        self.assertLessEqual(len(sources_20), len(sources_30))
    
    def test_centre_as_source_includes_itself(self):
        """Test that using source index as centre includes that source."""
        centre_idx = 1000
        radius = 1  # Very small radius
        sources = self.lf.get_source_inradius(centre_idx, radius)
        self.assertIn(centre_idx, sources)
    
    def test_empty_region_returns_empty(self):
        """Test that empty region returns empty array."""
        centre = [0, 0, 0]
        radius = 50
        region = 'NonExistentRegion_XYZ123'
        sources = self.lf.get_source_inradius(centre, radius, region=region)
        self.assertEqual(len(sources), 0)
    
    def test_invalid_centre_raises_error(self):
        """Test that invalid centre raises error."""
        with self.assertRaises(ValueError):
            self.lf.get_source_inradius([1, 2], 10)  # Only 2 coordinates
        
        with self.assertRaises(ValueError):
            self.lf.get_source_inradius([1, 2, 3, 4], 10)  # Too many coordinates
    
    def test_negative_radius_behavior(self):
        """Test behavior with negative radius (implementation-dependent)."""
        centre = [0, 0, 0]
        # This might return empty or raise error depending on implementation
        # Just test it doesn't crash
        try:
            sources = self.lf.get_source_inradius(centre, -10)
            # If doesn't raise check result is valid
            self.assertIsInstance(sources, np.ndarray)
        except ValueError:
            pass  # Also acceptable


if __name__ == '__main__':
    unittest.main(verbosity=2)