"""test_get_source_nearest.py"""
import unittest
import numpy as np
from leadfield.nyhead import lf_generate_from_nyhead, LeadField
from utils import EngineWrapper


class TestNearestVsMATLAB(unittest.TestCase):
    """Validate Python get_source_nearest against MATLAB implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Load leadfield once for all tests."""
        cls.eng = EngineWrapper()
        cls.lf: LeadField = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'], eng=cls.eng)
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup MATLAB engine."""
        cls.eng.quit()
    
    def _get_matlab_nearest(self, pos, region='.*'):
        """Get nearest source from MATLAB for comparison."""
        eng = self.eng
        
        # 'Transfer' leadfield to matlab
        eng.workspace['lf_mat'] = {
            'leadfield': self.lf.leadfield.astype(np.float64),
            'orientation': self.lf.orientation.astype(np.float64),
            'pos': self.lf.pos.astype(np.float64),
        } # this is an issue of matlab: python list are converted into row vectors but atlas should be column vectors
        eng.workspace['lf_atlas'] = self.lf.atlas
        eng.eval("lf_mat.atlas = lf_atlas'", nargout=0) 
                
        pos_arr = np.atleast_1d(pos).astype(np.float64)
        eng.workspace['pos'] = pos_arr
        
        if isinstance(region, str):
            region = [region]
        eng.workspace['region_cell'] = region
        
        eng.eval("[sourceIdx, dist] = lf_get_source_nearest(lf_mat, pos, 'region', region_cell);", nargout=0)
        
        source_idx_mat = eng.workspace['sourceIdx']
        dist_mat = eng.workspace['dist']
        
        source_idx = int(source_idx_mat) - 1  # Convert to 0-indexed
        dist = float(dist_mat)
        
        return source_idx, dist
    
    def test_nearest_to_origin(self):
        """Test finding source nearest to origin."""
        pos = [0, 0, 0]
        
        idx_py, dist_py = self.lf.get_source_nearest(pos)
        idx_mat, dist_mat = self._get_matlab_nearest(pos)
        
        self.assertEqual(idx_py, idx_mat, "Source indices differ")
        np.testing.assert_allclose(dist_py, dist_mat, rtol=1e-10,err_msg="Distances differ")
    
    def test_nearest_to_arbitrary_position(self):
        """Test with arbitrary position."""
        pos = [25, -30, 45]
        
        idx_py, dist_py = self.lf.get_source_nearest(pos)
        idx_mat, dist_mat = self._get_matlab_nearest(pos)
        
        self.assertEqual(idx_py, idx_mat)
        np.testing.assert_allclose(dist_py, dist_mat, rtol=1e-10)
    
    def test_nearest_to_exact_source_position(self):
        """Test with exact position of a source (distance should be 0)."""
        source_idx = 500
        pos = self.lf.pos[source_idx]
        
        idx_py, dist_py = self.lf.get_source_nearest(pos)
        idx_mat, dist_mat = self._get_matlab_nearest(pos)
        
        # Should find the exact source
        self.assertEqual(idx_py, source_idx)
        self.assertEqual(idx_mat, source_idx)
        np.testing.assert_allclose(dist_py, 0, atol=1e-10)
        np.testing.assert_allclose(dist_mat, 0, atol=1e-10)
    
    def test_with_region_filter(self):
        """Test with region filtering."""
        pos = [0, 0, 0]
        region = 'Central.*'
        
        idx_py, dist_py = self.lf.get_source_nearest(pos, region=region)
        idx_mat, dist_mat = self._get_matlab_nearest(pos, region=region)
        
        self.assertEqual(idx_py, idx_mat)
        np.testing.assert_allclose(dist_py, dist_mat, rtol=1e-10)
    
    def test_extreme_positions(self):
        """Test with extreme positions (far from brain)."""
        pos = [1000, 1000, 1000]
        
        idx_py, dist_py = self.lf.get_source_nearest(pos)
        idx_mat, dist_mat = self._get_matlab_nearest(pos)
        
        self.assertEqual(idx_py, idx_mat)
        np.testing.assert_allclose(dist_py, dist_mat, rtol=1e-9)  # Slightly relaxed tolerance for large numbers


class TestNearestProperties(unittest.TestCase):
    """Test mathematical and logical properties of get_source_nearest."""
    
    @classmethod
    def setUpClass(cls):
        """Load leadfield once for all tests."""
        cls.lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'])
    
    def test_returns_valid_index_and_distance(self):
        """Test that method returns valid index and positive distance."""
        idx, dist = self.lf.get_source_nearest([0, 0, 0])
        
        self.assertIsInstance(idx, int)
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, self.lf.n_sources)
        self.assertGreaterEqual(dist, 0)
    
    def test_distance_matches_actual_distance(self):
        """Test that returned distance matches actual Euclidean distance."""
        pos = np.array([15, -20, 25])
        
        idx, dist = self.lf.get_source_nearest(pos)
        
        # actual distance
        actual_dist = np.linalg.norm(self.lf.pos[idx] - pos)
        
        np.testing.assert_allclose(dist, actual_dist, rtol=1e-10)
    
    def test_no_other_source_is_closer(self):
        """Test that no other source is closer than the returned one."""
        pos = np.array([10, 10, 10])
        
        idx, dist = self.lf.get_source_nearest(pos)
        
        # Check all sources
        for i in range(self.lf.n_sources):
            other_dist = np.linalg.norm(self.lf.pos[i] - pos)
            self.assertGreaterEqual(
                other_dist, dist - 1e-10,  # Allow small numerical error
                f"Source {i} is closer ({other_dist}) than returned source {idx} ({dist})")
    
    def test_exact_position_gives_zero_distance(self):
        """Test that querying exact source position gives distance 0."""
        source_idx = 1234
        pos = self.lf.pos[source_idx]
        
        idx, dist = self.lf.get_source_nearest(pos)
        
        self.assertEqual(idx, source_idx)
        np.testing.assert_allclose(dist, 0, atol=1e-10)
    
    def test_deterministic_result(self):
        """Test that same position always gives same result."""
        pos = [5, -5, 10]
        
        idx1, dist1 = self.lf.get_source_nearest(pos)
        idx2, dist2 = self.lf.get_source_nearest(pos)
        
        self.assertEqual(idx1, idx2)
        self.assertEqual(dist1, dist2)
    
    def test_region_constrains_search(self):
        """Test that region filter actually constrains the search."""
        pos = [0, 0, 0]
        
        idx_all, _ = self.lf.get_source_nearest(pos, region='.*')
        idx_brain, _ = self.lf.get_source_nearest(pos, region='Brain.*')
        self.assertTrue(self.lf.atlas[idx_brain].startswith('Brain'))
    
    def test_invalid_position_raises_error(self):
        """Test that invalid position raises error."""
        with self.assertRaises(ValueError):
            self.lf.get_source_nearest([1, 2])  # Only 2 coordinates
        
        with self.assertRaises(ValueError):
            self.lf.get_source_nearest([1, 2, 3, 4])  # Too many coordinates
    
    def test_empty_region_raises_error(self):
        """Test that empty region raises error."""
        with self.assertRaises(ValueError):
            self.lf.get_source_nearest([0, 0, 0], region='NonExistent_XYZ')


if __name__ == '__main__':
    unittest.main(verbosity=2)