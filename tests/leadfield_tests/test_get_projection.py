"""test_projection_matlab_validation.py"""
import unittest
import numpy as np

from leadfield.nyhead import lf_generate_from_nyhead
from utils import with_matlab_engine, EngineWrapper


class TestProjectionVsMATLAB(unittest.TestCase):
    """Validate Python projection against MATLAB implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Load leadfield once for all tests."""
        cls.eng = EngineWrapper()
        cls.lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'], eng=cls.eng)
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup MATLAB engine."""
        cls.eng.quit()
    
    def _get_matlab_projection(self, source_idx, orientation=None, 
                           normalize_lf=False, normalize_ori=False):
        """Get projection from MATLAB for comparison."""
        eng = self.eng
        
        # Transfer leadfield to MATLAB - ENSURE FLOAT TYPE
        eng.workspace['lf_mat'] = {
            'leadfield': self.lf.leadfield.astype(np.float64),  # Force float64
            'orientation': self.lf.orientation.astype(np.float64),  # Force float64
            'pos': self.lf.pos.astype(np.float64)  # Force float64
        }
        
        # Build MATLAB call
        if isinstance(source_idx, (list, np.ndarray)):
            idx_str = f"[{' '.join(map(str, np.array(source_idx) + 1))}]"  # MATLAB 1-indexed
        else:
            idx_str = str(source_idx + 1)
        
        args = []
        if orientation is not None:
            ori_mat = np.atleast_2d(orientation).astype(np.float64)  # Force float64
            eng.workspace['ori'] = ori_mat
            args.append("'orientation', ori")
        if normalize_lf:
            args.append("'normaliseLeadfield', 1")
        if normalize_ori:
            args.append("'normaliseOrientation', 1")
        
        args_str = ', '.join(args)
        if args_str:
            args_str = ', ' + args_str
        
        # Call MATLAB function
        proj_matlab = eng.eval(f"lf_get_projection(lf_mat, {idx_str}{args_str})")
        return np.array(proj_matlab).flatten()
    
    def test_single_source_default_orientation(self):
        """Test single source with default orientation."""
        source_idx = 100
        
        # Python
        proj_python = self.lf.get_projection(source_idx)
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(source_idx)
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Python and MATLAB projections differ for single source"
        )
    
    def test_single_source_custom_orientation(self):
        """Test single source with custom orientation."""
        source_idx = 100
        orientation = np.array([0, 1, 0])  # Anterior-posterior
        
        # Python
        proj_python = self.lf.get_projection(source_idx, orientation=orientation)
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(source_idx, orientation=orientation)
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ with custom orientation"
        )
    
    def test_multiple_sources_default_orientation(self):
        """Test multiple sources with default orientations."""
        source_indices = [100, 200, 300]
        
        # Python
        proj_python = self.lf.get_projection(source_indices)
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(source_indices)
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ for multiple sources"
        )
    
    def test_multiple_sources_single_orientation(self):
        """Test multiple sources with single orientation applied to all."""
        source_indices = [100, 200, 300]
        orientation = np.array([1, 0, 0])  # Left-right
        
        # Python (should show warning)
        proj_python = self.lf.get_projection(source_indices, orientation=orientation)
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(source_indices, orientation=orientation)
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ with single orientation for multiple sources"
        )
    
    def test_multiple_sources_multiple_orientations(self):
        """Test multiple sources with individual orientations."""
        source_indices = [100, 200, 300]
        orientations = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # Python
        proj_python = self.lf.get_projection(source_indices, orientation=orientations)
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(source_indices, orientation=orientations)
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ with multiple orientations"
        )
    
    def test_normalization_leadfield(self):
        """Test with leadfield normalization."""
        source_idx = 100
        orientation = np.array([1, 1, 1])
        
        # Python
        proj_python = self.lf.get_projection(
            source_idx, 
            orientation=orientation,
            normalize_leadfield=True
        )
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(
            source_idx, 
            orientation=orientation,
            normalize_lf=True
        )
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ with leadfield normalization"
        )
    
    def test_normalization_orientation(self):
        """Test with orientation normalization."""
        source_idx = 100
        orientation = np.array([2, 3, 4])  # Non-unit vector
        
        # Python
        proj_python = self.lf.get_projection(
            source_idx, 
            orientation=orientation,
            normalize_orientation=True
        )
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(
            source_idx, 
            orientation=orientation,
            normalize_ori=True
        )
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ with orientation normalization"
        )
    
    def test_both_normalizations(self):
        """Test with both normalizations enabled."""
        source_idx = 100
        orientation = np.array([2, 3, 4])
        
        # Python
        proj_python = self.lf.get_projection(
            source_idx, 
            orientation=orientation,
            normalize_leadfield=True,
            normalize_orientation=True
        )
        
        # MATLAB
        proj_matlab = self._get_matlab_projection(
            source_idx, 
            orientation=orientation,
            normalize_lf=True,
            normalize_ori=True
        )
        
        # Compare
        np.testing.assert_allclose(
            proj_python, proj_matlab, 
            rtol=1e-10, atol=1e-12,
            err_msg="Projections differ with both normalizations"
        )


class TestProjectionMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of projections."""
    
    @classmethod
    def setUpClass(cls):
        """Load leadfield once for all tests."""
        cls.lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'])
    
    def test_projection_shape(self):
        """Test that projection has correct shape."""
        proj = self.lf.get_projection(100)
        self.assertEqual(proj.shape, (self.lf.n_channels,))
    
    def test_projection_not_all_zeros(self):
        """Test that projection is not all zeros."""
        proj = self.lf.get_projection(100)
        self.assertTrue(np.any(proj != 0))
    
    def test_projection_linearity(self):
        """Test that projection is linear in orientation."""
        source_idx = 100
        ori1 = np.array([1, 0, 0])
        ori2 = np.array([0, 1, 0])
        
        # Individual projections
        proj1 = self.lf.get_projection(source_idx, orientation=ori1)
        proj2 = self.lf.get_projection(source_idx, orientation=ori2)
        
        # Combined orientation
        proj_sum = self.lf.get_projection(source_idx, orientation=ori1 + ori2)
        
        # Should satisfy linearity: proj(ori1 + ori2) = proj(ori1) + proj(ori2)
        np.testing.assert_allclose(
            proj_sum, proj1 + proj2, 
            rtol=1e-10,
            err_msg="Projection doesn't satisfy linearity"
        )
    
    def test_projection_scaling(self):
        """Test that projection scales linearly with orientation magnitude."""
        source_idx = 100
        ori = np.array([1, 1, 1])
        scale = 2.5
        
        proj1 = self.lf.get_projection(source_idx, orientation=ori)
        proj2 = self.lf.get_projection(source_idx, orientation=scale * ori)
        
        # proj(scale * ori) = scale * proj(ori)
        np.testing.assert_allclose(
            proj2, scale * proj1, 
            rtol=1e-10,
            err_msg="Projection doesn't scale correctly"
        )
    
    def test_zero_orientation_gives_zero_projection(self):
        """Test that zero orientation gives zero projection."""
        proj = self.lf.get_projection(100, orientation=np.array([0, 0, 0]))
        np.testing.assert_allclose(proj, 0, atol=1e-12)
    
    def test_orthogonal_orientations_are_independent(self):
        """Test that orthogonal orientations give independent projections."""
        source_idx = 100
        ori_x = np.array([1, 0, 0])
        ori_y = np.array([0, 1, 0])
        ori_z = np.array([0, 0, 1])
        
        proj_x = self.lf.get_projection(source_idx, orientation=ori_x)
        proj_y = self.lf.get_projection(source_idx, orientation=ori_y)
        proj_z = self.lf.get_projection(source_idx, orientation=ori_z)
        
        # The projections should be different (not parallel)
        # Check that no two are scalar multiples
        ratio_xy = proj_x / (proj_y + 1e-10)
        ratio_xz = proj_x / (proj_z + 1e-10)
        
        # If independent, ratios shouldn't be constant
        self.assertGreater(np.std(ratio_xy), 0.01)
        self.assertGreater(np.std(ratio_xz), 0.01)
    
    def test_mean_of_multiple_sources(self):
        """Test that multiple sources return mean projection."""
        sources = [100, 101, 102]
        
        # Get individual projections
        projs = [self.lf.get_projection(s) for s in sources]
        expected_mean = np.mean(projs, axis=0)
        
        # Get combined projection
        proj_combined = self.lf.get_projection(sources)
        
        np.testing.assert_allclose(
            proj_combined, expected_mean,
            rtol=1e-10,
            err_msg="Mean projection incorrect"
        )
    
    def test_normalization_effect(self):
        """Test that normalization actually normalizes."""
        source_idx = 100
        ori = np.array([1, 2, 3])
        
        # Without normalization
        proj_unnorm = self.lf.get_projection(source_idx, orientation=ori)
        
        # With leadfield normalization
        proj_norm_lf = self.lf.get_projection(
            source_idx, orientation=ori, normalize_leadfield=True
        )
        
        # With orientation normalization
        proj_norm_ori = self.lf.get_projection(
            source_idx, orientation=ori, normalize_orientation=True
        )
        
        # Normalized projections should have different magnitudes
        self.assertNotAlmostEqual(
            np.linalg.norm(proj_unnorm),
            np.linalg.norm(proj_norm_lf)
        )
        self.assertNotAlmostEqual(
            np.linalg.norm(proj_unnorm),
            np.linalg.norm(proj_norm_ori)
        )
    
    def test_reproducibility(self):
        """Test that same inputs give same outputs."""
        source_idx = 100
        ori = np.array([1, 1, 1])
        
        proj1 = self.lf.get_projection(source_idx, orientation=ori)
        proj2 = self.lf.get_projection(source_idx, orientation=ori)
        
        np.testing.assert_array_equal(proj1, proj2)


if __name__ == '__main__':
    unittest.main(verbosity=2)