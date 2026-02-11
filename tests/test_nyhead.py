import unittest
from leadfield import LeadField, lf_generate_from_nyhead

class TestNyhead(unittest.TestCase):
    
    def test_montage_loading(self):
        """Test leadfield generation using a standard montage."""
        lf = lf_generate_from_nyhead(montage='S64', verbose=True)
        
        self.assertIsInstance(lf, LeadField)
        self.assertEqual(lf.shape, (64, lf.n_sources, 3))
    
    def test_generate_with_custom_labels(self):
        """Test leadfield generation with custom electrode labels."""
        labels = ['Fz', 'Cz', 'Pz', 'Oz']
        lf = lf_generate_from_nyhead(montage='S64', labels=labels, verbose=True)
        
        assert lf.n_channels == len(labels)
        assert lf.leadfield.shape[0] == len(labels)
        assert set(lf.channel_labels) == set(labels)
    
    def test_generate_with_small_subset(self):
        """Test with a minimal electrode set."""
        labels = ['Cz']
        lf = lf_generate_from_nyhead(montage='S64', labels=labels, verbose=True)
        
        assert lf.n_channels == 1
        assert lf.channel_labels[0] == 'Cz'
        assert lf.shape == (1, lf.n_sources, 3)


if __name__ == '__main__':
    unittest.main()
        