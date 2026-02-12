# pysereega

A Python port of [SEREEGA](https://github.com/lrkrol/SEREEGA) (Simulating Event-Related EEG Activity), a MATLAB toolbox for generating simulated EEG data with realistic forward models.

> **Status:** Early development (v0.1.0). Currently only the `leadfield` module is implemented. Other SEREEGA modules (sources, signals, epochs) are planned.

## Prerequisites

- **Python 3.10+**
- **MATLAB** (licensed installation) with the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
- **SEREEGA** MATLAB toolbox on your MATLAB path
- **EEGLAB** MATLAB toolbox on your MATLAB path (dependency of SEREEGA)
- **NY Head model** `.mat` file (`sa_nyhead.mat`) accessible to MATLAB

### Installing the MATLAB Engine for Python

```bash
pip install <MATLAB_ROOT>/extern/engines/python/
```

Where `<MATLAB_ROOT>` is your MATLAB installation directory (e.g., `/Applications/MATLAB_R2024b.app` on macOS).

### MATLAB path setup

Before using pysereega, ensure SEREEGA, EEGLAB, and the NY Head data directory are on your MATLAB path. You can do this in MATLAB:

```matlab
addpath(genpath('/path/to/SEREEGA'))
addpath(genpath('/path/to/EEGLAB'))
addpath('/path/to/nyhead-data-directory')
savepath
```

Or configure it at runtime in Python by passing a pre-configured `EngineWrapper`:

```python
from utils import EngineWrapper

eng = EngineWrapper()
eng.eval("addpath(genpath('/path/to/SEREEGA'))", nargout=0)
eng.eval("addpath(genpath('/path/to/EEGLAB'))", nargout=0)
eng.eval("addpath('/path/to/nyhead-data-directory')", nargout=0)
```

## Python dependencies

- `numpy`
- `tqdm`
- `scipy` (optional, for `.mat` file export)
- `pytest` (for running tests)

```bash
pip install numpy tqdm scipy pytest
```

## Usage

### Generating a lead field from the NY Head model

```python
from leadfield import lf_generate_from_nyhead

# Using a standard montage (e.g., 64-channel)
lf = lf_generate_from_nyhead(
    montage='S64',
    nyhead_path='/path/to/sa_nyhead.mat'
)

print(f"Channels: {lf.n_channels}, Sources: {lf.n_sources}")
# Channels: 64, Sources: 74382
```

### Using custom electrode labels

```python
lf = lf_generate_from_nyhead(
    montage='S64',
    labels=['Fz', 'Cz', 'Pz', 'Oz'],
    nyhead_path='/path/to/sa_nyhead.mat'
)
```

### Working with the LeadField object

```python
# Get scalp projection for a source with its default orientation
projection = lf.get_projection(source_idx=100)

# Get projection with a custom dipole orientation
import numpy as np
projection = lf.get_projection(100, orientation=np.array([0, 1, 0]))

# Normalize the lead field
lf_norm = lf.normalize(method='norm')

# Query anatomical regions
regions, counts, generic_counts = lf.get_regions()

# Find sources in a specific region (regex matching)
visual_sources = lf.get_sources_in_region(['visual'])

# Pick random sources from a region
random_sources = lf.get_source_random(number=5, region_patterns=['motor'])

# Save / load
lf.save('leadfield.npz')
from leadfield.leadfield import LeadField
lf_loaded = LeadField.load('leadfield.npz')
```

## Running tests

```bash
# Run all tests
pytest tests/ -v

# Run only the nyhead integration tests (requires MATLAB + SEREEGA + EEGLAB)
pytest tests/test_nyhead.py -v
```

**Note:** `tests/test_leadfield.py` currently imports symbols that are still in development and not yet pushed. Only `tests/test_nyhead.py` is runnable at this time.

## Project structure

```
pysereega/
├── leadfield/           # Forward model (lead field) module
│   ├── __init__.py
│   ├── leadfield.py     # LeadField and ChannelLocation dataclasses
│   ├── nyhead.py        # NY Head model loader (MATLAB bridge)
│   └── pha.py           # Pediatric Head Atlas loader (stub)
├── utils/               # Shared utilities
│   ├── __init__.py
│   └── engine_wrapper.py  # MATLAB engine management
├── tests/
│   ├── test_leadfield.py
│   └── test_nyhead.py
└── LICENSE              # GPL-3.0
```

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## References

- Krol, L. R., Pawlitzki, J., Lotte, F., Gramann, K., & Zander, T. O. (2018). SEREEGA: Simulating Event-Related EEG Activity. *Journal of Neuroscience Methods*, 309, 13-24.
- Huang, Y., Parra, L. C., & Haufe, S. (2016). The New York Head: Precise standardized volume conductor model for EEG source localization and tES targeting. *NeuroImage*, 140, 150-162.
