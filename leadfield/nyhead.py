from typing import Optional, Union, List
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .leadfield import LeadField, ChannelLocation
from utils import with_matlab_engine, EngineWrapper

NYHEAD_WORKING_DIR = Path(__file__).resolve().parents[1] / 'SEREEGA' / 'leadfield' / 'nyhead'
NYHEAD_LEADFIELD_PATH = Path(__file__).resolve().parents[1] / 'leadfield_mat' / 'sa_nyhead.mat'

def _convert_matlab_lf_to_leadfield(eng, var_name='lf') -> LeadField:
    """
    Convert MATLAB leadfield structure to Python LeadField object.
    Extracts fields directly from MATLAB workspace to avoid struct array issues.
    
    Parameters
    ----------
    eng : matlab.engine.MatlabEngine
        Running MATLAB engine instance
    var_name : str
        Name of the variable in MATLAB workspace
        
    Returns
    -------
    LeadField
        Python LeadField object
    """
    leadfield = np.array(eng.eval(f'{var_name}.leadfield'))
    orientation = np.array(eng.eval(f'{var_name}.orientation'))
    pos = np.array(eng.eval(f'{var_name}.pos'))
    
    n_channels = int(eng.eval(f'length({var_name}.chanlocs)'))
    
    atlas = None
    try:
        n_sources = int(eng.eval(f'length({var_name}.atlas)'))
        print(f"Loading atlases ({n_sources} in total)", end="...",flush=True)
        atlas = eng.eval(f'{var_name}.atlas'); print("done.")
        atlas = list(atlas)
    except:
        pass
    
    chanlocs = []
    for i in tqdm(range(n_channels), desc="Loading Channel Locations", ncols=100):
        idx = i + 1  # remember taht matlab indexing starts at 1
        
        labels = eng.eval(f'{var_name}.chanlocs({idx}).labels')
        X = float(eng.eval(f'{var_name}.chanlocs({idx}).X'))
        Y = float(eng.eval(f'{var_name}.chanlocs({idx}).Y'))
        Z = float(eng.eval(f'{var_name}.chanlocs({idx}).Z'))
        theta = float(eng.eval(f'{var_name}.chanlocs({idx}).theta'))
        radius = float(eng.eval(f'{var_name}.chanlocs({idx}).radius'))
        
        try:
            sph_theta = float(eng.eval(f'{var_name}.chanlocs({idx}).sph_theta'))
        except:
            sph_theta = 0.0
            
        try:
            sph_phi = float(eng.eval(f'{var_name}.chanlocs({idx}).sph_phi'))
        except:
            sph_phi = 0.0
            
        try:
            sph_radius = float(eng.eval(f'{var_name}.chanlocs({idx}).sph_radius'))
        except:
            sph_radius = 0.0
            
        try:
            chan_type = eng.eval(f'{var_name}.chanlocs({idx}).type')
        except:
            chan_type = 'EEG'
        
        chanlocs.append(ChannelLocation(
            labels=labels,
            X=X, Y=Y, Z=Z,
            theta=theta, radius=radius,
            sph_theta=sph_theta,
            sph_phi=sph_phi,
            sph_radius=sph_radius,
            type=chan_type,
            urchan=None,
            ref=None
        ))
    
    return LeadField(
        leadfield=leadfield,
        pos=pos,
        orientation=orientation,
        chanlocs=chanlocs,
        atlas=atlas,
        method='nyhead',
        source='New York Head (ICBM-NY)',
        unit='µV/(nA·m)',
        metadata={
            'reference': 'Huang et al. (2016) NeuroImage 140:150-162',
            'copyright': 'NY Head © 2015 Yu Huang, Lucas C. Parra, Stefan Haufe (GNU GPL 3)'
        }
    )

@with_matlab_engine
def lf_generate_from_nyhead(
    montage: str,
    labels: Optional[List[str]] = None,
    nyhead_path: Union[str, Path] = NYHEAD_LEADFIELD_PATH,
    normalize: bool = False,
    verbose : bool = False,
    eng : EngineWrapper = None
) -> LeadField:
    """
    Generate a leadfield from the New York Head model (Huang et al., 2016).

    This function generates a leadfield using a predefined EEG montage or custom channel labels
    from the NY Head model. It calls the MATLAB implementation of the NY Head model and
    converts the result to a Python LeadField object.

    Parameters
    ----------
    montage : str
        Name of the predefined EEG montage (e.g., 'S64', 'S128'). Only used if labels is None.
    labels : Optional[List[str]], default=None
        List of custom channel labels. If provided, overrides the montage parameter.
    nyhead_path : Union[str, Path], default=NYHEAD_LEADFIELD_PATH
        Path to the NY Head leadfield .mat file.
    normalize : bool, default=False
        If True, normalize the leadfield using the 'norm' method.
    verbose : bool, default=False
        If True, print progress messages during execution.

    Returns
    -------
    LeadField
        A LeadField object containing the leadfield matrix, channel locations, source positions,
        and orientation information from the NY Head model.

    Raises
    ------
    ValueError
        If nyhead_path is not a valid .mat file.
    """
    # Validate the leadfield path exists and is a valid MATLAB file
    # TODO: allow to add nyhead_path manually avoiding explicity adding the .mat file in MATLAB path
    nyhead_path = Path(nyhead_path)
    if not nyhead_path.is_file() or nyhead_path.suffix != '.mat':
        raise ValueError(f"Invalid nyhead_path: {nyhead_path}. Must be a .mat file.")
    
    #eng = start_matlab(working_dir=NYHEAD_WORKING_DIR)

    try:
        if labels is not None:
            eng.workspace['labels_list'] = list(labels)
            eng.eval("lf = lf_generate_fromnyhead('labels', labels_list);", nargout=0)
        else:
            eng.eval(f"lf = lf_generate_fromnyhead('montage', '{montage}');", nargout=0)

        if verbose:
            print("MATLAB function executed, extracting data...")
            
        lf = _convert_matlab_lf_to_leadfield(eng, var_name='lf')
        
        if normalize:
            lf = lf.normalize(method='norm')
        return lf
    except Exception as e:
        raise RuntimeError(f"Error generating leadfield from NY Head: {e}")
        
if __name__ == '__main__':
    lf = lf_generate_from_nyhead(montage='S64', verbose=True)
    #lf = lf_generate_from_nyhead(montage='S64', labels=['Fz', 'Cz', 'Pz', 'Oz'], verbose=True)
    
    print("leadfield loaded succesfully.")
    print(lf.leadfield.shape)