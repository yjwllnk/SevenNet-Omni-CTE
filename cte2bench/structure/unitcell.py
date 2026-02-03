import numpy as np
import ase.io as ase_IO
from copy import deepcopy
import gc, os
import torch
from tqdm import tqdm
from cte2bench.util.relax import get_relaxer
from cte2bench.util.utils import get_spgnum, log_stats
import sys
from cte2bench.util.io import dumpPKL

def enumerate_atoms(unitcell_dict, config):
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
    output_atoms = []
    for idx, _dct in unitcell_dict.items():
        atoms = ase_IO.read(f'{base_dir}/{_dct["suffix"]}/{config["unitcell"]["save"]}/CONTCAR')
        atoms.info.update(_dct.copy())
        output_atoms.append(atoms)
    try:
        ase_IO.write(f'{base_dir}/{calc_tag}-unitcell_relax.extxyz', output_atoms)
    except Exception as exec:
        print(f'Exception {exec} Occured While Saving Unitcell data')

def process_unitcell(config, calc):
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
    desc = 'Unit cell optimization'
    print(f'INFO: relaxing unit cell with MLIP {calc_tag}')
    unitcell_dict = {}

    input_atoms = ase_IO.read(config['directory']['input'], **config['directory']['load_args'])
    
    for idx, atoms0 in enumerate(tqdm(input_atoms, desc=desc)):
        atoms0.info['ID'] = f'ID-{idx}'
        _dct = atoms0.info
        suffix = f"ID-{idx}_{_dct['material_id']}_{_dct['name']}_{_dct['symm.no']}"

        unitcell_dict[idx] =  {}
        unitcell_dict[idx].update(atoms0.info)
        cwd = os.path.join(base_dir, suffix, config["unitcell"]["save"])
        os.makedirs(cwd, exist_ok = True)
        logfile = f'{cwd}/{calc_tag}-unitcell0.log' 
        
        relaxer = get_relaxer(config, calc, opt_type='unitcell', logfile=logfile)
        atoms = atoms0.copy()

        if _dct.get('symm.no', get_spgnum(atoms)) == 186:
            atoms.info['primitive_matrix'] = np.eye(3)

        atoms = relaxer.update_atoms(atoms)
        atoms.info['suffix'] = suffix
        atoms.info['calc_tag'] = calc_tag
        log_stats(config, atoms, task='unit')

        atoms = relaxer.relax_atoms(atoms)
        atoms = relaxer.update_atoms(atoms)
        log_stats(config, atoms, task='unit', stat='relax')

        steps = atoms.info['steps']
        init_sgn = atoms.info['symm.no']
        atoms.info['symm.no.unit'] = unit_sgn = get_spgnum(atoms)
        force_conv = atoms.info['force_conv']
        ase_IO.write(f'{cwd}/CONTCAR', atoms, format='vasp')

        if steps >= config['opt']['unitcell']['steps'] or not force_conv:
            atoms.info['unitcell.opt'] = False
            print(f'WARNING: {suffix} unit cell relaxation did not reach convergence in {steps}')
        else:
            atoms.info['unitcell.opt'] = True

        if init_sgn != unit_sgn:
            atoms.info['unitcell.symm'] = False
            print(f'WARNING: symmetry of {suffix} changed from {init_sgn} to {unit_sgn}')
        else:
            atoms.info['unitcell.symm'] = True

        unitcell_dict[idx].update(atoms.info)
        atoms.calc = None
        del relaxer
        gc.collect()
 
    dumpPKL(unitcell_dict, filename=f'{base_dir}/{calc_tag}-unitcell.pkl')

    enumerate_atoms(unitcell_dict, config)
    del input_atoms
    gc.collect()
    torch.cuda.empty_cache()
