from phono3py import Phono3py
import torch, gc
import numpy as np
from tqdm import tqdm
import ase.io as ase_IO
from ase import Atoms
import os, sys

from phono3py import file_IO as ph3_IO
from phonopy import file_IO as ph_IO

from cte2bench.util.calc import single_point_calculate_list
from cte2bench.util.utils import aseatoms2phonoatoms, phonoatoms2aseatoms, log_stats
from cte2bench.util.io import dumpPKL, loadPKL


def calculate_fc2(config, cwd, eps, ph3, calc, symmetrize_fc2=True):
    desc = 'FC2 calculation'
    forces = []
    nat = len(ph3.phonon_supercell)
    indices = []
    atoms_list = []
    for i, sc in enumerate(ph3.phonon_supercells_with_displacements):
        label = str(i+1).zfill(5)
        if sc is not None:
            atoms_list.append(Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True))
            # ase_IO.write(f"{cwd}/e{eps}/POSCAR_e{eps}_FC2-{label}", ase_sc, format='vasp')
            # atoms_list.append(ase_sc)
            indices.append(i)

    result = single_point_calculate_list(atoms_list, calc, desc=desc)
    try:
        ase_IO.write(f'{cwd}/e{eps}_FC2.extxyz', result, format='extxyz')
    except Exception as exec:
        print(f'Error {exec} occured while saving result atoms list of single point calc.')

    for j, sc in enumerate(ph3.phonon_supercells_with_displacements):
        label = str(j+1).zfill(5)
        if sc is not None:
            atoms = result[indices.index(j)]
            f = atoms.get_forces()
            log_stats(config, atoms, task='fc2', eps=eps, disp=label)
        else:
            f = np.zeros((nat, 3))
        np.save(f'{cwd}/e{eps}/force-{label}.npy', f)
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.phonon_forces = force_set
    ph3.produce_fc2(symmetrize_fc2=symmetrize_fc2)

    return ph3

def process_supercell(config, calc):
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
 
    desc = 'Phonon supercells'

    unit_dct_file = f'{base_dir}/{calc_tag}-unitcell.pkl'
    unit_dct = loadPKL(unit_dct_file)

    for idx, _dct in tqdm(unit_dct.items(), desc=desc):
        suffix = _dct['suffix']
        primitive_matrix = _dct.get('primitive_matrix', 'auto')

        phonon_kwargs = {'primitive_matrix': primitive_matrix,
            'supercell_matrix': np.diag(_dct['fc3_supercell']),
            'phonon_supercell_matrix': np.diag(_dct['fc2_supercell'])}


        strain_dir = f'{base_dir}/{suffix}/{config["strain"]["save"]}'
        strain_dct_file = f'{strain_dir}/{calc_tag}-strain_dct-{suffix}.pkl'
        strain_dct = loadPKL(strain_dct_file)
        strain_opt = ase_IO.read(f'{strain_dir}/{calc_tag}-strain_relax-{suffix}.extxyz', index=':')

        for i, eps in enumerate(config['strain']['eps']):
            cwd = os.path.join(base_dir,suffix,config['supercell']['save'])
            os.makedirs(cwd, exist_ok=True)
            s_dct = strain_dct.get(f'e{eps}',None)

            if not s_dct:
                print(f'WARNING: Skipping {suffix} - e{eps} .. no meta data available')
                continue
            if config['supercell']['cont']: 
                if os.path.isfile(f'{cwd}/FORCE_CONSTANTS_2ND_e{eps}'):
                    # try:
                        # ph_IO.parse_FORCE_CONSTANTS(f'{cwd}/FORCE_CONSTANTS_2ND_e{eps}')
                    continue
                    # except:
                        # print(f'INFO: Re-calculating FC2 of {suffix}-e{eps}')

            os.makedirs(f'{cwd}/e{eps}', exist_ok = True)
            unitcell = aseatoms2phonoatoms(strain_opt[i])

            phonon = Phono3py(unitcell=unitcell, **phonon_kwargs)
            phonon.generate_fc2_displacements(distance=config['supercell']['distance'], is_plusminus=True,
                random_seed=config['supercell']['random_seed'])

            try:
                phonon = calculate_fc2(config, cwd, eps, phonon, calc)
                ph_IO.write_FORCE_CONSTANTS(phonon.fc2, filename=f'{cwd}/FORCE_CONSTANTS_2ND_e{eps}')
            except Exception as exec:
                print(f'ERROR: Exception {exec} occurred while calculating FC2 of {suffix}-e{eps}')
            
            del phonon
            gc.collect()
        torch.cuda.empty_cache()
        del strain_opt, strain_dct
        gc.collect()
