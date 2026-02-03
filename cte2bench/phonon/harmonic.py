from __future__ import annotations

from phonopy import Phonopy
import phonopy.file_IO as ph_IO

import os, gc, warnings, json
from tqdm import tqdm
import ase.io as ase_IO

from cte2bench.util.io import loadPKL, dumpJSON, clean_for_json
from cte2bench.util.utils import load_mesh_yaml, load_mesh_hdf5, imag_dos_frac, aseatoms2phonoatoms, check_imaginary_freqs

def process_harmonic(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="seekpath.hpkot")
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
 
    desc = 'Mesh properties'
    mesh_args = {'is_time_reversal': True, 'is_mesh_symmetry': True,
                'is_gamma_center': False, 'with_eigenvectors': True,
                'with_group_velocities': True}
    # with_eigen_vectors; disable mesh_sym
    # for PES calculation, enable with_eigenvectors

    thermal_kwargs = {'t_min': config['harmonic']['t_min'],
                      't_max': config['harmonic']['t_max'],
                      't_step': config['harmonic']['t_step']}
   
    distance = config['supercell']['distance']
    unit_dct_file = f'{base_dir}/{calc_tag}-unitcell.pkl'
    unit_dct = loadPKL(unit_dct_file)

    RESULTS = {}
    RESULTS['calc'] = calc_tag

    for idx, _dct in tqdm(unit_dct.items(), desc=desc):
        idx_dct = {}
        suffix = _dct['suffix']
        ID, mp, name, symm = suffix.split('_')
        idx_dct['ID'] = ID
        idx_dct['name'] = name
        idx_dct['symm.no'] = symm
        idx_dct['mp-id'] = mp

        mesh_numbers = _dct.get('q_point_mesh', [19, 19, 19])

        strain_dir = f'{base_dir}/{suffix}/{config["strain"]["save"]}'
        strain_dct_file = f'{strain_dir}/{calc_tag}-strain_dct-{suffix}.pkl'
        strain_dct = loadPKL(strain_dct_file)
        strain_opt = ase_IO.read(f'{strain_dir}/{calc_tag}-strain_relax-{suffix}.extxyz', index=':')

        phonon_kwargs = {'primitive_matrix': _dct.get('primitive_matrix', 'auto'),
                'supercell_matrix': _dct['fc2_supercell'], 'symprec': 1e-05}

        strain_dir = os.path.join(base_dir, suffix, config['strain']['save'])
        supercell_dir = os.path.join(base_dir, suffix, config['supercell']['save'])
        cwd = os.path.join(base_dir, suffix, config['harmonic']['save'])
        os.makedirs(cwd, exist_ok = True)

        idx_dct['harmonic'] = {}
        for i, eps in enumerate(config['strain']['eps']):
            idx_dct['harmonic'][f'e{eps}'] = {}
            eps_dir = f'{cwd}/e{eps}'
            os.makedirs(eps_dir, exist_ok=True)
            h_dct = {'fc2': True, 'IMAGINARY': False, 'fraction': 0.0, 'QHA': True}

            if config['harmonic']['cont']:
                if os.path.isfile(f'{eps_dir}/mesh_e{eps}.hdf5'):
                    freqs, weights = load_mesh_hdf5(f'{eps_dir}/mesh_e{eps}.hdf5')
                    Im = check_imaginary_freqs(freqs)
                    Fraction = imag_dos_frac(freqs, weights)
                    QHA = (Fraction < 0.220)
                
                    h_dct['IMAGINARY'] = Im
                    h_dct['QHA'] = QHA
                    h_dct['fraction'] = Fraction
                    idx_dct['harmonic'][f'e{eps}'].update(h_dct)
                    continue
            
            strained = strain_opt[i]
            unitcell = aseatoms2phonoatoms(strained)
            phonon = Phonopy(unitcell=unitcell, **phonon_kwargs)
            phonon.generate_displacements(distance=config['supercell']['distance'], is_plusminus=True,
                    random_seed=config['supercell']['random_seed'])
            fc2 = ph_IO.parse_FORCE_CONSTANTS(f'{supercell_dir}/FORCE_CONSTANTS_2ND_e{eps}')
            phonon.force_constants = fc2

            phonon.run_mesh(mesh_numbers, **mesh_args)
            phonon.mesh.write_hdf5(filename=f'{eps_dir}/mesh_e{eps}.hdf5')
            freqs = phonon.get_mesh_dict()['frequencies']
            weights = phonon.get_mesh_dict()['weights']

            Im = check_imaginary_freqs(freqs)
            Fraction = imag_dos_frac(freqs,weights)
            QHA = (Fraction < 0.220)

            h_dct['IMAGINARY'] = Im
            h_dct['QHA'] = QHA
            h_dct['fraction'] = Fraction
            idx_dct['harmonic'][f'e{eps}'].update(h_dct)

            phonon.save(f'{eps_dir}/phonopy_e{eps}.yaml', compression=True)

            if config['harmonic']['run_thermal']:
                if not os.path.isfile(f'{eps_dir}/thermal_properties_e{eps}.svg'):
                    phonon.run_thermal_properties(**thermal_kwargs)
                    phonon.write_yaml_thermal_properties(f'{eps_dir}/thermal_properties_e{eps}.yaml')
                    thermal_plt = phonon.plot_thermal_properties()
                    thermal_plt.savefig(f'{eps_dir}/thermal_properties_e{eps}.svg')
                    thermal_plt.close()

            if config['harmonic']['run_band']:
                if not os.path.isfile(f'{eps_dir}/band_structure_e{eps}.svg'):
                    phonon.auto_band_structure(write_yaml=True, filename=f'{eps_dir}/band_e{eps}.yaml')
                    band_plt = phonon.plot_band_structure()
                    band_plt.savefig(f'{eps_dir}/band_structure_e{eps}.svg')
                    band_plt.close()

            if config['harmonic']['run_dos']:
                if not os.path.isfile(f'{eps_dir}/band_dos_e{eps}.svg'):
                    phonon.auto_total_dos(write_dat=True, filename=f'{eps_dir}/total_dos_e{eps}.dat', mesh=mesh_numbers)
                    band_dos_plt = phonon.plot_band_structure_and_dos()
                    band_dos_plt.savefig(f'{eps_dir}/band_dos_e{eps}.svg')
                    band_dos_plt.close()

            del phonon, unitcell, strained, freqs, weights
            gc.collect()

        RESULTS[idx] = idx_dct
        del strain_opt, strain_dct
        gc.collect()
    RESULTS = clean_for_json(RESULTS)
    dumpJSON(RESULTS, f'{base_dir}/{calc_tag}_results.json')
