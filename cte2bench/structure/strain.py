from tqdm import tqdm
import ase.io as ase_IO
import torch, gc, os
import numpy as np

from cte2bench.util.utils import get_spgnum, log_stats
from cte2bench.util.relax import get_relaxer
from cte2bench.util.io import dumpPKL

def enumerate_strained(strain_dct, suffix, config):
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
    output_atoms = []
    output_dir = f'{base_dir}/{suffix}/{config["strain"]["save"]}'
    for _, _dct in strain_dct.items():
        atoms = ase_IO.read(f'{output_dir}/CONTCAR_e{_dct["eps"]}')
        atoms.info.update(_dct.copy())
        output_atoms.append(atoms)
    ase_IO.write(f'{output_dir}/{calc_tag}-strain_relax-{suffix}.extxyz', output_atoms, format='extxyz')
    dumpPKL(strain_dct, f'{output_dir}/{calc_tag}-strain_dct-{suffix}.pkl')

def scale_unitcell(suffix, config):
    base_dir = config['directory']['cwd']
    calc_tag = config['calculator']['tag']
    cwd = os.path.join(base_dir, suffix, config['strain']['save'])
    poscar_opt = open(f"{base_dir}/{suffix}/{config['unitcell']['save']}/CONTCAR", 'r')
    lines = poscar_opt.readlines()
    desc = 'Scaling'
    for i, eps in enumerate(tqdm(config['strain']['eps'], desc=desc)):
        poscar_file = open(f"{cwd}/POSCAR_e{eps}", 'w')
        strained_lines = lines.copy()
        strained_lines[0] = f'{suffix}-{calc_tag}\n'
        strained_lines[1] = f'{1+eps}\n'
        poscar_file.writelines(strained_lines)
        poscar_file.close()
    scaled_list = [ase_IO.read(f'{cwd}/POSCAR_e{eps}',format='vasp') for eps in config['strain']['eps']]
    poscar_opt.close()
    ase_IO.write(f'{cwd}/{calc_tag}-strain-{suffix}.extxyz', scaled_list, format='extxyz', append=True)
    return

def process_strain(config, calc):
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
    input_atoms = ase_IO.read(f'{base_dir}/{calc_tag}-unitcell_relax.extxyz', index=':')

    print(f'INFO: pre-processing with MLIP {calc_tag}')
    desc = 'v-ZSISA relaxation'
    for idx, atoms0 in enumerate(tqdm(input_atoms, desc=desc)):
        _dct = atoms0.info.copy()
        suffix = _dct['suffix']
        cwd = os.path.join(base_dir, suffix, config['strain']['save'])
        os.makedirs(cwd, exist_ok = True)
        scale_unitcell(suffix, config)

        strained_input = ase_IO.read(f'{cwd}/{calc_tag}-strain-{suffix}.extxyz', index=':')

        det = np.linalg.norm(np.linalg.det(_dct['primitive_matrix']))

        strain_dct = {}

        for i, (strained, eps) in enumerate(zip(strained_input, config['strain']['eps'])):
            strain_dct[f'e{eps}'] = {} 
            strained.info = _dct
            strained.info['eps'] = eps
            logfile = f'{cwd}/strain_e{eps}.log'
            relaxer = get_relaxer(config, calc, opt_type='strain', logfile=logfile)
            strained = relaxer.update_atoms(strained)
            log_stats(config, strained, task='strain', stat='oneshot', eps=eps)

            init_vol = round(strained.get_volume()/len(strained), 4)

            strained = relaxer.relax_atoms(strained)
            strained = relaxer.update_atoms(strained)
            ase_IO.write(f'{cwd}/CONTCAR_e{eps}', strained, format='vasp')
            strained.info["symm.no.strain"] = strain_sgn = get_spgnum(strained)
            log_stats(config, strained, task='strain', stat='relax', eps=eps)
            steps, force_conv = strained.info['steps'], strained.info['force_conv']
            strain_vol = round(strained.get_volume()/len(strained), 4)

            if steps >= config['opt']['strain']['steps'] or not force_conv:
                strained.info['strain.opt'] = False
                print(f'WARNING: {suffix} e{eps} relaxation did not reach convergence in {steps}')
            else:
                strained.info['strain.opt'] = True

            if (unit_sgn := strained.info['symm.no.unit']) != strain_sgn:
                strained.info['strain.symm'] = False
                print(f'WARNING: symmetry of {suffix} changed from {unit_sgn} to {strain_sgn}')
            else:
                strained.info['strain.symm'] = True

            if init_vol != strain_vol:
                strained.info['strain.vol'] = False
                print(f'WARNING: volume of {suffix} changed from {init_vol} to {strain_vol}')
            else:
                strained.info['strain.vol'] = True

            strain_dct[f'e{eps}'].update(strained.info)
            del relaxer
            gc.collect()
        enumerate_strained(strain_dct, suffix, config)
        del strained_input, strain_dct
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
