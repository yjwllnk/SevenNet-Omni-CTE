from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter, FrechetCellFilter
from ase.optimize import LBFGS, FIRE, FIRE2
import numpy as np
import time
from datetime import datetime
import torch

OPT_DCT = {'fire': FIRE, 'fire2':FIRE2,'lbfgs': LBFGS}
FILTER_DCT = {'frechet': FrechetCellFilter, 'unitcell': UnitCellFilter}

"""
modified based on Jaesun Kim's code
"""

class AseAtomRelax:
    def __init__(
        self,
        calc,
        optimizer=None,
        cell_filter=None,
        mask=None,
        fix_symm=True,
        const_vol = False,
        fmax=0.000001,
        steps=5000,
        logfile='ase_relaxer.log',
        time_dct={'oneshot': {
                        'start': {'wall': 0, 'date': 0},
                        'end': {'wall': 0, 'date': 0},
                        },
                  'relax': {
                         'start': {'wall': 0, 'date': 0},
                         'end': {'wall': 0, 'date': 0},
                         },
                  }
        ):
        self.calc = calc
        self.optimizer = optimizer
        self.cell_filter = cell_filter
        self.mask = mask
        self.fix_symm = fix_symm
        self.fmax = fmax
        self.steps = steps
        self.logfile = logfile
        self.constant_volume = const_vol
        self.time_dct = time_dct

    def update_atoms(self, atoms):
        start_wall = time.time()
        start_dt = datetime.now()
        atoms = atoms.copy()
        atoms.calc = self.calc
        try:
            atoms.info['e_fr_energy'] = np.float64(atoms.get_potential_energy(force_consistent=True))
        except:
            atoms.info['e_fr_energy'] = np.float64(atoms.get_potential_energy(force_consistent=False))
        atoms.info['e_0_energy'] = atoms.get_potential_energy()
        atoms.info['force'] = atoms.get_forces()
        atoms.info['stress'] = atoms.get_stress()
        end_wall = time.time()
        end_dt = datetime.now()
        force_conv = check_atoms_conv(atoms.get_forces())
        atoms.info['force_conv'] = force_conv

        oneshot_dct = {'start': {'wall': start_wall, 'date': start_dt.strftime('%Y-%m-%d %H:%M:%S')},
                       'end': {'wall': end_wall, 'date': end_dt.strftime('%Y-%m-%d %H:%M:%S')},
                       }
        self.time_dct['oneshot'].update(oneshot_dct)
        atoms.info['oneshot'] = oneshot_dct
        return atoms

    def relax_atoms(self, atoms):
        start_wall = time.time()
        start_dt = datetime.now()

        atoms = atoms.copy()
        if self.fix_symm:
            atoms.set_constraint(FixSymmetry(atoms, symprec=1e-05))

        atoms.calc = self.calc
        cell_filter = self.cell_filter(atoms, constant_volume = self.constant_volume, mask=self.mask)
        optimizer = self.optimizer(cell_filter, logfile=self.logfile)

        optimizer.run(fmax=self.fmax, steps=self.steps)
        torch.cuda.synchronize()
        atoms.info['steps'] = optimizer.get_number_of_steps()

        end_wall = time.time()
        end_dt = datetime.now()
        force_conv = check_atoms_conv(atoms.get_forces())
        atoms.info['force_conv'] = force_conv

        relax_dct = {'start': {'wall': start_wall, 'date': start_dt.strftime('%Y-%m-%d %H:%M:%S')},
                     'end': {'wall': end_wall, 'date': end_dt.strftime('%Y-%m-%d %H:%M:%S')},
                       }
        self.time_dct['relax'].update(relax_dct)
        atoms.info['relax'] = relax_dct
        atoms.info['time_dct'] = self.time_dct
        return atoms

    def redo(self, atoms):
        pass

def get_relaxer(config, calc, opt_type='unitcell', logfile='ase_relax.log'):
    arr_args = config['opt'][f'{opt_type}'].copy()

    opt = OPT_DCT[arr_args['optimizer'].lower()]
    cell_filter = FILTER_DCT[arr_args['cell_filter']]

    arr_args['calc'] = calc
    arr_args['logfile'] = logfile

    if arr_args.get('optimizer', None) is not None:
        arr_args['optimizer'] = opt
    if arr_args.get('cell_filter', None) is not None:
        arr_args['cell_filter'] = cell_filter

    return AseAtomRelax(**arr_args)

def check_atoms_conv(forces: np.ndarray) -> bool:
    conv = True
    for i in range(forces.shape[-1]):
        if np.any(forces[:,i]) < 0:
            conv = False
    return conv


