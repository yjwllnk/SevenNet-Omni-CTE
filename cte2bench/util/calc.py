import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from ase.calculators.singlepoint import SinglePointCalculator

def calc_from_py(script): # TODO
    import importlib.util
    from pathlib import Path

    file_path = Path(script).resolve()
    spec = importlib.util.spec_from_file_location('generate_calc', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calc = module.generate_calc()
    return calc

def single_point_calculate(atoms, calc):
    start_wall = time.time()
    start_dt = datetime.now()

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    end_wall = time.time()
    end_dt = datetime.now()

    time_dct = {'start': {'wall': start_wall, 'date': start_dt.strftime('%Y-%m-%d %H:%M:%S')},
                'end': {'wall': end_wall, 'date': end_dt.strftime('%Y-%m-%d %H:%M:%S')},
                }

    calc_results = {"energy": energy, "forces": forces, "stress": stress}
    calculator = SinglePointCalculator(atoms, **calc_results)
    new_atoms = calculator.get_atoms()
    new_atoms.info['e_fr_energy'] = new_atoms.get_potential_energy()
    new_atoms.info['oneshot'] = time_dct
    return new_atoms


def single_point_calculate_list(atoms_list, calc, desc=None):
    calculated = []
    for atoms in tqdm(atoms_list, desc=desc, leave=False):
        calculated.append(single_point_calculate(atoms, calc))
    return calculated
