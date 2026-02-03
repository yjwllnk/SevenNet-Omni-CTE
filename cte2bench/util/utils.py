import numpy as np
import warnings
import yaml

import h5py
from ase import Atoms

import spglib
from phonopy.structure.atoms import PhonopyAtoms

def log_stats(config, atoms, task='NaN', stat='oneshot', eps=0, disp='#N/A'):
    atoms.calc = None # Error, property "free_energy" is not available . . .
    logfile = open(config['directory']['logfile'], 'a')
    lengths = ','.join(str(round(l,5)) for l in atoms.cell.lengths())
    angles = ','.join(str(round(d, 3)) for d in atoms.cell.angles())
    _dct = atoms.info.copy()

    try:
        time_dct = _dct.get(stat)
    except Exception as exec:
        time_dct = _dct.get('time_dct', {})
        time_dct = time_dct.get(stat)

    stat_dct = {
            # "date_i": time_dct["start"].get("date", "date_i"),
                # "date_f": time_dct["end"].get("date", "date_f"),
                'ID': _dct.get("ID", "ID-?"),
                "material_id": _dct.get("material_id", "mp-??"),
                "name": _dct.get("name", atoms.get_chemical_formula(empirical=True)),
                "task": task,
                "eps": eps,
                "disp": disp,
                "type": stat,
                "steps": _dct.get("steps", "steps?"),
                "force_conv": _dct.get("force_conv", "force_conv?"),
                "wall_i": time_dct["start"].get("wall","wall_i"),
                "wall_f": time_dct["end"].get("wall","wall_f"),
                "symm": _dct.get(f"symm.no.{task}", _dct.get("symm.no", 'task.symm.no?')),
                "natom": len(atoms),
                "energy": _dct.get("e_fr_energy", _dct.get('energy', 'energy?')),
                "volume": atoms.get_volume(),
                "a": atoms.cell.lengths()[0],
                "b": atoms.cell.lengths()[1],
                "c": atoms.cell.lengths()[2],
                "alpha": atoms.cell.angles()[0],
                "beta": atoms.cell.angles()[1],
                "gamma": atoms.cell.angles()[2],
                }
    head = ','.join(k for k in stat_dct.keys())
    line=','.join(str(v) for v in stat_dct.values())
    logfile.writelines(f'{line}\n')
    logfile.close()
    print('\n')
    print('----------------------------------------------------------------------------------------------------------------')
    print(f'[LOG]')
    print(f'{time_dct["start"].get("date", "date_i")}')
    print(head)
    print(line)
    print(f'{time_dct["end"].get("date", "date_i")}')
    print('----------------------------------------------------------------------------------------------------------------')
    print('\n')


def phonoatoms2aseatoms(phonoatoms):
    atoms = Atoms(
        phonoatoms.symbols,
        cell=phonoatoms.cell,
        positions=phonoatoms.positions,
        pbc=True
    )
    return atoms

def aseatoms2phonoatoms(atoms):
    phonoatoms = PhonopyAtoms(
        atoms.symbols,
        cell=atoms.cell,
        positions=atoms.positions,
    )
    return phonoatoms

def strain_c_axis(atoms, eps):
    cell = atoms.get_cell()
    cell[2] *= (1+eps)
    atoms.set_cell(cell, scale_atoms=True)
    return atoms

def check_imaginary_freqs(frequencies: np.ndarray) -> bool:
    try:
        if np.all(np.isnan(frequencies)):
            return True

        if np.any(frequencies[0, 3:] < 0):
            return True

        if np.any(frequencies[0, :3] < -1e-2):
            return True

        if np.any(frequencies[1:] < 0):
            return True
    except Exception as e:
        warnings.warn(f"Failed to check imaginary frequencies: {e}")

    return False

def get_spgnum(atoms, symprec=1e-5):
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    spgdat = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return spgdat.number

def get_spg(atoms, symprec=1e-5):
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    spg = spglib.get_spacegroup(cell, symprec=symprec)
    return spg

def load_mesh_yaml(mesh_yaml_path):
    with open(mesh_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    qblocks = data.get('phonon') or data.get('mesh') or data.get('mesh_points') or data['mesh']
    # try to be robust if key names differ:
    if not isinstance(qblocks, list):
        # some phonopy dumps nest differently
        qblocks = data['phonon'].get('qpoints', data['mesh'].get('qpoints'))

    weights = []
    freqs_list = []
    for qp in qblocks:
        w = qp.get('weight')
        w = int(w)
        bands = qp.get('band') or qp.get('bands')
        freqs = [float(b.get('frequency')) for b in bands]
        weights.append(w)
        freqs_list.append(freqs)

    weights = np.array(weights, dtype=float)         # shape (n_q,)
    freqs = np.array(freqs_list, dtype=float)       # shape (n_q, n_bands)
    return weights, freqs

def load_mesh_hdf5(mesh_hdf5_path):
    """
    Load q-point weights and frequencies from phonopy mesh.hdf5.

    Returns
    -------
    weights : np.ndarray, shape (n_q,)
    freqs   : np.ndarray, shape (n_q, n_bands)
    """
    with h5py.File(mesh_hdf5_path, "r") as f:
        if "frequency" not in f:
            raise ValueError("mesh.hdf5 missing 'frequency' dataset")

        freqs = f["frequency"][()]  # (n_q, n_bands)

        if "weight" in f:
            weights = f["weight"][()]  # (n_q,)
        else:
            # fallback: uniform weights
            weights = np.ones(freqs.shape[0], dtype=float)

    freqs = np.asarray(freqs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return weights, freqs


def imag_dos_frac(freqs: np.ndarray,
                      weights: np.ndarray | None = None) -> float:
    """
    Compute the DOS-weighted frac of imaginary modes (freq < 0)
    given a frequency matrix and optional q-point weights.

    Parameters
    ----------
    freqs : np.ndarray
        Shape (n_q, n_b). Frequencies in THz. Imaginary = negative.
    weights : np.ndarray or None
        Shape (n_q,). If None, all q-points get weight 1.

    Returns
    -------
    float
        frac of imaginary-mode weight (0 to 1).
    """

    if freqs.ndim != 2:
        raise ValueError("freqs must have shape (n_q, n_b).")

    n_q, n_b = freqs.shape

    # uniform weights if none provided
    if weights is None:
        weights = np.ones(n_q, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_q,):
            raise ValueError(f"weights must have shape ({n_q},), got {weights.shape}")

    # mode weights = q-weight replicated across bands
    mode_weights = np.repeat(weights[:, None], n_b, axis=1)

    imag_mask = freqs < 0

    imag_area = np.sum(mode_weights * imag_mask)
    total_area = np.sum(mode_weights)

    if total_area == 0:
        raise ValueError("Total mode weight is zero. Check weights.")

    return float(imag_area / total_area)
