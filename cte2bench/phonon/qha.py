import os, gc
from tqdm import tqdm
import os.path as osp
from phonopy.api_qha import PhonopyQHA
from phonopy.file_IO import read_thermal_properties_yaml
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import ase.io as ase_IO
import matplotlib
from cte2bench.util.io import loadPKL, loadJSON, dumpJSON, clean_for_json
import pandas as pd

#TODO: rcparams

def process_qha(config):
    calc_tag = config['calculator']['tag']
    base_dir = config['directory']['cwd']
    distance = config['supercell']['distance']
 
    unit_dct_file = f'{base_dir}/{calc_tag}-unitcell.pkl'
    unit_dct = loadPKL(unit_dct_file)
    RESULTS = loadJSON(f'{base_dir}/{calc_tag}_results.json')

    # -------- preprocess --------- #
    conf = config['qha']
    thin_number = config['qha']['thin_number']
    qha_eps_list = [f'e{eps}' for eps in conf['eps']]
    desc = 'QHA'

    for idx, _dct in tqdm(unit_dct.items(), desc=desc):
        results = RESULTS.get(str(idx), {str(idx): '??'})

        suffix = _dct['suffix']
        mesh_dir = f'{base_dir}/{suffix}/{config["harmonic"]["save"]}'
        strain_dir = f'{base_dir}/{suffix}/{config["strain"]["save"]}'
        strain_dct_file = f'{strain_dir}/{calc_tag}-strain_dct-{suffix}.pkl'
        strain_dct = loadPKL(strain_dct_file)
        strain_opt = ase_IO.read(f'{strain_dir}/{calc_tag}-strain_relax-{suffix}.extxyz', index=':')

        primitive_matrix = _dct.get('primitive_matrix', np.eye(3))

        mesh_dct = results.get('harmonic', None)
        if not mesh_dct:
            print(f'WARNING: {suffix} - mesh calculation must be preceded')
            continue

        cwd = f'{base_dir}/{suffix}/{conf["save"]}'
        cwd_plot = f'{cwd}/{conf["plot"]}'
        cwd_data = f'{cwd}/{conf["data"]}'
        cwd_full = f'{cwd}/{conf["full"]}'

        os.makedirs(cwd, exist_ok=True)
        os.makedirs(cwd_plot, exist_ok=True)
        os.makedirs(cwd_data, exist_ok=True)
        os.makedirs(cwd_full, exist_ok=True)

        eps_list = []
        thermal_filenames= []
        volumes = []
        free_energies = []

        for i, (key, m_dct) in enumerate(mesh_dct.items()):
            if key not in qha_eps_list:
                continue
            thermal_props = f'{mesh_dir}/{key}/thermal_properties_{key}.yaml'
            if not (m_dct.get('QHA') or os.path.isfile(thermal_props)):
                continue
            strained = strain_opt[i]
            eps_list.append(key)
            thermal_filenames.append(thermal_props)
            volumes.append(strained.get_volume()* np.linalg.norm(np.linalg.det(primitive_matrix)))
            free_energies.append(strained.info.get('e_fr_energy', strain_dct[key].get('e_fr_energy',0)) * np.linalg.norm(np.linalg.det(primitive_matrix)))

        temperatures, cv, entropy, fe_phonon, _, _ = read_thermal_properties_yaml(filenames=thermal_filenames)
        temperatures = np.array(temperatures, dtype=float)
        cv = np.array(cv, dtype=float)
        entropy = np.array(entropy, dtype=float)
        fe_phonon = np.array(fe_phonon, dtype=float)
        volumes, free_energies = np.array(volumes, dtype=float), np.array(free_energies, dtype=float)

        qha_kwargs = {'volumes': volumes, 'electronic_energies': free_energies,
                      'temperatures': temperatures, 'free_energy': fe_phonon,
                      'cv': cv, 'entropy': entropy, 'eos': conf['eos'], 't_max': conf['t_max'],
                      'verbose': True}

        if len(volumes) < 5:
            print('At least 5 volume points needed for EOS fitting .. returning')
            continue

        with open(f'{cwd}/qha.x', 'w') as f, redirect_stdout(f), redirect_stderr(f):
            qha = PhonopyQHA(**qha_kwargs)
   
        # plot everything at once
        print('plotting qha results')
        os.chdir(cwd_plot)
        qha.plot_qha(thin_number=thin_number).savefig(f'{cwd}/qha_plot.svg')
        qha.plot_qha(thin_number=thin_number).savefig(f'{cwd}/qha_plot.pdf')
        matplotlib.pyplot.close()

        qha.plot_helmholtz_volume(thin_number=thin_number).savefig('helmholtz_volume.svg')
        qha.plot_volume_temperature().savefig('volume_temperature.svg')
        qha.plot_thermal_expansion().savefig('thermal_expansion.svg')
        matplotlib.pyplot.close()

        qha.plot_gibbs_temperature().savefig('gibbs_temperature.svg')
        qha.plot_bulk_modulus_temperature().savefig('bulk_modulus.svg')
        matplotlib.pyplot.close()

        try:
            qha.plot_heat_capacity_P_polyfit().savefig('heat_capacity_P_poly.svg')
            qha.plot_heat_capacity_P_numerical().savefig('heat_capacity_P_numer.svg')

        except Exception as exc:
            print(exc)

        qha.plot_gruneisen_temperature().savefig('gruneisen_temperature.svg')
        matplotlib.pyplot.close()

        # save dat files at once
        print('writting down qha data')
        os.chdir(cwd_data)
        qha.write_helmholtz_volume()
        qha.write_helmholtz_volume_fitted(thin_number=thin_number)
        qha.write_volume_temperature()
        qha.write_thermal_expansion()
        qha.write_gibbs_temperature()
        qha.write_bulk_modulus_temperature()

        try:
            qha.write_heat_capacity_P_numerical()
            qha.write_heat_capacity_P_polyfit()
        except Exception as exc:
            print(exc)

        qha.write_gruneisen_temperature()

        results['CTE'] = {'CALC': {10: None, 300: None, 500: None, 800: None}}
        df = pd.read_csv(f'{cwd_data}/thermal_expansion.dat',
                              names=['temp', 'cte'], header=None,
                              comment='#', sep=r'\s+')
        results['CTE'] = {'CALC': {
                                    10: df.loc[df['temp']==10, 'cte'].to_numpy(),
                                    300: df.loc[df['temp']==300, 'cte'].to_numpy(),
                                    500: df.loc[df['temp']==500, 'cte'].to_numpy(),
                                    800: df.loc[df['temp']==800, 'cte'].to_numpy(),
                                    }
                          }
        RESULTS[str(idx)].update(clean_for_json(results))
        dumpJSON(RESULTS[str(idx)], f'{base_dir}/{suffix}/{calc_tag}_results.json')

        # thin_numbers were set for readability, plot entire data
        os.chdir(cwd_full)
        qha.write_helmholtz_volume_fitted(thin_number=config['harmonic']['t_step'])
        qha.plot_pdf_helmholtz_volume(thin_number=config['harmonic']['t_step'])

        os.chdir(cwd)
        # plot eos
        qha._bulk_modulus.plot().savefig(f'{cwd}/{conf["eos"]}.svg')

        matplotlib.pyplot.close()
        del qha
        gc.collect()
    dumpJSON(clean_for_json(RESULTS), f'{base_dir}/{calc_tag}_results.json')
