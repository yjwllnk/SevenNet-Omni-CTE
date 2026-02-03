import os
import argparse

def parse_args(argv: list[str]| None=None):
    parser = argparse.ArgumentParser(description= "cli tool")

    parser.add_argument('--task', type=str, default='all',
            help= 'relax, fc2, phonon, gpu, cpu, qha etc')

    parser.add_argument('--config', type=str, default='./config.yaml', 
            help='config yaml file directory')

    parser.add_argument('--calc', type=str, default='7net',
            help='7net, mace, orb, eSEN, dpa, UMA etc')

    parser.add_argument('--model', type=str, default='omni',
            help='omni, ompa, omat, zero')

    parser.add_argument('--modal', type=str, default='omat24',
            help='mpa, omat24, matpes_pbe, mp_r2scan, matpes_r2scan')

    return parser.parse_args(argv)

def overwrite_default(config, argv: list[str] | None=None):
    args = parse_args(argv)
    config['calculator']['calc'] = args.calc
    config['calculator']['model'] = args.model.lower()
    config['calculator']['modal'] =args.modal.lower()

    is_7 = (args.calc.lower() in ['7net', 'sevenn', 'sevennet'])

    if is_7:
        tag = f'{args.model.lower()}_{args.modal}'
    else:
        tag = f'{args.calc.upper()}_{args.model.lower()}_{args.modal}'

    config['calculator']['tag'] = tag
    
    if is_7:
        config['directory']['prefix'] = f'./{args.model.lower()}/{args.modal.lower()}'
    else:
        config['directory']['prefix'] = f'./{args.calc.lower()}/{args.model.lower()}/{args.modal.lower()}'

    config['directory']['cwd'] = os.path.abspath(config['directory']['prefix'])
    os.makedirs(config['directory']['cwd'], exist_ok = True)
    config['directory']['logfile'] = f'{config["directory"]["cwd"]}/{tag}_stats.log'
    if os.path.isfile(config['directory']['logfile']):
        logfile = open(config['directory']['logfile'], 'a')
    else:
        logfile = open(config['directory']['logfile'], 'w')
    logfile.write(f'ID,MP-ID,NAME,TASK,EPSILON,DISP,TYPE,STEPS,FORCE_CONV,WALL_I,WALL_F,SYMM,NATOM,ENERGY,VOLUME,A,B,C,ALPHA,BETA,GAMMA\n')
    logfile.close()
    config['calculator']['calc_args']['modal'] = args.modal.lower()
    # TODO: enable flash if avail
    return config

def check_dir_config(config):
    conf = config['directory']
    assert os.path.isfile(conf['input']), f'input files not found at {conf["input"]}'

def check_unitcell_config(config):
    conf = config['unitcell'].copy()
    assert isinstance(conf.get('run'), (type(None), bool))
    if conf.get('load'):
        assert os.path.isfile(conf['load'])

def check_strain_config(config):
    conf = config['strain'].copy()
    assert isinstance(conf.get('run'), (type(None), bool))
    if conf.get('load'):
        assert os.path.isfile(conf['load'])
    if conf.get('load_opt'):
        assert os.path.isfile(conf['load_opt'])

def check_supercell_config(config):
    conf = config['supercell']
    assert isinstance(conf.get('run'), (type(None), bool))
    if conf.get('load'):
        assert os.path.isfile(conf['load'])
    assert isinstance(conf['distance'], float)
    # assert isinstance(conf.get('symm_fc2'), (bool, int))

def check_harmonic_config(config):
    conf = config['harmonic']
    for run in ['run_mesh', 'run_thermal', 'run_dos', 'run_band']:
        assert isinstance(conf.get(run), (type(None), bool))

    for load in ['load_thermal']:
        if conf.get('load'):
            assert os.path.isfile(conf['load'])

    for t in ['t_min', 't_max', 't_step']:
        if conf.get(t):
            assert isinstance(conf[t], (int, float))


def check_qha_config(config):
    conf = config['qha']
    assert isinstance(conf.get('run'), (type(None), bool))
    assert conf['eos'] in ['birch', 'vinet', 'birch_murnaghan']
    
    if conf.get('thin_number'):
        assert isinstance(conf['thin_number'], (int, float))
    elif conf.get('spares'):
        assert isinstance(conf['sparse'], (int, float))
    else:
        print("WARNING: your QHA plot's going to look like rubbish")

def check_calc_config(config):
    conf = config['calculator']
    if conf['calc'].lower() in ['sevennet', 'seven', 'sevenn', '7net']:
        assert os.path.isfile(conf['path'])

def parse_config(config, argv: list[str] | None=None):
    config = overwrite_default(config, argv)

    check_dir_config(config)
    check_calc_config(config)

    check_unitcell_config(config)
    check_strain_config(config)
    check_supercell_config(config)
    check_harmonic_config(config)
    check_qha_config(config)

    return config
