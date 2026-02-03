from ase.io import read, write
import warnings, sys, os
import yaml

from cte2bench.util.parser import parse_args, parse_config
from cte2bench.util.io import dumpYAML

import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning, module="seekpath.hpkot")
def main(argv: list[str] | None=None) -> None:
    args = parse_args(argv)

    # config.yaml file to read
    config_dir = args.config 

    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = parse_config(config)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    dumpYAML(config, f'{config["directory"]["cwd"]}/{timestamp}_config.yaml')

    if any([config['unitcell']['run'], config['strain']['run'], config['supercell']['run']]):
        from cte2bench.calculator.loader import load_calc
        calc = load_calc(config)

        if config['unitcell']['run']:
            from cte2bench.structure.unitcell import process_unitcell
            process_unitcell(config, calc)

        if config['strain']['run']:
            from cte2bench.structure.strain import process_strain
            process_strain(config, calc)

        if config['supercell']['run']:
            from cte2bench.structure.supercell import process_supercell
            process_supercell(config, calc)

    if config['harmonic']['run']:
        from cte2bench.phonon.harmonic import process_harmonic
        process_harmonic(config)

    if config['qha']['run']:
        from cte2bench.phonon.qha import process_qha
        process_qha(config)

    if args.task.lower() in ['qha']:
        from cte2bench.phonon.qha import process_qha
        process_qha(config)

if __name__ == '__main__':
    main()
