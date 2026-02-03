from upet.calculator import UPETCalculator


def return_calc(config):
    conf = config['calculator']
    calc, modal, model = conf['calc'].lower(), conf['model'].lower(), conf['modal'].lower()

    calc_kwargs = {
            'model': f'{calc}-{modal}-{model}',
            'version': 'latest',
            'non_conservative': False,
            'device': "cuda",
            'calculate_uncertainty': False,
            'calculate_ensemble': False,
            }

    print(f"[PET] model={model}, modal={modal}")

    calc = UPETCalculator(**calc_kwargs)
    return calc
