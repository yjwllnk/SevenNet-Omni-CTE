"""
Modified based on Jinmu Yu's code
"""

from types import NotImplementedType
import warnings
from sevenn.calculator import SevenNetCalculator

CALC_DCT = {
    'ompa': '',
    'omni': '',
    }

FUNC_DCT = {
    'mpa': 'PBE',
    'omat24': 'PBE',
    'matpes_pbe': 'PBE',
    'spice': 'wB97M',
    'qcml': 'PBE0',
    'oc20': 'RPBE',
    'oc22': 'PBE',
    'mp_r2scan': 'r2SCAN',
    'matpes_r2scan': 'r2SCAN',
}

def return_calc(config):
    conf = config['calculator']
    model, modal = conf['model'], conf['modal']
    model_path = CALC_DCT[model]
    calc_kwargs = {
        'model': model_path,
        'modal': modal,
        'enable_flash': True, #TODO;
    }

    # functional = FUNC_DCT.get(modal, None)
    print(f"[SevenNet] model={model}, modal={modal}")
    print(f"[SevenNet] potential path: {model_path}")

    calc = SevenNetCalculator(**calc_kwargs)
    if conf.get('d3'):
        from ase.calculators.mixing import SumCalculator
        from sevenn.calculator import D3Calculator
        d3 = D3Calculator()
        calc_d3 = D3Calculator(functional_name = 'pbe')
        return SumCalculator([calc, calc_d3])
    return calc
