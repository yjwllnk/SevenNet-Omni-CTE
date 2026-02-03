
def load_sevenn(config):
    from cte2bench.calculator.sevenn_calculator import return_calc
    calc = return_calc(config)
    return calc

def load_esen(config):
    from cte2bench.calculator.esen_calculator import return_calc
    calc = return_calc(config)
    return calc

def load_pet(config):
    from cte2bench.calculator.pet_calculator import return_calc
    calc = return_calc(config)
    return calc


def load_calc(config):
    calc_type = config['calculator']['calc'].lower()
    if calc_type in ['7net', 'sevenn', 'sevennet']:
        calc = load_sevenn(config)

    elif calc_type == 'pet':
        calc = load_pet(config)

    elif calc_type == 'esen':
        calc = load_esen(config)

    return calc
