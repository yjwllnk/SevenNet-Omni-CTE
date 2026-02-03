import json, yaml, pickle
import os
from typing import Any, Optional, Union
import json
import numpy as np

def clean_for_json(obj):
    if isinstance(obj, np.generic):
        return obj.item()   # unwrap numpy scalar
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_for_json(x) for x in obj]
    return obj

class InlineListEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        # base generator
        for chunk in super().iterencode(obj, _one_shot=_one_shot):
            yield chunk

    def _iterencode_list(self, lst, _current_indent_level):
        # Force lists to stay on one line if they contain only scalars
        if all(not isinstance(i, (list, dict)) for i in lst):
            # inline: [1, 2, 3]
            return '[' + ', '.join(self.encode(i) for i in lst) + ']'
        # fallback to normal pretty-print behavior
        return super()._iterencode_list(lst, _current_indent_level)

def dumpJSON(data, filename, indent=4, sort_keys=False):
    with open(filename, 'w') as fp:
        json.dump(data, fp, cls=InlineListEncoder, indent=indent, sort_keys = sort_keys, ensure_ascii=False)

def loadJSON(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def dumpPKL(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def loadPKL(path):
    with open(path, 'rb') as f:
        data=pickle.load(f)
    return data


def dict_representer(dumper, data=None):
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data, flow_style=False)

def list_representer(dumper, data=None):
    return dumper.represent_sequence(yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, data, flow_style=True)

class WDumper(yaml.Dumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()

def dumpYAML(data, filename, indent=4, sort_keys=False, explicit_start=True, explicit_end=True, default_flow_style=False,):
    yaml.add_representer(dict, dict_representer, Dumper=WDumper)
    yaml.add_representer(list, list_representer, Dumper=WDumper)
    with open(filename, 'w') as fp:
        yaml.dump(data, fp, Dumper=WDumper, sort_keys=sort_keys, explicit_start=explicit_start, explicit_end=explicit_end, default_flow_style=default_flow_style, indent=indent)


def DatToCsv(inp: Union[str, os.PathLike], out: Union[str, os.PathLike], columns: Optional[str]=None) -> None:
    """
    ----------------

    Parameters
    ----------------
    inp: str|os.PathLike
        dir of the input .dat file

    out: str|os.PathLike
        dir of the output .csv file 

    columns: Optional[str]
        the columns of the output csv file
        something like 'temperature,cte'
        don't forget to remove all whitespaces

    Returns:
    ----------------
    Nothing 
    ...
    
    """

    inp_dat = open(inp, 'r')
    out = open(out, 'w', buffering=1)
    dat_lines = inp_dat.readlines()
    if columns is not None:
        out.writelines(f'{columns}\n')
    for _, dat_line in enumerate(dat_lines):
        if '#' in dat_line:
            continue
        try:
            col1=dat_line.split()[0]
            col2=eval(dat_line.split()[1])
            out.write(f'{col1},{col2}\n')
        except Exception as e:
            print('ERROR')
            print(e)
            continue
    inp_dat.close()
    out.close()

