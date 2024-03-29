
from copy import deepcopy
from collections import namedtuple
import os.path as osp
import json
import numbers

from exptools.collections import AttrDict

VARIANT = "variant_config.json"

VariantLevel = namedtuple("VariantLevel", ["keys", "values", "dir_names"])
''' Use variant levels to descript a single combination you want to make.
    NOTE: last two items should be the lists with the same length
Parameters
----------
    "keys": a list of tuples. In each tuple, specifies the path to get acccess 
        to the exact value via dict[key[0]][key[1]]...
        Then, the length of the key should match the length of each item in values.
    "values": a list of list of values you want to assign (in terms of each key-paths)
    "dir_names": a list of names you want to addd in terms of this variant
        (to make the log_dir more interpertable)
'''

def make_variants(*variant_levels, create_subdirs= True):
    """ Given a list of variant_levels, do _cross_variants to generate
    a list of variant conbinations and the sub-directory that they should stay.
    Args:
        create_subdirs: if True, each variant will create a subdir. Otherwise, the dir name will be connected by "_"
    """
    variants, log_dirs = [AttrDict()], [""]
    for variant_level in variant_levels:
        variants, log_dirs = _cross_variants(
            variants, log_dirs, variant_level,
            create_subdirs= create_subdirs,
        )
    return variants, log_dirs


def _cross_variants(prev_variants, prev_log_dirs, variant_level, create_subdirs= True):
    """For every previous variant, make all combinations with new values."""
    keys, values, dir_names = variant_level
    assert len(prev_variants) == len(prev_log_dirs)
    assert len(values) == len(dir_names)
    assert len(keys) == len(values[0])
    assert all(len(values[0]) == len(v) for v in values)

    variants = list()
    log_dirs = list()
    for prev_variant, prev_log_dir in zip(prev_variants, prev_log_dirs):
        for vs, n in zip(values, dir_names):
            variant = prev_variant.copy()
            log_dir = osp.join(prev_log_dir, n) if create_subdirs else prev_log_dir + "_" + n
            if log_dir in log_dirs:
                raise ValueError("Names must be unique.")
            for v, key_path in zip(vs, keys):
                current = variant
                for k in key_path[:-1]:
                    if k not in current:
                        current[k] = AttrDict()
                    current = current[k]
                current[key_path[-1]] = v
            variants.append(AttrDict(variant))
            log_dirs.append(log_dir)
    return variants, log_dirs


def load_variant(log_dir):
    with open(osp.join(log_dir, VARIANT), "r") as f:
        variant = json.load(f)
    return AttrDict(variant)


def save_variant(variant, log_dir):
    with open(osp.join(log_dir, VARIANT), "w") as f:
        json.dump(variant, f, indent= 4)


def update_config(default, variant):
    """Performs deep update on all dict structures from variant, updating only
    individual fields, which must be present in default."""
    new = default.copy()
    for k, v in variant.items():
        if k not in new:
            raise KeyError(f"Variant key {k} not found in default config.")
        if isinstance(v, dict) != isinstance(new[k], dict):
            raise TypeError(f"Variant dict structure at key {k} mismatched with"
                " default.")
        new[k] = update_config(new[k], v) if isinstance(v, dict) else v
    return new

def flatten_variant4hparams(variant: dict):
    """ Make the nested dictionary into a single-level dictionary
        NOTE/WARNING: the input variable is modified.
    """
    kv_pair = [i for i in variant.items()]
    for key, value in kv_pair:
        assert isinstance(key, str)
        if isinstance(value, dict):
            sub_variant = flatten_variant4hparams(value)
            for k, v in sub_variant.items():
                variant.update({
                    key + "." + k: v
                })
            variant.pop(key)
        elif not isinstance(value, numbers.Number):
            variant[key] = str(value)
    return variant
