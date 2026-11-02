"""YAML configuration loading.

A config may name a parent with ``extends: <relative path>``; mappings are merged recursively, the child wins; a child
mapping with ``_replace: true`` replaces the parent's mapping instead of being merged into it.
Strings may contain ``${VAR}`` environment references; they are expanded at load time (``ISBI_REPO`` defaults to the
repository root). A reference that is still unexpanded when a path is actually needed raises (see ``path``); an
optional file (``is_set``) whose variable is unset counts as absent.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml


def _merge(base, child):
    out = copy.deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and value.get('_replace'):
            out[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != '_replace'}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    os.environ.setdefault('ISBI_REPO', str(REPO_ROOT))
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop('extends', None)
    if parent is not None:
        raw = _merge(load_config(path.parent / parent), raw)
    raw = _expand(raw)
    raw['_config_file'] = str(path)
    return raw


LOCATION_KEYS = ('_config_file', 'paths', 'checkpoint', 'file')


def _recipe(value):
    if isinstance(value, dict):
        return {k: _recipe(v) for k, v in value.items() if k not in LOCATION_KEYS}
    if isinstance(value, list):
        return [_recipe(v) for v in value]
    return value


def config_sha256(cfg):
    """Digest of the resolved recipe; file locations (paths, checkpoints, init files) are excluded."""
    return hashlib.sha256(json.dumps(_recipe(cfg), sort_keys=True, default=str).encode()).hexdigest()


def path(value, what='path'):
    """Return a Path, refusing unexpanded ${VAR} references."""
    if value is None:
        raise ValueError(f'{what} is not set')
    value = str(value)
    if '${' in value:
        raise ValueError(f'{what} uses an unset environment variable: {value}')
    return Path(value)


def is_set(value):
    """True for a non-empty value without unexpanded ${VAR} references."""
    return bool(value) and '${' not in str(value)


def torch_dtype(name):
    """'float16' / 'bfloat16' / 'float32' / None -> torch dtype (None and float32 mean: no autocast)."""
    import torch
    if name in (None, 'float32', 'fp32'):
        return None
    return {'float16': torch.float16, 'fp16': torch.float16, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}[name]
