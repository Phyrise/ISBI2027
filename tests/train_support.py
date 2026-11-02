"""Support for the training-side equivalence tests (CPU only).

Each test imports the original code from its source tree and compares it with the port on small inputs. The source
trees and data are located through environment variables; a test is skipped when what it needs is absent.

  ISBI_ORIG_PROJECTS  folder holding the original training trees (mama_3d_20261018_r2, ..._r11_arch, ...)
  ISBI_ORIG_MX        the men_clinical tree (foundation_afp.py, expert_afp.py)
  ISBI_ORIG_TR        the nnUNet_translation snapshot the paper runs imported (sha identity with the pinned tags)
  ISBI_ORIG_ROOT      the paper experiment folder (manifest.json, design*.json, doses/, inits, results/, data/)
  ISBI_NNF_CKPT       nnFoundationCNN checkpoint_final.pth
  ISBI_MEDICALNET_CKPT  resnet_50_23dataset.pth
  ISBI_SLICE_WEIGHTS  folder with authors_SAM2.1_Small.pt and authors_VGG16.pt
  ISBI_MAMA_RAW       MAMA-MIA release folder (raw NIfTI; preprocessing and split tests)
  ISBI_TEST_SLOW=1    also run the heavy tests (every epoch-0 checkpoint, preprocessing of 2 raw cases)
  ISBI_TEST_REPORT    JSON-lines file where every measured gap is appended

Run (CPU):  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m unittest discover -s tests -p 'test_train_*.py' -v
"""
import contextlib
import functools
import importlib
import json
import os
import sys
import time
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TREE_MODULES = ('common', 'train', 'calibrate', 'evaluate', 'gan_parts', 'mednet_teacher', 'slice_teacher', 'launch',
                'hist_teacher', 'fork')


def env_path(name):
    v = os.environ.get(name)
    return Path(v) if v and Path(v).exists() else None


def need(*names):
    """Skip unless every environment variable names an existing path."""
    missing = [n for n in names if env_path(n) is None]
    return unittest.skipIf(bool(missing), f'missing {missing}')


def slow():
    return unittest.skipUnless(os.environ.get('ISBI_TEST_SLOW') == '1', 'set ISBI_TEST_SLOW=1')


def tree(name):
    return env_path('ISBI_ORIG_PROJECTS') / name


def root():
    return env_path('ISBI_ORIG_ROOT')


@contextlib.contextmanager
def original(tree_name, *modules):
    """Import modules of one original tree (and men_clinical from ISBI_ORIG_MX) for the duration of the block.

    The trees share module names (common, train, ...), so they are purged from sys.modules before and after."""
    for n in TREE_MODULES:
        sys.modules.pop(n, None)
    paths = [str(tree(tree_name))]
    if env_path('ISBI_ORIG_MX'):
        paths.append(str(env_path('ISBI_ORIG_MX')))
    for p in reversed(paths):
        sys.path.insert(0, p)
    try:
        mods = [importlib.import_module(m) for m in modules]
        yield mods[0] if len(mods) == 1 else mods
    finally:
        for p in paths:
            sys.path.remove(p)
        for n in TREE_MODULES:
            sys.modules.pop(n, None)


@contextlib.contextmanager
def cuda_calls_on_cpu():
    """The original training / inference code calls ``.cuda()``; on a CPU-only host make it a no-op.
    (``torch.autocast('cuda')`` and ``GradScaler('cuda')`` already disable themselves without CUDA.)"""
    saved = torch.Tensor.cuda, torch.nn.Module.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda, torch.nn.Module.cuda = saved


def cpu_fp32_teacher(build):
    """Wrap an original ``common.teacher``: build it on the CPU, with the AFP teacher network in fp32 (the paper runs
    held it in fp16 on the GPU, under fp16 autocast; neither exists on a CPU host). The slice and MedicalNet teachers
    are fp32 modules already."""
    def teacher(kind, device='cuda:0', checkpoint=None):
        import men_clinical.foundation_afp as F
        saved = F.FoundationAFP
        F.FoundationAFP = functools.partial(saved, dtype='float32')
        try:
            return build(kind, device='cpu', checkpoint=checkpoint)
        finally:
            F.FoundationAFP = saved
    return teacher


def strict_fp32():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))


def max_abs(a, b):
    if isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError(f'key sets differ: {sorted(set(a) ^ set(b))[:5]}')
        return max((max_abs(a[k], b[k]) for k in a), default=0.)
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    if a.shape != b.shape:
        raise AssertionError(f'shape {tuple(a.shape)} vs {tuple(b.shape)}')
    if a.numel() == 0:
        return 0.
    if a.dtype == torch.bool or not (a.is_floating_point() or b.is_floating_point()):
        return float((a.long() != b.long()).any())
    return float((a.double() - b.double()).abs().max())


def report(test, quantity, gap, **extra):
    row = dict(test=test, quantity=quantity, max_abs_diff=gap, time=time.strftime('%F %T'), **extra)
    print('GAP', json.dumps(row), flush=True)
    out = os.environ.get('ISBI_TEST_REPORT')
    if out:
        with open(out, 'a') as f:
            f.write(json.dumps(row) + '\n')


def arm_config(name, kind='arms', **env):
    """Load configs/<kind>/<name>.yaml with the paper locations filled in from the test environment."""
    from synthesis.core.config import load_config
    r = root()
    defaults = {'ISBI_DATA': str(r / 'data') if r else '/nonexistent', 'ISBI_RUNS': str(r / 'results') if r else '/nonexistent',
                'ISBI_WEIGHTS': '/nonexistent'}
    for k, v in dict(defaults, **env).items():
        os.environ[k] = v
    os.environ.pop('ISBI_INITS', None)
    cfg = load_config(REPO / 'configs' / kind / f'{name}.yaml')
    p = cfg['paths']
    if env_path('ISBI_NNF_CKPT'):
        p['nnf_checkpoint'] = str(env_path('ISBI_NNF_CKPT'))
    if env_path('ISBI_MEDICALNET_CKPT'):
        p['medicalnet_checkpoint'] = str(env_path('ISBI_MEDICALNET_CKPT'))
    t = cfg.get('teacher')
    if t and t['kind'] == 'nnf' and env_path('ISBI_NNF_CKPT'):
        t['checkpoint'] = str(env_path('ISBI_NNF_CKPT'))
    if t and t['kind'] == 'medicalnet' and env_path('ISBI_MEDICALNET_CKPT'):
        t['checkpoint'] = str(env_path('ISBI_MEDICALNET_CKPT'))
    if t and t['kind'] in ('sam', 'vgg') and env_path('ISBI_SLICE_WEIGHTS'):
        t['checkpoint'] = str(env_path('ISBI_SLICE_WEIGHTS') / Path(t['checkpoint']).name)
    return cfg


ALL_ARMS = ('l1', 'ssl', 'mam_scratch300', 'mam300', 'mednet', 'sam', 'vgg', 'l1_nnf', 'ssl_nnf',
            'mam_scratch300_nnfoff', 'swin_l1', 'swin_mam_scratch300', 'gan_nnf', 'gan_nnf_mam_scratch300')
# Tree whose train.py / common.py trained each arm (r10_nnfoff is not read: its train.py is byte-identical to r14's and
# its schedule is the OFFICIAL_NNF branch of r11 common.lr_at).
ARM_TREE = {'l1': ('mama_3d_20261018_r2',), 'ssl': ('mama_3d_20261018_r2',),
            'mam_scratch300': ('mama_3d_20261018_r4_ext',), 'mam300': ('mama_3d_20261018_r4_ext',),
            'mednet': ('mama_3d_20261018_r5_mednet_v4',),
            'sam': ('mama_3d_20261018_r7_percep2d',), 'vgg': ('mama_3d_20261018_r7_percep2d',),
            'l1_nnf': ('mama_3d_20261018_r6_nnfinit',), 'ssl_nnf': ('mama_3d_20261018_r6_nnfinit',),
            'mam_scratch300_nnfoff': ('mama_3d_20261018_r14_hist', 'mama_3d_20261018_r11_arch'),
            'swin_l1': ('mama_3d_20261018_r11_arch',), 'swin_mam_scratch300': ('mama_3d_20261018_r11_arch',),
            'gan_nnf': ('mama_3d_20261018_r11_arch',), 'gan_nnf_mam_scratch300': ('mama_3d_20261018_r11_arch',)}
# Experts: (config, original kind of the 0-100 run and its tree, original kind of the 100-300 run and its tree)
EXPERTS = {'nnf_ft': ('expert', 'mama_3d_20261018_r2', 'expert_c300', 'mama_3d_20261018_r4_ext'),
           'scratch': ('expert_scratch', 'mama_3d_20261018_r3_scratch', 'expert_scratch_c300', 'mama_3d_20261018_r4_ext')}
