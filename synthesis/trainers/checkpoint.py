"""Checkpoints, resume and fork.

A checkpoint holds everything a resume needs: ``network_weights``, ``optimizer_state``, ``grad_scaler_state``,
``current_epoch``, the torch / CUDA / numpy / python RNG states and, for the GAN, ``discriminator_weights`` and
``optimizer_d_state``. The key names are those of the paper checkpoints, which therefore load here unchanged.
``checkpoint_latest.pth`` is rewritten after every epoch; ``checkpoint_epNNN.pth`` snapshots are byte copies of it.

Resume restores weights, optimiser and scaler states, the RNG states and the epoch; the sampler is stateless, so the
resumed run draws the batches it would have drawn without interruption (bit-identity is not claimed on GPU:
cuDNN autotuning is on).

Fork (``python -m synthesis.trainers.checkpoint fork``): seeds a new run directory from another run's snapshot (checkpoint as
``checkpoint_latest.pth`` plus the batch plan and epoch log up to that epoch). This is how the two experts were
continued from epoch 100 to 300 in the paper; training 0 -> 300 in one run directory with ``--resume`` is equivalent.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from synthesis.core.io import atomic_torch_save, save_json, sha256


def payload(model, optimizer, scaler, epoch, run_name, record, config_sha, discriminator=None, optimizer_d=None):
    p = {'network_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
         'grad_scaler_state': scaler.state_dict(), 'current_epoch': epoch, 'kind': run_name, 'proof': record,
         'torch_rng': torch.get_rng_state(),
         'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
         'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(), 'config_sha256': config_sha}
    if discriminator is not None:
        p.update(discriminator_weights=discriminator.state_dict(), optimizer_d_state=optimizer_d.state_dict())
    return p


def optimizer_to(optimizer, device, keep_step_on_cpu=False):
    """Move optimiser state tensors to the device (Adam's 'step' stays on CPU, as torch expects)."""
    for state in optimizer.state.values():
        for k, t in state.items():
            if torch.is_tensor(t) and not (keep_step_on_cpu and k == 'step'):
                state[k] = t.to(device)


def restore(ck, model, optimizer, scaler, device, discriminator=None, optimizer_d=None, keep_step_on_cpu=False):
    model.load_state_dict(ck['network_weights'], strict=True)
    optimizer.load_state_dict(ck['optimizer_state'])
    optimizer_to(optimizer, device, keep_step_on_cpu)
    if discriminator is not None:
        discriminator.load_state_dict(ck['discriminator_weights'], strict=True)
        optimizer_d.load_state_dict(ck['optimizer_d_state'])
        optimizer_to(optimizer_d, device, keep_step_on_cpu)
    scaler.load_state_dict(ck['grad_scaler_state'])
    torch.set_rng_state(ck['torch_rng'])
    if torch.cuda.is_available() and ck.get('cuda_rng'):
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    np.random.set_state(ck['numpy_rng'])
    random.setstate(ck['python_rng'])
    return int(ck['current_epoch'])


def save_epoch(out, p, epoch, snapshots, stop):
    latest = Path(out) / 'checkpoint_latest.pth'
    atomic_torch_save(latest, p)
    if epoch in snapshots or epoch == stop:
        dst = Path(out) / f'checkpoint_ep{epoch:03d}.pth'
        tmp = dst.with_suffix('.tmp')
        shutil.copyfile(latest, tmp)
        tmp.replace(dst)


def save_initial(out, p):
    """Epoch-0 snapshot (the initial state), written before the first update."""
    atomic_torch_save(Path(out) / 'checkpoint_ep000.pth', p)


def truncate_plan(planfile, n_lines):
    """On resume, keep the first n_lines of the batch plan (the epochs that the checkpoint contains)."""
    lines = Path(planfile).read_text().splitlines()
    if len(lines) < n_lines:
        raise RuntimeError(f'batch plan has {len(lines)} lines, checkpoint needs {n_lines}')
    tmp = Path(planfile).with_suffix('.resume_tmp')
    tmp.write_text('\n'.join(lines[:n_lines]) + '\n')
    tmp.replace(planfile)


def fork(parent_dir, child_dir, epoch, steps_per_epoch):
    parent, child = Path(parent_dir), Path(child_dir)
    src = parent / f'checkpoint_ep{epoch:03d}.pth'
    if child.exists() and any(child.iterdir()):
        raise RuntimeError(f'{child} is not empty')
    ck = torch.load(src, map_location='cpu', weights_only=False)
    if int(ck['current_epoch']) != epoch:
        raise ValueError(f'{src} holds epoch {ck["current_epoch"]}')
    del ck
    plan = (parent / 'batch_plan.jsonl').read_text().splitlines()
    if len(plan) < epoch * steps_per_epoch:
        raise ValueError('parent batch plan is shorter than the fork epoch')
    rows = {}
    for line in (parent / 'epoch_metrics.jsonl').read_text().splitlines():
        if line.strip():
            rows[int(json.loads(line)['epoch'])] = line
    child.mkdir(parents=True, exist_ok=True)
    tmp = child / 'checkpoint_latest.pth.tmp'
    shutil.copyfile(src, tmp)
    tmp.replace(child / 'checkpoint_latest.pth')
    (child / 'batch_plan.jsonl').write_text('\n'.join(plan[:epoch * steps_per_epoch]) + '\n')
    (child / 'epoch_metrics.jsonl').write_text('\n'.join(rows[e] for e in range(1, epoch + 1) if e in rows) + '\n')
    record = {'parent': str(parent), 'epoch': epoch, 'parent_checkpoint_sha256': sha256(src),
              'copy_sha256': sha256(child / 'checkpoint_latest.pth')}
    record['verified'] = record['parent_checkpoint_sha256'] == record['copy_sha256']
    save_json(child / 'fork.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description='Fork a run directory from a snapshot of another run.')
    p.add_argument('action', choices=['fork'])
    p.add_argument('--parent', required=True)
    p.add_argument('--child', required=True)
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--steps-per-epoch', type=int, default=250)
    a = p.parse_args()
    print(json.dumps(fork(a.parent, a.child, a.epoch, a.steps_per_epoch)))


if __name__ == '__main__':
    main()
