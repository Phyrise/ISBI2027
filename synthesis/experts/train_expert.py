"""Train one of the two AFP lesion experts, then run its quality gate.

The experts are ResEnc-L networks (2 classes) trained by this repository's own SGD loop (synthesis/trainers/sgd_trainer.py) with
the nnU-Net Dice + CE loss (synthesis/losses/segmentation.py) on the same stateless sampler and patch as the generators. They are
NOT trained with a stock nnUNetv2 trainer: no nnU-Net planning, preprocessing or augmentation (paired mirrors only).
Input: the post-contrast patch, z-scored per patch; target: the lesion mask, -1 (ignored) outside the volume.
Recipe: seed 20261019, SGD Nesterov 0.99, weight decay 3e-5, lr 1e-3, 15-epoch warm-up, PolyLR 0.9 to 300 epochs,
float16 autocast, clip 12.
- configs/experts/nnf_ft.yaml (teacher of AFP_seg-ft-nnF, mam300): nnFoundationCNN encoder + random decoder.
- configs/experts/scratch.yaml ("AFP_seg", teacher of every *mam_scratch300* arm): random init.

    python -m synthesis.experts.train_expert --config configs/experts/scratch.yaml [--stop 300] [--resume] [--no-gate]
"""
import argparse
import subprocess
import sys

import torch

from synthesis.core.config import load_config
from synthesis.trainers.run import make_trainer


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--stop', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--no-gate', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg['task'] != 'segmentation':
        raise SystemExit('not an expert config')
    torch.set_num_threads(2)
    if torch.device(a.device).type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(torch.device(a.device))
    make_trainer(cfg, a.device).train(stop=a.stop, resume=a.resume, workers=a.workers)
    stop = a.stop or cfg['epochs']
    if not a.no_gate and stop in cfg['gate']['epochs']:
        # Only the final gate decides whether the expert may be used as a teacher; an intermediate failure is
        # reported (the paper continued the scratch expert after its epoch-100 gate failed).
        done = subprocess.run([sys.executable, '-m', 'synthesis.experts.gate', '--config', a.config, '--epoch', str(stop),
                               '--device', a.device])
        if done.returncode and stop == cfg['epochs']:
            raise SystemExit('final quality gate failed: do not use this expert as an AFP teacher')
        if done.returncode:
            print(f'GATE_FAILED_AT_EPOCH {stop} (intermediate)', flush=True)


if __name__ == '__main__':
    main()
