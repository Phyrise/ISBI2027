"""Train one arm or expert from its config.

    python -m synthesis.trainers.run --config configs/arms/l1.yaml [--stop 100] [--resume] [--device cuda]

``--stop`` trains up to that epoch (the paper ran the ResUNet arms in slices 0-100, 100-200, 200-300 with --resume);
without it the run goes to ``epochs``. The trainer is chosen by the config's ``trainer`` field (sgd or adam_gan).
"""
import argparse

import torch

from synthesis.core.config import load_config


def make_trainer(cfg, device):
    if cfg['trainer'] == 'sgd':
        from synthesis.trainers.sgd_trainer import SGDTrainer
        return SGDTrainer(cfg, device)
    if cfg['trainer'] == 'adam_gan':
        from synthesis.trainers.adam_gan_trainer import AdamGANTrainer
        return AdamGANTrainer(cfg, device)
    raise ValueError(cfg['trainer'])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--stop', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    cfg = load_config(a.config)
    torch.set_num_threads(a.threads)
    if torch.device(a.device).type == 'cuda':
        torch.backends.cudnn.benchmark = True  # as in the paper runs (not bit-reproducible)
        torch.cuda.set_device(torch.device(a.device))
    make_trainer(cfg, a.device).train(stop=a.stop, resume=a.resume, workers=a.workers)


if __name__ == '__main__':
    main()
