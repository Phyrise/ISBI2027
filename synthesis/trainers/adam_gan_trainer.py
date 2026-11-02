"""Adam / GAN trainer: SwinUNETR arms (11, 12) and the 3D pix2pixHD-style GAN arms (13, 14).

Same sampler, batches, valid masks, L1 and AFP as the ResUNet arms. Adam with the learning rate set before every update
(synthesis.trainers.schedules.constant_then_linear). Precision: the generator forward runs under bfloat16 autocast and its output
is cast to fp32 (``precision.generator_autocast``); the losses, the AFP teacher and the discriminator run under float16
autocast (``precision.train_autocast``).

GAN update (pix2pixHD order): one discriminator loss 0.5 (LSGAN(D(x, y), real) + LSGAN(D(x, safe.detach()), fake)) is
back-propagated into D first; D is then frozen and the generator loss
    pixel_weight (L1 + lambda AFP) + adversarial_weight LSGAN(D(x, safe), real) + fm_weight FM(D(x, safe), D(x, y))
is back-propagated into G; then one G update and one D update. The generated volume enters D as
``where(valid, prediction, target)``, so generated and real padding are identical. No gradient clipping for the GAN;
clip 12 for SwinUNETR.

Usage: python -m synthesis.trainers.run --config configs/arms/<arm>.yaml [--stop N] [--resume]
"""
import json
import time

import numpy as np
import torch

from synthesis.core.amp import autocast, grad_scaler
from synthesis.core.config import config_sha256, path, torch_dtype
from synthesis.core.io import save_json
from synthesis.data.sampler import loader, split_cases
from synthesis.losses.adversarial import adversarial, feature_matching
from synthesis.losses.pixel import masked_l1
from synthesis.losses.teachers import build_teacher, needs_step
from synthesis.trainers import checkpoint as CK
from synthesis.trainers.build import build_discriminator, build_model
from synthesis.trainers.schedules import update_lr


class AdamGANTrainer:
    def __init__(self, cfg, device='cuda'):
        self.cfg, self.device = cfg, torch.device(device)
        self.run_name = cfg['run_name']
        self.dtype = torch_dtype(cfg['precision']['train_autocast'])
        self.gen_dtype = torch_dtype(cfg['precision']['generator_autocast'])
        self.loss_cfg = cfg['loss']

    def prepare(self):
        cfg = self.cfg
        model, self.record = build_model(cfg)
        self.disc = None
        if cfg.get('discriminator'):
            disc, self.record['discriminator'] = build_discriminator(cfg)
            self.disc = disc.to(self.device).train()
        self.model = model.to(self.device).train()
        self.teacher = build_teacher(cfg.get('teacher'), self.device)
        if needs_step(cfg.get('teacher')):
            raise ValueError('slice teachers are not used with this trainer')
        self.weight = float(self.loss_cfg['lambda']) if self.teacher is not None else 0.
        o = cfg['optimizer']
        if o['name'] != 'adam':
            raise ValueError('Adam/GAN trainer needs optimizer.name = adam')
        self.opts = [torch.optim.Adam(self.model.parameters(), lr=update_lr(cfg, 0), betas=tuple(o['betas']),
                                      weight_decay=o['weight_decay'])]
        if self.disc is not None:
            self.opts.append(torch.optim.Adam(self.disc.parameters(), lr=update_lr(cfg, 0), betas=tuple(o['betas']),
                                              weight_decay=o['weight_decay']))
        self.scaler = grad_scaler(self.device, o['grad_scaler_init'])
        self.clip = o.get('clip')
        return self

    def generate(self, source):
        with autocast(self.device, self.gen_dtype):
            return self.model(source).float()

    def pixel_terms(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        valid = batch['valid'].to(self.device, non_blocking=True)
        source, target = data[:, 0:1], data[:, 1:2]
        with autocast(self.device, self.dtype):
            out = self.generate(source)
            l1 = masked_l1(out, target, valid)
            safe = torch.where(valid.bool(), out, target.detach())
            d = self.teacher(safe, target) if self.teacher is not None else l1 * 0
        return source, target, safe, l1, d

    def validation_loss(self, batch):
        _, _, _, l1, d = self.pixel_terms(batch)
        return float(l1 + self.weight * d)  # GAN: pixel + AFP terms only, no adversarial term

    def step(self, batch, update):
        lr = update_lr(self.cfg, update)
        for o in self.opts:
            for group in o.param_groups:
                group['lr'] = lr
            o.zero_grad(set_to_none=True)
        source, target, safe, l1, d = self.pixel_terms(batch)
        details = {'l1': float(l1.detach()), 'afp': float(d.detach()), 'lambda': self.weight, 'lr': lr}
        if self.disc is None:
            loss = l1 + self.weight * d
        else:
            L = self.loss_cfg
            self.disc.train().requires_grad_(True)
            with autocast(self.device, self.dtype):
                dloss = .5 * (adversarial(self.disc(source, target), True)
                              + adversarial(self.disc(source, safe.detach()), False))
                if not torch.isfinite(dloss):
                    raise RuntimeError('nonfinite D loss')
                self.scaler.scale(dloss).backward()
                self.disc.requires_grad_(False)
                fake = self.disc(source, safe)
                with torch.no_grad():
                    real = self.disc(source, target)
                adv = adversarial(fake, True)
                fm = feature_matching(fake, real, self.disc.layers)
                loss = L['pixel_weight'] * (l1 + self.weight * d) + L['adversarial_weight'] * adv + L['fm_weight'] * fm
            details.update(adversarial_D=float(dloss.detach()), adversarial_G=float(adv.detach()),
                           feature_matching=float(fm.detach()))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        details['loss'] = float(loss.detach())
        self.scaler.scale(loss).backward()
        for o in self.opts:
            self.scaler.unscale_(o)
        clip = self.clip
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip if clip else float('inf'))
        if self.disc is not None:
            nd = float(torch.nn.utils.clip_grad_norm_(self.disc.parameters(), float('inf')))
            details['grad_norm_D'] = nd if np.isfinite(nd) else None
        scale = self.scaler.get_scale()
        for o in self.opts:
            self.scaler.step(o)
        self.scaler.update()
        n = float(norm)
        if self.disc is not None:
            self.disc.requires_grad_(True)
        details.update(grad_norm=n if np.isfinite(n) else None, amp_scale=self.scaler.get_scale(),
                       amp_skipped=bool(self.scaler.get_scale() < scale),
                       clipped=bool(clip and np.isfinite(n) and n > clip))
        return details

    def payload(self, epoch, sha):
        return CK.payload(self.model, self.opts[0], self.scaler, epoch, self.run_name, self.record, sha,
                          self.disc, self.opts[1] if self.disc is not None else None)

    def train(self, stop=None, resume=False, workers=2):
        cfg = self.cfg
        t = cfg['train']
        horizon, stop = int(cfg['schedule']['horizon']), int(stop or cfg['epochs'])
        out = path(cfg['paths']['runs_dir'], 'paths.runs_dir') / self.run_name
        out.mkdir(parents=True, exist_ok=True)
        latest = out / 'checkpoint_latest.pth'
        if (out / f'checkpoint_ep{stop:03d}.pth').exists():
            print('Requested slice already complete; preserved.')
            return
        if latest.exists() and not resume:
            raise RuntimeError('existing training requires --resume')
        self.prepare()
        sha, epoch0 = config_sha256(cfg), 0
        if resume:
            ck = torch.load(latest, map_location='cpu', weights_only=False)
            if ck.get('kind') != self.run_name:
                raise RuntimeError(f'checkpoint belongs to run {ck.get("kind")!r}')
            if ck.get('config_sha256') not in (None, sha):
                raise RuntimeError('checkpoint was written with another config')
            epoch0 = CK.restore(ck, self.model, self.opts[0], self.scaler, self.device, self.disc,
                                self.opts[1] if self.disc is not None else None, keep_step_on_cpu=True)
            del ck
        if not epoch0 < stop <= horizon:
            raise ValueError(f'need {epoch0} < stop={stop} <= horizon={horizon}')
        save_json(out / 'initialization_record.json', dict(self.record, run=self.run_name, config_sha256=sha,
                                                           teacher=getattr(self.teacher, 'provenance', None)))
        iterations, val_n, seed = int(t['steps_per_epoch']), int(t['val_batches']), int(cfg['seed'])
        data_dir = path(cfg['paths']['data_dir'], 'paths.data_dir')
        split = path(cfg['paths']['split'], 'paths.split')
        train_ids, dev_ids = split_cases(split, 'train'), split_cases(split, 'dev')
        batches = iter(loader(data_dir, train_ids, seed, epoch0 * iterations, stop * iterations, workers=workers,
                              pin_memory=self.device.type == 'cuda'))
        if epoch0 == 0:
            CK.save_initial(out, self.payload(0, sha))
        planfile = out / 'batch_plan.jsonl'
        if resume:
            CK.truncate_plan(planfile, epoch0 * iterations)
        snapshots = set(t['snapshots'])
        logs = (out / 'epoch_metrics.jsonl').open('a', buffering=1)
        with planfile.open('a', buffering=1) as plans:
            for epoch in range(epoch0, stop):
                started = time.perf_counter()
                self.model.train()
                rows, wait = [], []
                for i in range(iterations):
                    t0 = time.perf_counter()
                    b = next(batches)
                    wait.append(time.perf_counter() - t0)
                    rows.append(self.step(b, epoch * iterations + i))
                    plans.write(b['plan'] + '\n')
                self.model.eval()
                vals = []
                with torch.inference_mode():
                    for b in loader(data_dir, dev_ids, seed + int(t['val_seed_offset']), epoch * val_n,
                                    (epoch + 1) * val_n, workers=0, pin_memory=False):
                        vals.append(self.validation_loss(b))
                skip = int(sum(x['amp_skipped'] for x in rows))
                summary = {'epoch': epoch + 1, 'lr': rows[0]['lr'], 'lr_first': rows[0]['lr'], 'lr_last': rows[-1]['lr'],
                           'train_loss': float(np.mean([x['loss'] for x in rows])),
                           'validation_patch_loss': float(np.mean(vals)), 'amp_skipped': skip,
                           'clipped': int(sum(x['clipped'] for x in rows)),
                           'grad_norm_median': float(np.median([x['grad_norm'] for x in rows if x['grad_norm'] is not None])),
                           'data_wait_median_s': float(np.median(wait)), 'epoch_seconds': time.perf_counter() - started,
                           'l1': float(np.mean([x['l1'] for x in rows])), 'afp': float(np.mean([x['afp'] for x in rows]))}
                if self.disc is not None:
                    for key in ('adversarial_D', 'adversarial_G', 'feature_matching'):
                        summary[key] = float(np.mean([x[key] for x in rows]))
                logs.write(json.dumps(summary) + '\n')
                print(json.dumps(summary), flush=True)
                if skip > max(2, .05 * iterations):
                    raise RuntimeError('AMP skipped >5% of the updates of an epoch; stopped for inspection')
                CK.save_epoch(out, self.payload(epoch + 1, sha), epoch + 1, snapshots, stop)
                save_json(out / 'status.json', {'state': 'trained' if epoch + 1 == stop else 'training',
                                                'epoch': epoch + 1, 'stop': stop, 'horizon': horizon, 'last': summary,
                                                'updated': time.strftime('%F %T')})
        logs.close()
        print('TRAINING_SLICE_DONE', self.run_name, stop, flush=True)
