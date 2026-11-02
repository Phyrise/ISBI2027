"""SGD trainer: ResUNet synthesis arms 1-10 and the two lesion experts.

Paired, resumable training with nnU-Net architectures (and, for the experts, the nnU-Net Dice + CE loss) on this
repository's stateless sampler. This is not stock nnU-Net: no nnU-Net planning, data augmentation or trainer.

One update (``step``): zero grads; forward under autocast (``precision.train_autocast``: float16 for every arm except
MedicalNet, which uses bfloat16); loss = L1 + lambda * D for synthesis (D = 0 without a teacher; the teacher sees
``where(valid, prediction, target)`` so its gradient is zero outside the volume), Dice + CE for the experts; scaled
backward; unscale; clip the gradient norm at 12; scaler step and update. SGD with Nesterov momentum 0.99 and weight
decay 3e-5; the learning rate is set once per epoch (synthesis.trainers.schedules.poly_warmup). 250 updates per epoch, then 20
validation batches (dev split, seed + 999), then ``checkpoint_latest.pth`` and the snapshots.

Usage: python -m synthesis.trainers.run --config configs/arms/<arm>.yaml [--stop N] [--resume]
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from synthesis.core.amp import autocast, grad_scaler
from synthesis.core.config import config_sha256, path, torch_dtype
from synthesis.core.io import save_json
from synthesis.data.sampler import loader, split_cases, zscore
from synthesis.losses.pixel import masked_l1
from synthesis.losses.segmentation import segmentation_loss
from synthesis.losses.teachers import build_teacher, needs_step
from synthesis.trainers import checkpoint as CK
from synthesis.trainers.build import build_model
from synthesis.trainers.schedules import epoch_lr


class SGDTrainer:
    def __init__(self, cfg, device='cuda'):
        self.cfg, self.device = cfg, torch.device(device)
        self.task = cfg['task']
        self.run_name = cfg['run_name']
        self.dtype = torch_dtype(cfg['precision']['train_autocast'])
        self.clip = float(cfg['optimizer']['clip'])

    # ---- setup -------------------------------------------------------------------------------------------------
    def prepare(self):
        cfg = self.cfg
        model, self.record = build_model(cfg)
        self.model = model.to(self.device).train()
        self.teacher = None if self.task == 'segmentation' else build_teacher(cfg.get('teacher'), self.device)
        self.slice_step = needs_step(cfg.get('teacher'))
        self.lossfn = segmentation_loss() if self.task == 'segmentation' else None
        self.weight = float(cfg['loss']['lambda']) if self.teacher is not None else 1.
        o = cfg['optimizer']
        if o['name'] != 'sgd':
            raise ValueError('SGD trainer needs optimizer.name = sgd')
        self.opt = torch.optim.SGD(self.model.parameters(), lr=epoch_lr(cfg, 0), momentum=o['momentum'],
                                   nesterov=o['nesterov'], weight_decay=o['weight_decay'])
        self.scaler = grad_scaler(self.device, o['grad_scaler_init'])
        return self

    # ---- one update ---------------------------------------------------------------------------------------------
    def forward(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        valid = batch['valid'].to(self.device, non_blocking=True)
        target = data[:, 1:2]
        with autocast(self.device, self.dtype):
            if self.task == 'segmentation':
                out = self.model(zscore(target))
                loss = self.lossfn(out, batch['label'].to(self.device, non_blocking=True))
                details = {'loss': float(loss.detach())}
            else:
                out = self.model(data[:, 0:1])
                l1 = masked_l1(out, target, valid)
                safe = torch.where(valid.bool(), out.float(), target.detach())
                if self.slice_step:
                    d = self.teacher(safe, target, valid=valid, step=json.loads(batch['plan'])['step'])
                else:
                    d = self.teacher(safe, target) if self.teacher is not None else l1 * 0
                loss = l1 + self.weight * d
                details = {'loss': float(loss.detach()), 'l1': float(l1.detach()), 'afp': float(d.detach()),
                           'lambda': self.weight}
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        return loss, details

    def step(self, batch):
        opt, scaler = self.opt, self.scaler
        opt.zero_grad(set_to_none=True)
        loss, details = self.forward(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        n = float(norm)
        details.update(grad_norm=n if np.isfinite(n) else None, amp_scale=scaler.get_scale(),
                       amp_skipped=bool(scaler.get_scale() < scale), clipped=bool(np.isfinite(n) and n > self.clip))
        return details

    # ---- training loop ------------------------------------------------------------------------------------------
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
            if ck.get('kind') not in (self.run_name, cfg.get('resume_from_run')):
                raise RuntimeError(f'checkpoint belongs to run {ck.get("kind")!r}')
            if ck.get('config_sha256') not in (None, sha):
                raise RuntimeError('checkpoint was written with another config')
            epoch0 = CK.restore(ck, self.model, self.opt, self.scaler, self.device)
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
        snapshots = set(t['snapshots'])
        if epoch0 == 0:
            CK.save_initial(out, self.payload(0, sha))
        planfile = out / 'batch_plan.jsonl'
        if resume:
            CK.truncate_plan(planfile, epoch0 * iterations)
        logs = (out / 'epoch_metrics.jsonl').open('a', buffering=1)
        with planfile.open('a', buffering=1) as plans:
            for epoch in range(epoch0, stop):
                started = time.perf_counter()
                self.model.train()
                rows, wait = [], []
                for group in self.opt.param_groups:
                    group['lr'] = epoch_lr(cfg, epoch)
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    b = next(batches)
                    wait.append(time.perf_counter() - t0)
                    rows.append(self.step(b))
                    plans.write(b['plan'] + '\n')
                self.model.eval()
                vals = []
                with torch.inference_mode():
                    for b in loader(data_dir, dev_ids, seed + int(t['val_seed_offset']), epoch * val_n,
                                    (epoch + 1) * val_n, workers=0, pin_memory=False):
                        vals.append(self.forward(b)[1]['loss'])
                skip = int(sum(r['amp_skipped'] for r in rows))
                summary = {'epoch': epoch + 1, 'lr': epoch_lr(cfg, epoch),
                           'train_loss': float(np.mean([r['loss'] for r in rows])),
                           'validation_patch_loss': float(np.mean(vals)), 'amp_skipped': skip,
                           'clipped': int(sum(r['clipped'] for r in rows)),
                           'grad_norm_median': float(np.median([r['grad_norm'] for r in rows if r['grad_norm'] is not None])),
                           'data_wait_median_s': float(np.median(wait)), 'epoch_seconds': time.perf_counter() - started,
                           'l1': float(np.mean([r.get('l1', 0.) for r in rows])),
                           'afp': float(np.mean([r.get('afp', 0.) for r in rows]))}
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

    def payload(self, epoch, sha):
        return CK.payload(self.model, self.opt, self.scaler, epoch, self.run_name, self.record, sha)
