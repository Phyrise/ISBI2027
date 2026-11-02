"""Learning-rate schedules.

- ``poly_warmup`` (SGD trainer, set once per epoch): linear warm-up over ``warmup`` epochs, lr = peak (e + 1) / warmup,
  then PolyLR with exponent 0.9 to the horizon, lr = peak (1 - (e - warmup) / (horizon - warmup))^0.9.
  Synthesis arms: peak 1e-2, warm-up 5, horizon 300. Experts and M2 (nnFoundationCNN fine-tuning recipe of
  nnU-Net's PretrainedTrainer_300ep): peak 1e-3, warm-up 15, horizon 300.
- ``constant_then_linear`` (Adam / GAN trainer, set before every update): constant, then linear to 0 at the horizon.
  SwinUNETR: 1e-3, constant for 167 epochs, 0 at 250. GAN: 2e-4, constant for 100 epochs, 0 at 200.
"""


def poly_warmup(epoch, peak, warmup, horizon, exponent=.9):
    if epoch < warmup:
        return peak * (epoch + 1) / warmup
    return peak * max(0., 1 - (epoch - warmup) / (horizon - warmup)) ** exponent


def constant_then_linear(update, lr, constant_epochs, horizon, steps_per_epoch):
    boundary = constant_epochs * steps_per_epoch
    end = horizon * steps_per_epoch
    return lr * (1. if update < boundary else max(0., (end - update) / (end - boundary)))


def epoch_lr(cfg, epoch):
    s = cfg['schedule']
    if s['name'] != 'poly_warmup':
        raise ValueError(s['name'])
    return poly_warmup(epoch, s['peak_lr'], s['warmup_epochs'], s['horizon'], s.get('exponent', .9))


def update_lr(cfg, update):
    s = cfg['schedule']
    if s['name'] != 'constant_then_linear':
        raise ValueError(s['name'])
    return constant_then_linear(update, s['lr'], s['constant_epochs'], s['horizon'], cfg['train']['steps_per_epoch'])
