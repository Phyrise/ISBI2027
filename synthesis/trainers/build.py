"""Build the initial networks of an arm or expert from its config (CPU; the trainer moves them to the device).

Config blocks used:
  architecture: {name: resunet | swinunetr | resenc_l_seg, ...}
  init:
    seed: seed drawn before building the network (common random init)
    file: optional {"network_weights": ...} file of that common init (paper: generator_init.pth, ...); loaded strictly;
          a value whose ${VAR} is unset counts as absent (redraw from the seed)
    nnf_encoder: true -> copy the nnFoundationCNN encoder over the init (synthesis.architectures.init.copy_nnf_encoder)
  discriminator: {width, scales, layers, init: {seed, file}}   (GAN arms only)
  paths.nnf_checkpoint: the nnFoundationCNN checkpoint (needed when init.nnf_encoder is true)

The paper trainer seeded with the run seed, built the network, then loaded the common-init file; a redraw from the
init seed gives the same tensors (tests/test_train_inits.py), so ``init.file`` is optional.
"""
import torch

from synthesis.architectures.init import build, copy_nnf_encoder, load_network_weights, scratch_record
from synthesis.core.config import is_set, path
from synthesis.core.io import seed_everything, state_sha256

NNF_REFERENCE_SHA256 = 'ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd'


def _arch_kwargs(cfg):
    a = dict(cfg['architecture'])
    name = a.pop('name')
    config = a.pop('config', None)
    if name == 'resunet':
        return name, config
    return name, a or None


def build_model(cfg):
    """Returns (model, init record)."""
    name, kw = _arch_kwargs(cfg)
    init = cfg['init']
    seed_everything(int(cfg['seed']))
    model = build(name, kw)
    record = {'architecture': name, 'run_seed': int(cfg['seed'])}
    if init.get('seed') is not None and int(init['seed']) != int(cfg['seed']):
        seed_everything(int(init['seed']))
        model = build(name, kw)
    if is_set(init.get('file')):
        record['init_file'] = load_network_weights(model, path(init['file'], 'init.file'))
    elif cfg['task'] == 'segmentation' and not init.get('nnf_encoder'):
        record['init'] = scratch_record(model, init.get('seed', cfg['seed']))
    record['initial_state_sha256'] = state_sha256(model.state_dict())
    if init.get('nnf_encoder'):
        record['nnf_encoder'] = copy_nnf_encoder(model, path(cfg['paths']['nnf_checkpoint'], 'paths.nnf_checkpoint'),
                                                 reference_sha256=NNF_REFERENCE_SHA256)
    return model, record


def build_discriminator(cfg):
    from synthesis.architectures.patchgan3d import build_discriminator as make
    d = dict(cfg['discriminator'])
    init = d.pop('init')
    seed_everything(int(init['seed']))
    disc = make(source_channels=1, width=d['width'], scales=d['scales'], layers=d['layers'])
    record = {'seed': int(init['seed'])}
    if is_set(init.get('file')):
        record['init_file'] = load_network_weights(disc, path(init['file'], 'discriminator.init.file'))
    record['initial_state_sha256'] = state_sha256(disc.state_dict())
    return disc, record


def load_for_inference(cfg, checkpoint):
    """Network of an arm / expert with the weights of a checkpoint (paper or this repository's format)."""
    name, kw = _arch_kwargs(cfg)
    model = build(name, kw)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['network_weights']
    model.load_state_dict(state, strict=True)
    return model
