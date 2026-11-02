"""Initialisations: seeded common random inits, the nnFoundationCNN encoder copy, and the scratch-expert record.

Common random inits (drawn on CPU right after ``seed_everything(seed)``; the paper files are bit-identical to a redraw
with torch 2.8.0 / MONAI 1.5.1 / dynamic-network-architectures ``isbi2027-results``, see tests/test_train_inits.py):
- ResUNet generator: seed 20261018 (paper file ``generator_init.pth``), shared by every ResUNet arm and the GAN generator;
- SwinUNETR: seed 20261018 (``generator_init_swin_r11.pth``);
- PatchGAN discriminator: seed 20261022 (``discriminator_init_r11.pth``);
- ResEnc-L experts: seed 20261019, drawn inside the trainer (both experts share the same random decoder).

"<- nnF": the encoder tensors of the nnFoundationCNN checkpoint (``network_weights``, keys ``encoder.*``) are copied
exactly over the common init; the decoder keeps its init tensors. The ResUNet's ``decoder.encoder.*`` entries are
aliases of the same modules, so they change with the encoder.

Stride caveat: the generator's last encoder stride is (1, 2, 2), while nnFoundationCNN's ResEnc-L uses (2, 2, 2) at
every stage after the first. Stride does not change a convolution's weight shape, so every encoder tensor matches in
shape and the copy is exact; but the deepest generator stage then runs the pretrained filters on a bottleneck that is
twice as deep (in D) as the one they were trained on. The copy is a weight transfer, not the pretrained network.

Usage (regenerate the common init files):
    python -m synthesis.architectures.init --what resunet --out generator_init.pth
    python -m synthesis.architectures.init --what swinunetr --out generator_init_swin.pth
    python -m synthesis.architectures.init --what discriminator --out discriminator_init.pth
"""
import argparse
import warnings

import torch

from synthesis.core.io import atomic_torch_save, sha256, state_sha256, seed_everything

SEEDS = {'resunet': 20261018, 'swinunetr': 20261018, 'discriminator': 20261022, 'expert': 20261019}


def seeded(build, seed):
    seed_everything(int(seed))
    return build()


def load_network_weights(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)['network_weights']
    model.load_state_dict(state, strict=True)
    return {'file': str(path), 'sha256': sha256(path)}


def copy_nnf_encoder(model, checkpoint, reference_sha256=None):
    """Exact load of the nnFoundationCNN encoder into ``model`` (ResUNet generator or ResEnc-L expert).

    Strict checks: the checkpoint's ``encoder.*`` keys equal the model's, nothing unexpected, only ``decoder.*`` keys
    missing, every copied tensor equal after the load, every non-encoder tensor unchanged."""
    digest = sha256(checkpoint)
    if reference_sha256 and digest != reference_sha256:
        warnings.warn(f'nnFoundationCNN checkpoint sha256 {digest} differs from the paper file {reference_sha256}')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)['network_weights']
    enc = {k: v for k, v in weights.items() if k.startswith('encoder.')}
    before = {k: v.clone() for k, v in model.state_dict().items()}
    keys = {k for k in before if k.startswith('encoder.')}
    if set(enc) != keys:
        raise ValueError(f'encoder keys differ: {sorted(set(enc) ^ keys)[:5]}')
    out = model.load_state_dict(enc, strict=False)
    if out.unexpected_keys or not all(k.startswith('decoder.') for k in out.missing_keys):
        raise ValueError(f'unexpected {out.unexpected_keys[:5]} / missing {out.missing_keys[:5]}')
    after = model.state_dict()
    if not all(torch.equal(after[k], v) for k, v in enc.items()):
        raise ValueError('encoder copy is not exact')
    rest = [k for k in after if not k.startswith(('encoder.', 'decoder.encoder.'))]
    if not rest or not all(torch.equal(after[k], before[k]) for k in rest):
        raise ValueError('non-encoder tensors changed')
    return {'checkpoint': str(checkpoint), 'sha256': digest, 'encoder_tensors': len(enc),
            'unchanged_other_tensors': len(rest), 'exact': True}


def scratch_record(model, seed):
    """Random init, nothing loaded: record the seed and the digest of the initial state."""
    return {'initialization': f'random, drawn after seed_everything({seed}); no pretrained weights',
            'seed': int(seed), 'initial_state_sha256': state_sha256(model.state_dict())}


def build(what, cfg_arch=None):
    if what == 'resunet':
        from .resunet import build_resunet
        return build_resunet(*([cfg_arch] if cfg_arch else []))
    if what == 'swinunetr':
        from .swinunetr import build_swinunetr
        return build_swinunetr(**(cfg_arch or {}))
    if what == 'discriminator':
        from .patchgan3d import build_discriminator
        return build_discriminator(**(cfg_arch or {}))
    if what == 'resenc_l_seg':
        from .resenc_l import build_resenc_l
        return build_resenc_l(input_channels=1, num_classes=2)
    raise ValueError(what)


def main():
    p = argparse.ArgumentParser(description='Draw a common random init (CPU) and save it as {"network_weights": ...}.')
    p.add_argument('--what', choices=['resunet', 'swinunetr', 'discriminator'], required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    s = a.seed if a.seed is not None else SEEDS[a.what]
    net = seeded(lambda: build(a.what), s)
    atomic_torch_save(a.out, {'network_weights': net.state_dict(), 'seed': s, 'torch': torch.__version__})
    print({'out': a.out, 'seed': s, 'state_sha256': state_sha256(net.state_dict())})


if __name__ == '__main__':
    main()
