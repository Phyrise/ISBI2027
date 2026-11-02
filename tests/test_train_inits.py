"""Initialisations: seeded redraws vs the paper init files, the nnF-encoder init vs the original init_encoder, and the
initial state of every arm and expert vs the epoch-0 checkpoint of its paper run."""
import gc
import unittest

import torch

from synthesis.architectures.init import SEEDS, build, seeded
from synthesis.core.io import state_sha256
from tests.train_support import ALL_ARMS, arm_config, max_abs, need, original, report, root, slow, strict_fp32
from synthesis.trainers.build import build_discriminator, build_model

R11 = 'mama_3d_20261018_r11_arch'


def weights(path, key='network_weights'):
    return torch.load(path, map_location='cpu', weights_only=False)[key]


@need('ISBI_ORIG_ROOT', 'ISBI_ORIG_PROJECTS', 'ISBI_ORIG_MX', 'ISBI_NNF_CKPT')
class TestInits(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    def test_seeded_inits_vs_paper_files(self):
        for what, f in (('resunet', 'generator_init.pth'), ('swinunetr', 'generator_init_swin_r11.pth'),
                        ('discriminator', 'discriminator_init_r11.pth')):
            net = seeded(lambda: build(what), SEEDS[what])
            ours, paper = net.state_dict(), weights(root() / f)
            gap = max_abs(ours, paper)
            report('inits', f'seeded {what} (seed {SEEDS[what]}) vs {f}', gap, tensors=len(paper),
                   state_sha256_equal=state_sha256(ours) == state_sha256(paper))
            self.assertEqual(gap, 0.)
            del net, ours, paper
            gc.collect()

    def test_nnf_init_vs_original_init_encoder(self):
        ours, rec = build_model(arm_config('l1_nnf'))
        ours = ours.state_dict()
        with original(R11, 'common') as C:
            C.seed(20261018)
            g = C.generator()
            g.load_state_dict(weights(C.ROOT / 'generator_init.pth'), strict=True)
            C.init_encoder(g)
            theirs = g.state_dict()
        gap = max_abs(ours, theirs)
        report('inits', 'ResUNet <- nnF encoder vs original init_encoder', gap, tensors=len(theirs),
               encoder_tensors=rec['nnf_encoder']['encoder_tensors'])
        self.assertEqual(gap, 0.)
        del ours, theirs, g
        gc.collect()
        ours, _ = build_model(arm_config('nnf_ft', 'experts'))
        ours = ours.state_dict()
        with original(R11, 'common') as C:
            C.seed(20261019)
            m = C.segmentation_model()
            C.init_encoder(m)
            theirs = m.state_dict()
        gap = max_abs(ours, theirs)
        report('inits', 'expert ResEnc-L <- nnF encoder vs original init_encoder', gap, tensors=len(theirs))
        self.assertEqual(gap, 0.)

    @slow()
    def test_initial_state_vs_epoch0_checkpoints(self):
        runs = [(a, a, 'arms') for a in ALL_ARMS] + [('nnf_ft', 'expert', 'experts'),
                                                     ('scratch', 'expert_scratch', 'experts')]
        for name, run, kind in runs:
            cfg = arm_config(name, kind)
            model, _ = build_model(cfg)
            ck = torch.load(root() / 'results' / run / 'checkpoint_ep000.pth', map_location='cpu', weights_only=False)
            self.assertEqual(int(ck['current_epoch']), 0)
            gap = max_abs(model.state_dict(), ck['network_weights'])
            if cfg.get('discriminator'):
                disc, _ = build_discriminator(cfg)
                gap = max(gap, max_abs(disc.state_dict(), ck['discriminator_weights']))
            report('inits', f'initial state {name} vs results/{run}/checkpoint_ep000.pth', gap,
                   tensors=len(ck['network_weights']) + len(ck.get('discriminator_weights', {})))
            self.assertEqual(gap, 0.)
            del model, ck
            gc.collect()
