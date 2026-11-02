"""AFP (anatomical feature-prioritised) distance with a frozen ResEnc-L teacher, encoder features only.

Teachers used in the paper (all the same ResEnc-L architecture, only the weights differ):
- nnFoundationCNN (MIC-DKFZ; nnssl MAE pre-training; CC-BY-SA-4.0 weights, not redistributed): the nnssl checkpoint;
- the two lesion experts trained here (experts/): AFP_seg-ft-nnF (nnF encoder fine-tuned for the lesion) and
  "AFP_seg" (trained from scratch).
The full network is rebuilt and loaded strictly; only its encoder runs, and the features are the six encoder stage
outputs ``encoder.0`` ... ``encoder.5`` (every AFP arm of the paper; the reconstruction / segmentation decoder never
runs).

Distance = ``nnunetv2.training.loss.afp_features.AFPFeatureDistance`` (Phyrise/nnUNet_translation, tag
``isbi2027-results``): prediction and target are z-scored separately (per sample, over D, H, W), zero-padded
symmetrically to a multiple of 2^5, and D = sum over the six stages of mean |phi_s(prediction) - phi_s(target)|, with the
target features computed without gradient.

Precision: the teacher weights are cast to ``dtype`` (paper: float16) and features are compared in fp32. Callers pass
``torch.where(valid, prediction, target)`` so that the AFP gradient is zero outside the valid volume.

Port of men_clinical/foundation_afp.py (FoundationFeatureExpert, foundation_expert, FoundationAFP) and
men_clinical/expert_afp.py (ExpertAFP interface): the paper teachers set mae_weight = 0 and afp_weight = 1, so the
forward here returns the distance only (the zero-weighted L1 term of the original adds exactly 0 to the value and
the gradient; see tests/test_train_afp.py).
"""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from synthesis.architectures.resenc_l import RESENC_L, build_resenc_l
from synthesis.core.io import sha256

SELECTIONS = {'encoder': tuple(f'encoder.{i}' for i in range(6)),
              'encoder_historical': tuple(f'encoder.{i}' for i in range(4))}


def _feature_expert_class():
    from nnunetv2.training.loss.afp_features import ExpertOutput, NNUNetFeatureExpert

    class FoundationFeatureExpert(NNUNetFeatureExpert):
        """NNUNetFeatureExpert whose features are the encoder stage outputs only (the decoder is never run)."""

        def __init__(self, network, modalities):
            super().__init__(network, modalities)
            self.feature_names = self.encoder_names

        def select_features(self, selection='encoder'):
            if isinstance(selection, str):
                if selection not in SELECTIONS:
                    raise ValueError(f'foundation features: {sorted(SELECTIONS)} or explicit encoder names')
                return SELECTIONS[selection]
            names = tuple(selection)
            if len(names) != len(set(names)) or any(n not in self.encoder_names for n in names):
                raise ValueError(f'Invalid/duplicate encoder feature names {names}')
            return names

        def forward(self, x):
            if x.ndim != 5 or x.shape[1] != 1:
                raise ValueError('Expected B x 1 x D x H x W input')
            if any(s % d for s, d in zip(x.shape[2:], self.required_divisibility)):
                raise ValueError(f'Input must be padded to multiples of {self.required_divisibility}')
            param = next(self.network.parameters())
            self.network.eval()
            skips = self.network.encoder(x.to(dtype=param.dtype))
            if len(skips) != len(self.encoder_names):
                raise RuntimeError('Incomplete encoder forward')
            return ExpertOutput(OrderedDict(zip(self.encoder_names, skips)), skips[-1])

    return FoundationFeatureExpert


def load_teacher_network(checkpoint, trusted, device, dtype):
    """Rebuild the ResEnc-L of an nnssl pre-training checkpoint (MAE head, 1 class) or of a lesion-expert checkpoint
    written by this repository (``network_weights`` with ``decoder.seg_layers.0``), and load it strictly.

    trusted=False: ``torch.load(weights_only=True)`` (downloaded files). trusted=True: full unpickling, needed for the
    expert checkpoints written by the trainers (they also hold numpy RNG states)."""
    from nnunetv2.training.loss.afp_features import load_strict_weights
    payload = torch.load(checkpoint, map_location='cpu', weights_only=not trusted)
    plan = payload.get('nnssl_adaptation_plan')
    if plan is not None:
        arch = (plan.get('architecture_plans') or {}).get('arch_class_name')
        if arch != 'ResEncL' or plan.get('pretrain_num_input_channels') != 1:
            raise ValueError(f'unsupported nnssl checkpoint: arch {arch}, inputs {plan.get("pretrain_num_input_channels")}')
        fmt, num_classes = 'nnssl pre-training (MAE head)', 1
    else:
        heads = [k for k in payload['network_weights'] if k.endswith('decoder.seg_layers.0.weight')]
        if len(heads) != 1:
            raise ValueError(f'{checkpoint}: neither an nnssl checkpoint nor a ResEnc-L segmentation checkpoint')
        num_classes = int(payload['network_weights'][heads[0]].shape[0])
        fmt = f'segmentation checkpoint (epoch {payload.get("current_epoch")})'
    network = build_resenc_l(num_classes=num_classes)
    load_strict_weights(network, payload['network_weights'])  # encoder + decoder, including the alias entries
    return network.to(device=device, dtype=dtype), {'checkpoint_format': fmt, 'num_classes': num_classes}


class EncoderAFP(nn.Module):
    """Frozen ResEnc-L encoder teacher: forward(prediction, target) -> AFP distance (scalar)."""

    def __init__(self, checkpoint, trusted=False, modality='first_postcontrast', features='encoder',
                 normalize_features=False, dtype='float16', device='cuda'):
        super().__init__()
        from nnunetv2.training.loss.afp_features import AFPFeatureDistance
        network, info = load_teacher_network(checkpoint, trusted, device, getattr(torch, dtype))
        expert = _feature_expert_class()(network, [modality])
        self.distance = AFPFeatureDistance(expert, features=features, normalize_features=normalize_features,
                                           target_modality=modality)
        self.model = expert.network  # frozen, eval
        self.provenance = dict(info, checkpoint=Path(checkpoint).name, checkpoint_sha256=sha256(checkpoint),
                               architecture='ResEnc-L (dynamic_network_architectures ResidualEncoderUNet)',
                               arch_kwargs=RESENC_L, features=features, feature_names=list(self.distance.selected),
                               normalize_features=normalize_features, dtype=dtype, encoder_only=True,
                               distance='AFP L1, separate z-score, padding to 2^5')

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, prediction, target):
        return self.distance(prediction, target)
