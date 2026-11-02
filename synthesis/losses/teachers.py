"""Build the frozen feature teacher of an arm from its ``teacher`` config block.

kind      teacher                                         call
nnf       nnFoundationCNN ResEnc-L encoder (synthesis.losses.afp)    afp(safe, target)
expert    a lesion expert's ResEnc-L encoder (synthesis.losses.afp)  afp(safe, target)
medicalnet MedicalNet ResNet-50 layer4 (synthesis.losses.medicalnet) afp(safe, target)
sam, vgg  2D slice teacher (synthesis.losses.slice2d)               afp(safe, target, valid=valid, step=batch step)
"""
import warnings

from synthesis.core.config import path

SLICE_KINDS = ('sam', 'vgg')


def build_teacher(tcfg, device):
    """Returns the frozen teacher module, or None when the arm has no feature term."""
    if not tcfg or tcfg.get('kind') in (None, 'none'):
        return None
    kind = tcfg['kind']
    ckpt = path(tcfg.get('checkpoint'), f'teacher.checkpoint ({kind})')
    if kind in ('nnf', 'expert'):
        from .afp import EncoderAFP
        net = EncoderAFP(ckpt, trusted=kind == 'expert', modality='first_postcontrast',
                         features=tcfg.get('features', 'encoder'), dtype=tcfg.get('dtype', 'float16'), device=device)
    elif kind == 'medicalnet':
        from .medicalnet import MedicalNetTeacher
        net = MedicalNetTeacher(checkpoint=ckpt, device=device)
    elif kind in SLICE_KINDS:
        from .slice2d import SliceTeacher
        net = SliceTeacher(kind, ckpt, device=device)
    else:
        raise ValueError(f'unknown teacher kind {kind!r}')
    if any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('teacher parameters must be frozen')
    ref, got = tcfg.get('reference_sha256'), net.provenance.get('checkpoint_sha256')
    if ref and got != ref:
        warnings.warn(f'teacher checkpoint {ckpt} (sha256 {got}) differs from the paper file (sha256 {ref})')
    return net


def needs_step(tcfg):
    return bool(tcfg) and tcfg.get('kind') in SLICE_KINDS
