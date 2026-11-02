"""nnFoundationViT 3D token distance (perceptual family; vit_body, vit_lesion, vit_healthy; lower = closer).

Reader: the frozen encoder (patch embedding + 40 EVA blocks + final norm) of nnFoundationViT (MIC-DKFZ, Primus
architecture, embed 1056, 16 heads, patch 8^3, OpenMind brain-MRI pre-training; not validated on breast MRI). On brain
meningioma synthesis it passed identity and monotonicity checks but failed the lesion-specificity check (it weighs
texture, blur and noise more than lesion errors).
Per patient:
  grid: real, synthetic, body and lesion resampled from 1 x 0.7 x 0.7 mm to 1 mm isotropic (trilinear,
        align_corners False; masks interpolated then >= 0.5);
  outside the body both volumes take the background value of the real image (median outside the body);
  tiles: the body bounding box + 4 voxels split into n = ceil(L / 192) equal parts per axis (no overlap), each centred
        in a 192^3 canvas filled with the background; both canvases normalised with the mean / std of the REAL canvas;
  distance per token: 1 - cos(f_real, f_synthetic) on the final norm output (fp16 autocast; TF32 off);
  scores: token-weighted means over all tiles, weight = fraction of the token's voxels inside the region:
        vit_body (body), vit_lesion (lesion dilated by 2 voxels, cube, inside the body), vit_healthy (body minus the
        lesion dilated by 5); a region needs a total weight >= 1 token, else None; secondary: layer-normed outputs of
        blocks 10 / 20 / 30 (body, lesion).
Identity control: the real image against itself gives < 1e-6.

usage: python -m evaluation.metrics.perceptual.nnf_vit ENTRY... --data DIR --pred-dir DIR --out DIR --vit-dir DIR [--device cuda]
"""
import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from evaluation.metrics import io

WIN = 192
MARGIN = 4
SECONDARY = (10, 20, 30)
REGIONS = ('body', 'lesion', 'healthy')


def dilate(m, k):
    from scipy import ndimage
    return ndimage.maximum_filter(m, size=2 * k + 1) if k > 0 else m


def regions(lesion, body):
    wt = lesion > 0
    return {'body': body, 'lesion': dilate(wt, 2) & body, 'healthy': body & ~dilate(wt, 5)}


def canvas(vol, body, bg, box, off, fill=None):
    """WIN^3 canvas: vol inside the body (outside -> bg), cropped to box, centred at off; `fill` for masks."""
    c = np.full((WIN,) * 3, bg if fill is None else fill, dtype=np.float32)
    v = vol[box].astype(np.float32)
    if fill is None:
        v = np.where(body[box], v, bg)
    sl = tuple(slice(o, o + v.shape[d]) for d, o in enumerate(off))
    c[sl] = v
    return c


def build_encoder(vit_dir, device):
    import torch
    from dynamic_network_architectures.architectures.primus import Primus
    vit_dir = Path(vit_dir); prov = json.loads((vit_dir / 'PROVENANCE.json').read_text())
    ck_path = vit_dir / 'checkpoint_final.pth'
    if ck_path.stat().st_size != prov['size_checkpoint_bytes']:
        raise RuntimeError('checkpoint size differs from PROVENANCE.json')
    ck = torch.load(ck_path, map_location='cpu', weights_only=True, mmap=True)
    kw = ck['nnssl_adaptation_plan']['architecture_plans']['arch_kwargs']
    if (kw['embed_dim'], kw['encoder_eva_depth'], kw['encoder_eva_numheads'], tuple(kw['patch_embed_size'])) != (1056, 40, 16, (8, 8, 8)):
        raise RuntimeError(f'unexpected architecture {kw}')
    net = Primus(input_channels=1, embed_dim=1056, patch_embed_size=(8, 8, 8), num_classes=1, eva_depth=40,
                 eva_numheads=16, input_shape=(WIN,) * 3, drop_path_rate=0.0, patch_drop_rate=0.0, attn_drop_rate=0.0,
                 init_values=kw['init_values'], scale_attn_inner=kw['scale_attn_inner'])
    enc = {k: v for k, v in ck['network_weights'].items() if k.startswith(('down_projection.', 'eva.'))}
    own = {k for k in net.state_dict() if k.startswith(('down_projection.', 'eva.'))}
    if own != set(enc):
        raise RuntimeError(f'encoder keys differ: {sorted(own ^ set(enc))[:5]}')
    res = net.load_state_dict(enc, strict=False)
    if [k for k in res.missing_keys if k.startswith(('down_projection.', 'eva.'))] or res.unexpected_keys:
        raise RuntimeError(f'load: {res}')
    net.up_projection = None
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    info = {'revision': prov['revision'], 'sha256': prov['sha256']['checkpoint_final.pth'], 'tensors': len(enc),
            'window': [WIN] * 3, 'tokens': (WIN // 8) ** 3}
    del ck
    return net, info


def features(net, x):
    """{'norm': final norm tokens [N, C], 'blockK': parameter-free layer-normed block outputs} for x [1, 1, 192, 192, 192]."""
    import torch.nn.functional as F
    from einops import rearrange
    t = net.down_projection(x)
    t = rearrange(t, 'b c w h d -> b (w h d) c')
    t, rope, keep = net.eva._pos_embed(t)
    assert keep is None
    out = {}
    for i, blk in enumerate(net.eva.blocks, 1):
        t = blk(t, rope=rope)
        if i in SECONDARY:
            out[f'block{i}'] = F.layer_norm(t.float(), t.shape[-1:])[0]
    out['norm'] = net.eva.norm(t).float()[0]
    return out


def token_weights(region_canvases, device):
    import torch
    import torch.nn.functional as F
    return {k: F.avg_pool3d(torch.from_numpy(m.astype(np.float32)).to(device)[None, None], 8, 8).flatten()
            for k, m in region_canvases.items()}   # same token order as rearrange 'b c w h d -> b (w h d) c'


def tiles(b, win=WIN, margin=MARGIN):
    idx = np.nonzero(b); axes = []
    for d in range(3):
        lo, hi = max(int(idx[d].min()) - margin, 0), min(int(idx[d].max()) + 1 + margin, b.shape[d])
        e = np.linspace(lo, hi, math.ceil((hi - lo) / win) + 1).round().astype(int)
        axes.append([(slice(int(e[i]), int(e[i + 1])), (win - int(e[i + 1] - e[i])) // 2) for i in range(len(e) - 1)])
    return [(tuple(s for s, _ in c), [o for _, o in c]) for c in itertools.product(*axes)]


class Reader:
    def __init__(self, vit_dir, device):
        self.device = device
        self.net, self.info = build_encoder(vit_dir, device)

    def feat(self, c, mu, sd):
        import torch
        x = torch.from_numpy((c - mu) / max(sd, 1e-6)).to(self.device)[None, None]
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16):
            return features(self.net, x)

    def run(self, vols, names, body, les, props):
        import torch
        import torch.nn.functional as F
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
        sp = props['spacing_DHW']; assert sp[0] == 1.0, sp
        size = tuple(int(round(n * s)) for n, s in zip(body.shape, sp))

        def rs(a):
            t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)[None, None]
            return F.interpolate(t, size=size, mode='trilinear', align_corners=False)[0, 0].cpu().numpy()
        b, l = rs(body) >= .5, rs(les) >= .5
        reg = regions(l.astype(np.int16), b)
        X = {n: rs(vols[n]) for n in names}
        outside = X['real'][~b]; bg = float(np.median(outside)) if outside.size else float(X['real'].min())
        tl = tiles(b)
        keys = [f'vit_{r}' for r in REGIONS] + [f'vit_block{k}_{r}' for k in SECONDARY for r in ('body', 'lesion')]
        num = {n: dict.fromkeys(keys, 0.) for n in names}; den = dict.fromkeys(keys, 0.)
        for box, off in tl:
            w = token_weights({k: canvas(m, b, 0, box, off, fill=0.0) for k, m in reg.items()}, self.device)
            rc = canvas(X['real'], b, bg, box, off); mu, sd = float(rc.mean()), float(rc.std())
            fr = self.feat(rc, mu, sd)
            for layer in fr:
                for r in (REGIONS if layer == 'norm' else ('body', 'lesion')):
                    den[f'vit_{r}' if layer == 'norm' else f'vit_{layer}_{r}'] += float(w[r].sum())
            for n in names:
                fs = self.feat(canvas(X[n], b, bg, box, off), mu, sd)
                for layer in fr:
                    d = 1 - F.cosine_similarity(fr[layer], fs[layer], dim=-1, eps=1e-6)
                    for r in (REGIONS if layer == 'norm' else ('body', 'lesion')):
                        num[n][f'vit_{r}' if layer == 'norm' else f'vit_{layer}_{r}'] += float((w[r] * d).sum())
                del fs
            del fr, w
        rows = {n: {k: (num[n][k] / den[k] if den[k] >= 1.0 else None) for k in keys} for n in names}
        ident = max(abs(v) for v in rows['real'].values() if v is not None)
        return {'grid_DHW_1mm': list(size), 'background': bg, 'tiles': len(tl),
                'tiles_per_axis': [len({t[0][d].start for t in tl}) for d in range(3)],
                'voxels': {k: int(m.sum()) for k, m in reg.items()}, 'token_weight': den,
                'identity_max': ident, 'identity_pass': ident < 1e-6, 'rows': rows}


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('entries', nargs='+'); p.add_argument('--data', required=True); p.add_argument('--pred-dir', required=True)
    p.add_argument('--out', required=True); p.add_argument('--vit-dir', default=os.environ.get('ISBI_NNF_VIT_DIR'))
    p.add_argument('--device', default='cuda'); p.add_argument('--cases', nargs='*')
    a = p.parse_args()
    reader = Reader(a.vit_dir, torch.device(a.device))
    names = list(dict.fromkeys(['real'] + a.entries))
    for case in a.cases or io.cases(a.data):
        todo = [e for e in names if not (Path(a.out) / e / f'{case}.json').exists()]
        if not todo:
            continue
        t0 = time.perf_counter(); pre, post, body, label, props = io.load_case(Path(a.data) / case)
        vols = {e: io.synthetic(e, case, pre, post, a.pred_dir) for e in set(todo) | {'real'}}
        r = reader.run(vols, ['real'] + [e for e in todo if e != 'real'], body > 0, label > 0, props)
        common = {k: v for k, v in r.items() if k != 'rows'}
        for e in todo:
            io.save_json(Path(a.out) / e / f'{case}.json', {'patient': case, 'entry': e, 'row': r['rows'][e],
                                                           'real_row': r['rows']['real'], **common,
                                                           'seconds': time.perf_counter() - t0})
        print(case, todo, flush=True)


if __name__ == '__main__':
    main()
