"""CPU metric equivalence on three saved patients and two arms. Source and artifact paths are supplied through
ISBI environment variables; missing inputs skip. ISBI_EVAL_REPORT optionally saves all measured max-absolute gaps.
Float32 image-stage reference values and float16 stored-prediction values are reported separately."""
import ast, json, math, os, sys, time, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.metrics import io
from evaluation.metrics.image import image_metrics
from evaluation.metrics.global_.psnr import psnr
from evaluation.metrics.offtarget import false_enhancement as FE
from evaluation.metrics.perceptual import sharpness as SHp
from evaluation.metrics.segmenter.mama_reader import official_fold, seg_native

REQUIRED = ('ISBI_ORIG_ROOT', 'ISBI_ORIG_PROJECTS', 'ISBI_ORIG_TOOLS', 'ISBI_EVAL_SAVED',
            'ISBI_EVAL_FOCUS', 'ISBI_READER_DIR', 'ISBI_EVAL_TMP')
CASES = ['DUKE_019', 'DUKE_055', 'DUKE_271']
ARMS = ['l1_ep300', 'gan_nnf_mam_scratch300_ep200']


def original(path, names, extra=None):
    src = Path(path).read_text(); tree = ast.parse(src); ns = dict(np=np, ndi=ndi, structural_similarity=structural_similarity, Path=Path)
    ns.update(extra or {})
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('SP', 'SIGMA_MM', 'MARGIN_MM', 'METRICS') for t in node.targets):
            exec(compile(ast.Module([node], []), str(path), 'exec'), ns)
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), 'exec'), ns)
    return ns



def gap(x, y):
    out = {}
    for k, v in x.items():
        w = y.get(k)
        if isinstance(v, bool) or isinstance(w, bool) or v is None or w is None:
            out[k] = 0.0 if v == w else float('inf')
        elif isinstance(v, (int, float)) and isinstance(w, (int, float)):
            out[k] = abs(float(v) - float(w))
    return out


def worst(d, acc):
    for k, v in d.items():
        acc[k] = max(acc.get(k, 0.0), v)
    return acc



def run_checks(tmp):
    R = Path(os.environ['ISBI_ORIG_ROOT']); T = Path(os.environ['ISBI_ORIG_TOOLS'])
    A = Path(os.environ['ISBI_EVAL_SAVED']); F = Path(os.environ['ISBI_EVAL_FOCUS'])
    DATA = R / 'test91' / 'data'; PRED = A / 'pred'
    P = Path(os.environ['ISBI_ORIG_PROJECTS'])
    ORIG_EV = original(P / 'mama_3d_20261018_r11_arch' / 'evaluate.py', ['image_metrics'])
    ORIG_SH = original(T / 'mama_sharpness.py', ['grad_mag', 'log_mm', 'context', 'metrics'])
    ORIG_PL = original(T / 'mama_planche_ep300.py', ['seg_native'])


    class SelectedPred:
        def __truediv__(self, name):
            return PRED / name
        def iterdir(self):
            return iter([PRED / a for a in ARMS])

    PS = original(T / 't91_psnr_20261029.py', ['one'],
                  dict(DATA=DATA, PRED=SelectedPred(), OUT=tmp / 'psnr', json=json))
    FO = original(T / 't91_false_enh_20261029.py', ['one'],
                  dict(DATA=DATA, PRED=PRED, OUT=tmp / 'fe', ARMS=ARMS, TAUS=(2., 3.),
                       ST=ndi.generate_binary_structure(3, 3), json=json))
    res = {'cases': CASES, 'arms': ARMS, 'checks': {}}
    W = res['checks']
    t0 = time.time()
    for case in CASES:
        PS['one'](case)
        FO['one'](case)
        pre, post, body, label, props = io.load_case(DATA / case)
        b, les = body > 0, label > 0
        ctx_port = SHp.context(post, les); ctx_orig = ORIG_SH['context'](post, les)
        O = FE.offtarget_region(body, label); vml = float(np.prod(props['spacing_DHW'])) / 1000.
        for ent in ARMS + ['pre']:
            y = io.synthetic(ent, case, pre, post, PRED)
            # image metrics (gen stage): (a) port == original on the same float32 input; (b) vs saved values (saved from float32,
            # recomputed here from the float16 file)
            mp = image_metrics(y, pre, post, body, label)
            W.setdefault('image|port_vs_original_same_input', {})
            worst(gap(mp, ORIG_EV['image_metrics'](y, pre, post, body, label)), W['image|port_vs_original_same_input'])
            saved = io.read_json((R / 'test91' / 'evaluations' / 'copie' / f'{case}.json') if ent == 'pre'
                                 else (R / 'evaluations_test91' / ent / f'{case}.json'))['metrics']
            W.setdefault(f'image|port_on_float16_vs_saved|{ent}', {})
            worst(gap(mp, saved), W[f'image|port_on_float16_vs_saved|{ent}'])
            # r (lesion recovery)
            r_port = mp['lesion_enhancement_prediction_z'] / mp['lesion_enhancement_real_z']
            r_saved = saved['lesion_enhancement_prediction_z'] / saved['lesion_enhancement_real_z']
            W.setdefault(f'r|port_on_float16_vs_saved|{ent}', {}); worst({'r': abs(r_port - r_saved)}, W[f'r|port_on_float16_vs_saved|{ent}'])
            # PSNR (paper)
            W.setdefault('psnr|port_vs_saved', {}); worst(gap(psnr(y, post, body), io.read_json(F / 'psnr' / ent / f'{case}.json')), W['psnr|port_vs_saved'])
            W.setdefault('psnr|port_vs_original_same_input', {})
            worst(gap(psnr(y, post, body), io.read_json(tmp / 'psnr' / ent / f'{case}.json')),
                  W['psnr|port_vs_original_same_input'])
            # sharpness
            sp = SHp.metrics(y, post, b, ctx_port)
            W.setdefault('sharpness|port_vs_original_same_input', {}); worst(gap(sp, ORIG_SH['metrics'](y, post, b, ctx_orig)), W['sharpness|port_vs_original_same_input'])
            W.setdefault('sharpness|port_vs_saved', {})
            worst(gap({k: sp[k] for k in SHp.METRICS}, io.read_json(A / 'extra' / 'cpu' / ent / f'{case}.json')['metrics']), W['sharpness|port_vs_saved'])
            # false enhancement (no saved row for the copy)
            if ent != 'pre':
                fe = FE.false_enhancement(y - pre, post - pre, O, label, vml)
                sv = io.read_json(F / 'false_enh' / ent / f'{case}.json')
                W.setdefault('offtarget|port_vs_saved', {})
                for t in ('t2', 't3'):
                    worst({f'{t}_{k}': v for k, v in gap(fe[t], sv[t]).items()}, W['offtarget|port_vs_saved'])
                assert sv['voxel_ml'] == vml
                ref = io.read_json(tmp / 'fe' / ent / f'{case}.json')
                W.setdefault('offtarget|port_vs_original_same_input', {})
                worst({f'{t}_{k}': v for t in ('t2', 't3') for k, v in gap(fe[t], ref[t]).items()},
                      W['offtarget|port_vs_original_same_input'])
        # off-target control: real image
        fe = FE.false_enhancement(post - pre, post - pre, O, label, vml); sv = io.read_json(F / 'false_enh' / 'real' / f'{case}.json')
        W.setdefault('offtarget|real|port_vs_saved', {}); worst({f'{t}_{k}': v for t in ('t2', 't3') for k, v in gap(fe[t], sv[t]).items()}, W['offtarget|real|port_vs_saved'])
        print(case, 'done', round(time.time() - t0), 's', flush=True)

    # segmenter: fold rule on all 91 patients vs the folds saved by the reader; seg_native port vs original on synthetic masks
    RD = Path(os.environ['ISBI_READER_DIR'])
    all_cases = io.cases(DATA); folds = official_fold(RD, all_cases)
    mism = [c for c in all_cases if io.read_json(A / 'lecteur' / 'l1_ep300' / f'{c}.json')['fold'] != folds[c][0]
            or io.read_json(A / 'lecteur' / 'l1_ep300' / f'{c}.json')['official_csv_dice'] != folds[c][1]]
    W['segmenter|fold_rule_vs_saved_all91'] = {'n': len(all_cases), 'mismatches': len(mism)}
    rng = np.random.default_rng(0); g = {}
    for i in range(20):
        pm = ndi.binary_dilation(rng.random((40, 60, 60)) > 0.995, iterations=2); gm = ndi.binary_dilation(rng.random((40, 60, 60)) > 0.995, iterations=2)
        sp_ = [float(x) for x in rng.uniform(0.5, 2.0, 3)]
        worst(gap(seg_native(pm, gm, sp_), ORIG_PL['seg_native'](pm, gm, sp_)), g)
    W['segmenter|seg_native_port_vs_original_random_masks'] = g
    if os.environ.get('ISBI_EVAL_REPORT'):
        Path(os.environ['ISBI_EVAL_REPORT']).write_text(json.dumps(res, indent=1))
    for k, v in W.items():
        vals = [x for x in v.values() if isinstance(x, float)]
        print(f'{k:55s} max gap {max(vals) if vals else v}  ' + ('' if not vals else ' '.join(f'{kk}={vv:.2e}' for kk, vv in v.items() if vv > 0)))

    # Same-input ports and float16-derived saved stages must agree exactly. Image-stage values saved before float16
    # storage are reported separately, rather than being labelled exact equivalence or hidden by a wider tolerance.
    for key, values in W.items():
        if 'float16_vs_saved' not in key:
            for metric, value in values.items():
                if metric != 'n':
                    assert value == 0., (key, metric, value)


@unittest.skipUnless(all(os.environ.get(k) for k in REQUIRED), 'set the ISBI original-source/data variables')
class TestSavedMetrics(unittest.TestCase):
    def test_three_cases_two_arms(self):
        with tempfile.TemporaryDirectory(dir=os.environ['ISBI_EVAL_TMP'], prefix='equivalence_') as tmp:
            run_checks(Path(tmp))


@unittest.skipUnless(os.environ.get('ISBI_READER_CPU') and os.environ.get('ISBI_EVAL_SAVED'),
                     'set ISBI_READER_CPU and ISBI_EVAL_SAVED after full CPU reader recomputation')
class TestReaderSaved(unittest.TestCase):
    def test_two_cases_two_arms(self):
        rows = {}
        for case in ('DUKE_019', 'DUKE_055'):
            for entry in ARMS:
                a = io.read_json(Path(os.environ['ISBI_READER_CPU']) / entry / f'{case}.json')
                b = io.read_json(Path(os.environ['ISBI_EVAL_SAVED']) / 'lecteur' / entry / f'{case}.json')
                self.assertEqual((a['fold'], a['official_csv_dice']), (b['fold'], b['official_csv_dice']))
                self.assertEqual(set(a['row']), set(b['row']))
                self.assertEqual(a['row']['true_lesion_ml'], b['row']['true_lesion_ml'])
                d = gap(a['row'], b['row'])
                self.assertTrue(all(np.isfinite(v) for v in d.values()))
                rows[f'{entry}/{case}'] = d
                print('GAP reader CPU fp32 vs saved GPU',entry,case,json.dumps(d),flush=True)
        if os.environ.get('ISBI_READER_REPORT'):
            Path(os.environ['ISBI_READER_REPORT']).write_text(json.dumps(rows,indent=1))


if __name__ == '__main__':
    unittest.main()
