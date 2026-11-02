"""Split, preprocessing and sampler vs the paper run: the regenerated split vs manifest.json, a fresh preprocessing of
two raw cases vs the paper arrays, and the sampler batches vs the original Batches for fixed seeds and update indices."""
import json
import shutil
import tempfile
import unittest

import numpy as np

from synthesis.data.preprocess import prepare_case
from synthesis.data.sampler import Batches, split_cases
from synthesis.data.split import field_strength, official_duke, records, split_ids
from tests.train_support import REPO, env_path, max_abs, need, original, report, root, slow

R11 = 'mama_3d_20261018_r11_arch'
SPLIT = REPO / 'configs/split_duke.json'
SEED = 20261018
# update indices: first batches, the 100-epoch fork point, the last update of a 300-epoch run
TRAIN_STEPS = (0, 1, 2, 3, 25000, 74999)
VAL_STEPS = (0, 1, 19)


def manifest():
    return json.loads((root() / 'manifest.json').read_text())


@need('ISBI_ORIG_ROOT')
class TestSplit(unittest.TestCase):
    def test_split_file_vs_manifest(self):
        m, s = manifest()['splits'], json.loads(SPLIT.read_text())['splits']
        for k in ('train', 'dev', 'test'):
            self.assertEqual(s[k], [r['patient'] for r in m[k]], k)  # same order: the sampler indexes this list
        report('data', 'configs/split_duke.json vs manifest.json (ordered IDs, 160/40/91)', 0.)

    @need('ISBI_MAMA_RAW')
    def test_regenerated_split(self):
        raw = env_path('ISBI_MAMA_RAW')
        m = manifest()['splits']
        tr, te = official_duke(raw)
        field_of = lambda c: field_strength(raw, c)  # noqa: E731
        train, dev, test = split_ids(tr, te, field_of)
        differ = sum(a != b for k, ids in (('train', train), ('dev', dev), ('test', test))
                     for a, b in zip(ids, sorted(r['patient'] for r in m[k])))
        sizes = (len(train), len(dev), len(test))
        fields = sum(field_of(r['patient']) != r['field_strength'] for k in ('train', 'dev') for r in m[k])
        mine = {r['patient']: r for r in records(raw, train + dev + test, field_of)}
        paths = sum(mine[r['patient']][k] != r[k] for s in ('train', 'dev', 'test') for r in m[s]
                    for k in ('pre', 'post', 'label'))
        report('data', 'regenerated split vs manifest.json: IDs differing', float(differ), sizes=sizes)
        report('data', 'field strength patient_info_files vs manifest.json: cases differing', float(fields))
        report('data', 'image / mask paths vs manifest.json: paths differing', float(paths))
        self.assertEqual(sizes, (160, 40, 91))
        self.assertEqual((differ, fields, paths), (0, 0, 0))


@need('ISBI_ORIG_ROOT', 'ISBI_ORIG_PROJECTS', 'ISBI_ORIG_MX')
class TestSampler(unittest.TestCase):
    def compare(self, split, seed, steps):
        ours = Batches(root() / 'data', split_cases(SPLIT, split), seed, 0, max(steps) + 1)
        gaps, plans = {}, 0
        with original(R11, 'common') as C:
            theirs = C.Batches(split, seed, 0, max(steps) + 1)
            self.assertEqual(ours.ids if hasattr(ours, 'ids') else ours.cases, theirs.ids)
            for s in steps:
                a, b = ours[s], theirs[s]
                plans += a['plan'] != b['plan']
                for k in ('data', 'valid', 'label'):
                    self.assertEqual(a[k].dtype, b[k].dtype)
                    self.assertEqual(tuple(a[k].shape), tuple(b[k].shape))
                    gaps[k] = max(gaps.get(k, 0.), max_abs(a[k], b[k]))
        return gaps, plans

    def test_train_batches(self):
        gaps, plans = self.compare('train', SEED, TRAIN_STEPS)
        report('data', f'sampler train batches at updates {TRAIN_STEPS} (seed {SEED})', max(gaps.values()),
               per_array=gaps, plans_differing=plans)
        self.assertEqual((plans, max(gaps.values())), (0, 0.))

    def test_validation_batches(self):
        gaps, plans = self.compare('dev', SEED + 999, VAL_STEPS)
        report('data', f'sampler dev batches {VAL_STEPS} (seed {SEED + 999})', max(gaps.values()), per_array=gaps,
               plans_differing=plans)
        self.assertEqual((plans, max(gaps.values())), (0, 0.))


@need('ISBI_ORIG_ROOT', 'ISBI_MAMA_RAW', 'ISBI_TEST_TMP')
class TestPreprocess(unittest.TestCase):
    @slow()
    def test_prepare_two_cases_vs_paper_arrays(self):
        raw = env_path('ISBI_MAMA_RAW')
        env_path('ISBI_TEST_TMP').mkdir(parents=True, exist_ok=True)
        out = tempfile.mkdtemp(prefix='prep_', dir=env_path('ISBI_TEST_TMP'))
        try:
            for case in ('DUKE_001', 'DUKE_059'):  # a 1.5 T train case and a 3 T dev case
                rec = records(raw, [case], lambda c: field_strength(raw, c))[0]
                prepare_case(rec, out)
                mine, paper = f'{out}/{case}', root() / 'data' / case
                gaps = {}
                for k in ('data', 'label', 'valid', 'body'):
                    a, b = np.load(f'{mine}/{k}.npy'), np.load(paper / f'{k}.npy')
                    self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), k)
                    gaps[k] = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                with np.load(f'{mine}/locations.npz') as f, np.load(paper / 'locations.npz') as g:
                    for k in ('body', 'tumour'):
                        self.assertEqual(f[k].shape, g[k].shape, k)
                        gaps[f'locations.{k}'] = float(np.abs(f[k].astype(np.int64) - g[k].astype(np.int64)).max())
                p, q = (json.loads(open(f'{mine}/properties.json').read()),
                        json.loads((paper / 'properties.json').read_text()))
                keys = ('shape_DHW', 'affine_RAS', 'spacing_DHW', 'normalization', 'native_shape_XYZ',
                        'native_spacing_XYZ', 'body_fraction', 'lesion_voxels', 'lesion_fraction_in_source_body',
                        'real_enhancement_lesion_z', 'real_mean_body_shift_z')
                props = [k for k in keys if p[k] != q[k]]
                sources = [k for k in ('pre', 'post', 'label') if p['sources'][k]['sha256'] != q['sources'][k]['sha256']]
                report('data', f'preprocessing {case} vs paper data/{case}', max(gaps.values()), per_array=gaps,
                       properties_differing=props, source_sha_differing=sources)
                self.assertEqual((max(gaps.values()), props, sources), (0., [], []))
        finally:
            shutil.rmtree(out, ignore_errors=True)
