"""Reading prepared cases and stage outputs; writing one JSON per entry and patient."""
import json
from pathlib import Path

import numpy as np


def read_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, x):
    """Atomic write (temporary file, then rename)."""
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(x, indent=1, allow_nan=False)); t.replace(p)


def save_npy(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t = p.with_name(p.name + '.tmp')
    with t.open('wb') as f:
        np.save(f, x)
    t.replace(p)


def entry_name(key, epoch=None):
    """Folder name of an entry: '<arm>_epNNN', or 'pre' / 'real'."""
    return key if key in ('pre', 'real') else f'{key}_ep{int(epoch):03d}'


def cases(data_dir):
    return sorted(p.name for p in Path(data_dir).iterdir() if (p / 'data.npy').exists())


def load_case(case_dir):
    """(pre, post, body, label, properties) of one prepared case; pre / post float32 z units, masks as stored."""
    d = Path(case_dir)
    data = np.load(d / 'data.npy', mmap_mode='r')
    return (np.array(data[0], dtype=np.float32), np.array(data[1], dtype=np.float32), np.load(d / 'body.npy'),
            np.load(d / 'label.npy'), read_json(d / 'properties.json'))


def load_prediction(pred_dir, entry, case):
    """float16 prediction of the gen stage, returned as float32."""
    return np.load(Path(pred_dir) / entry / f'{case}.npy').astype(np.float32)


def synthetic(entry, case, pre, post, pred_dir):
    """The volume read as 'synthetic' for an entry: the copy is the pre-contrast, 'real' the post-contrast."""
    if entry == 'pre':
        return pre
    if entry == 'real':
        return post
    return load_prediction(pred_dir, entry, case)
