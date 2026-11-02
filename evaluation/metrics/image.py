"""Image metrics of one synthetic volume, as written by the gen stage (one JSON per arm and patient).

Same keys and values as the training-side development evaluation, so development and test outputs stay comparable.
Compute them on the float32 prediction, before it is stored in float16 (recomputing from the float16 file moves the
values by up to ~1e-5 z; see docs/metrics.md).
"""
from evaluation.metrics.global_.image import SSIM_DATA_RANGE_Z, body_bias, mae_body, ssim_body
from evaluation.metrics.lesion.enhancement import lesion_metrics


def image_metrics(pred, pre, post, body, label):
    les = lesion_metrics(pred, pre, post, body, label)
    return {'MAE_body_z': mae_body(pred, post, body), 'MAE_lesion_z': les['MAE_lesion_z'], 'MAE_ring_z': les['MAE_ring_z'],
            'SSIM_body': ssim_body(pred, post, body), 'SSIM_data_range_z': SSIM_DATA_RANGE_Z,
            'lesion_enhancement_prediction_z': les['lesion_enhancement_prediction_z'],
            'lesion_enhancement_real_z': les['lesion_enhancement_real_z'],
            'abs_error_mean_lesion_enhancement_z': les['abs_error_mean_lesion_enhancement_z'],
            'body_bias_z': body_bias(pred, post, body), 'positive_error_outside_lesion_z': les['positive_error_outside_lesion_z'],
            'body_mask_is_breast_anatomical_segmentation': False}


def main():
    """Image metrics per entry and patient: <out>/<entry>/<case>.json.

    run ENTRY...  synthetic volumes <pred-dir>/<entry>/<case>.npy, read as float32. The paper values were computed on
                  the float32 prediction: write it with `synthesis.inference.predict --save-dtype float32`, and pass
                  --store-float16 DIR to write the float16 copy that every later stage reads.
    copy          the pre-contrast copy read as the synthetic volume (entry 'pre').
    """
    import argparse
    from pathlib import Path

    import numpy as np

    from evaluation.metrics import io
    p = argparse.ArgumentParser(description=main.__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=main.__doc__)
    p.add_argument('cmd', choices=['run', 'copy']); p.add_argument('entries', nargs='*')
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--pred-dir'); p.add_argument('--store-float16')
    a = p.parse_args()
    entries = ['pre'] if a.cmd == 'copy' else a.entries
    assert a.cmd == 'copy' or (entries and a.pred_dir), 'run needs ENTRY... and --pred-dir'
    for entry in entries:
        for case in io.cases(a.data):
            o = Path(a.out) / entry / f'{case}.json'
            if o.exists():
                continue
            pre, post, body, label, _ = io.load_case(Path(a.data) / case)
            if entry == 'pre':
                pred = pre
            else:
                raw = np.load(Path(a.pred_dir) / entry / f'{case}.npy')
                pred = raw.astype(np.float32)
                if a.store_float16:
                    io.save_npy(Path(a.store_float16) / entry / f'{case}.npy', raw.astype(np.float16))
            io.save_json(o, {'patient': case, 'entry': entry, 'prediction_dtype': 'float32' if entry == 'pre' else str(raw.dtype),
                             'metrics': image_metrics(pred, pre, post, body, label)})
            print(entry, case, flush=True)


if __name__ == '__main__':
    main()
