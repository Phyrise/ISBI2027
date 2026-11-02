# Tests

`tests/` holds CPU equivalence checks between this code and the original experiment code that produced the paper:
schedules, seeds and initial states, data sampling, losses and AFP teachers, optimiser updates, checkpoints,
inference and every evaluation metric. Each check compares both implementations on the same inputs and reports the
measured gap; no tolerance is widened.

The original code, data and weights are not part of this repository, so every check that needs them is skipped
unless its environment variables are set (listed at the top of `tests/train_support.py`, `tests/test_eval_cpu.py`
and `tests/test_eval_perceptual.py`).

```bash
# training side (training environment)
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -t . -p 'test_train_*.py' -v
# evaluation side (evaluation environment)
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -t . -p 'test_eval_*.py' -v
```

`ISBI_TEST_SLOW=1` adds the heavy checks (saved checkpoints, raw-case preprocessing, full-volume expert inference).
