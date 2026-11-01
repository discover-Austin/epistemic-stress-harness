# legacy/

The three v0.1 result files from the original single-run experiment
(Byzantine-consensus prompt; baseline, confidence, token_100).

They are kept as **golden regression fixtures**: `python validate.py` re-extracts
metrics from their stored text and asserts every v0.1 metric still reproduces.

They are **not evidence**. Each is one response from one run of one model, with
no repeats, so sampling noise alone could produce every difference between
them. Use `cli.py run` + `cli.py analyze` for anything you intend to conclude.
