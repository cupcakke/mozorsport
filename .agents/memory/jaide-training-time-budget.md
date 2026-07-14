---
name: JAIDE Phase C training time budget
description: Time ceiling for mini/status-bench-scale training runs in the JAIDE Modal pipeline
---

For the small-scale ("mini") Phase C training run used by the status benchmark (`scripts/modal_status_bench.py`), the user requires the whole training phase to complete in roughly 5 minutes maximum.

**Why:** This is meant to be a fast smoke/status check on real GPU hardware, not a full training run. The user considers any run that blows past ~5 minutes at this scale to be a bug requiring an immediate fix (e.g. wrong batch/step/dataset-size config, an accidental full-dataset epoch, a hung loop), not something to just let finish or move to a bigger machine.

**How to apply:** Before launching a live GPU run, check the configured step count, batch size, and dataset slice for Phase C match a ~5 minute budget on the target GPU (B200). If a run exceeds this, treat it as a defect: diagnose and fix the root cause (config/data-loading/step count), do not simply increase the timeout or downgrade expectations.
