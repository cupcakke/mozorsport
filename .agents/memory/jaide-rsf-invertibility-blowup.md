---
name: RSF invertibility depth-blowup
description: why the RSF forward/inverse round-trip diverged catastrophically at deeper configs, and the legitimate fix
---

Symptom: RSF invertibility check failed with max_abs_diff ~1e24 at dim=512/layers=12/batch=64, while the
forward/inverse math (OFTB mix adjoint, layer iteration order, translation/scale row calls) was verified
correct on paper multiple times.

Root cause: per-layer scale/translation weights (s_weight/t_weight) were Xavier-initialized with no bound
on their operator (spectral) norm. Across 12 stacked layers, per-layer scale factors near exp(clip_max=5)
compounded, driving the intermediate tensor magnitude toward ~1e26. At that magnitude, float32 rounding
error in the per-layer scale/translation round-trip becomes large in absolute terms and — because the
clip step is a saturating, non-smooth function — a tiny forward-vs-inverse rounding discrepancy can land
on opposite sides of the clip boundary, blowing up the reconstruction error by many orders of magnitude.

**Why:** this is a genuine numerical-stability defect in the architecture (unconstrained weight scale
composed across depth), not a coding bug in forward/inverse control flow, and not something a looser
comparison tolerance can legitimately paper over.

**How to apply:** constrain the spectral norm of newly-initialized per-layer weight matrices (power
iteration, target norm comfortably below 1, e.g. 0.9) right after Xavier init, before first use. This keeps
per-layer scale/translation near-identity so depth composition stays well-conditioned, without touching
clip_min/clip_max (which may be hard-coded elsewhere, e.g. GPU-kernel compatibility gates expecting exact
values) and without loosening test tolerances.
