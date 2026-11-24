#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — MLOps: TFX والمراقبة
Professional TensorFlow — Mlops

Book slug : tensorflow-professional-ar
Chapter   : ch-20-ch20-mlops (lesson 20)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-20-ch20-mlops.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — MLOps: TFX والمراقبة
Professional TensorFlow — Mlops

Book slug : tensorflow-professional-ar
Chapter   : ch-20-ch20-mlops (lesson 20)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch20_mlops.md
# ────────────────────────────────────────────────────────────────────────
# سجل الحد الأدنى للتجربة في منصة تتبع مثل MLflow:
run_metadata = {
    "data_version": "orders-2026-04",
    "code_revision": "git-sha-recorded-by-ci",
    "seed": 42,
    "metric_name": "macro_f1",
    "model_version": "candidate-17",
}
print(run_metadata)
# المتوقع: سجل يربط القياس بمصدر البيانات والشيفرة والنموذج.

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch20_mlops.md
# ────────────────────────────────────────────────────────────────────────
record = {"data_version": "v3", "seed": 42, "macro_f1": 0.81}
# Link the record to code, preprocessing, and model versions.

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch20_mlops.md
# ────────────────────────────────────────────────────────────────────────
baseline_mean = train_feature.mean()
current_mean = production_feature.mean()
print(current_mean - baseline_mean)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-20-ch20-mlops.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# سجل الحد الأدنى للتجربة في منصة تتبع مثل MLflow:
run_metadata = {
    "data_version": "orders-2026-04",
    "code_revision": "git-sha-recorded-by-ci",
    "seed": 42,
    "metric_name": "macro_f1",
    "model_version": "candidate-17",
}
print(run_metadata)
# المتوقع: سجل يربط القياس بمصدر البيانات والشيفرة والنموذج.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-20-ch20-mlops.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
record = {"data_version": "v3", "seed": 42, "macro_f1": 0.81}
# Link the record to code, preprocessing, and model versions.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-20-ch20-mlops.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
baseline_mean = train_feature.mean()
current_mean = production_feature.mean()
print(current_mean - baseline_mean)
