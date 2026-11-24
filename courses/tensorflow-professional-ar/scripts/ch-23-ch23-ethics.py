#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الأخلاق والخصوصية والتفسير والأمان
Professional TensorFlow — Ethics

Book slug : tensorflow-professional-ar
Chapter   : ch-23-ch23-ethics (lesson 23)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-23-ch23-ethics.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الأخلاق والخصوصية والتفسير والأمان
Professional TensorFlow — Ethics

Book slug : tensorflow-professional-ar
Chapter   : ch-23-ch23-ethics (lesson 23)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch23_ethics.md
# ────────────────────────────────────────────────────────────────────────
for group in groups:
    mask = sensitive_group == group
    print(group, (y_pred[mask] != y_true[mask]).mean(), mask.sum())

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch23_ethics.md
# ────────────────────────────────────────────────────────────────────────
score_a = model(x, training=False)
score_b = model(x_perturbed, training=False)
delta = tf.reduce_max(tf.abs(score_a - score_b))
print(float(delta))

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch23_ethics.md
# ────────────────────────────────────────────────────────────────────────
group_rates = {
    group: ((y_pred[group_mask] != y_true[group_mask]).mean(), int(group_mask.sum()))
    for group, group_mask in masks.items()
}
print(group_rates)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-23-ch23-ethics.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
for group in groups:
    mask = sensitive_group == group
    print(group, (y_pred[mask] != y_true[mask]).mean(), mask.sum())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-23-ch23-ethics.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
score_a = model(x, training=False)
score_b = model(x_perturbed, training=False)
delta = tf.reduce_max(tf.abs(score_a - score_b))
print(float(delta))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-23-ch23-ethics.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
group_rates = {
    group: ((y_pred[group_mask] != y_true[group_mask]).mean(), int(group_mask.sum()))
    for group, group_mask in masks.items()
}
print(group_rates)
