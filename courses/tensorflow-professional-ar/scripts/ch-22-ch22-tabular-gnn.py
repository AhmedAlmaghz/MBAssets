#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — البيانات الجدولية والتوصيات والرسوم البيانية
Professional TensorFlow — Tabular Gnn

Book slug : tensorflow-professional-ar
Chapter   : ch-22-ch22-tabular-gnn (lesson 22)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-22-ch22-tabular-gnn.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — البيانات الجدولية والتوصيات والرسوم البيانية
Professional TensorFlow — Tabular Gnn

Book slug : tensorflow-professional-ar
Chapter   : ch-22-ch22-tabular-gnn (lesson 22)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch22_tabular_gnn.md
# ────────────────────────────────────────────────────────────────────────
normalizer = keras.layers.Normalization()
normalizer.adapt(train_numeric)  # بيانات التدريب فقط
normalized = normalizer(train_numeric)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch22_tabular_gnn.md
# ────────────────────────────────────────────────────────────────────────
candidate_ids = retriever(user_vector, k=100)
scores = ranker.score(user_id, candidate_ids)
recommendations = candidate_ids[np.argsort(scores)[::-1][:10]]

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch22_tabular_gnn.md
# ────────────────────────────────────────────────────────────────────────
lookup = keras.layers.StringLookup(mask_token=None, num_oov_indices=1)
lookup.adapt(train_categories)
ids = lookup(["known-category", "new-category"])
print(ids.numpy())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-22-ch22-tabular-gnn.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
normalizer = keras.layers.Normalization()
normalizer.adapt(train_numeric)  # بيانات التدريب فقط
normalized = normalizer(train_numeric)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-22-ch22-tabular-gnn.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
candidate_ids = retriever(user_vector, k=100)
scores = ranker.score(user_id, candidate_ids)
recommendations = candidate_ids[np.argsort(scores)[::-1][:10]]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-22-ch22-tabular-gnn.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
lookup = keras.layers.StringLookup(mask_token=None, num_oov_indices=1)
lookup.adapt(train_categories)
ids = lookup(["known-category", "new-category"])
print(ids.numpy())
