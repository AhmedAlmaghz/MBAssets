#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٥: محرّك توصيات
Mastering scikit-learn 1.9 — Chapter 103 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-103-ch103 (lesson 103)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-scikit-learn-1-9/code/ch-103-ch103.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان scikit-learn 1.9 — مشروع ٥: محرّك توصيات
Mastering scikit-learn 1.9 — Chapter 103 — Projects

Book slug : mastering-scikit-learn-1-9
Chapter   : ch-103-ch103 (lesson 103)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/part12/ch103.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

items = ["فضاء", "مغامرة", "مدينة", "طبخ", "ألغاز", "طبيعة"]
features = np.array([
    [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
    [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0],
], dtype=float)
similarity = cosine_similarity(features)
target = items.index("فضاء")
neighbors = np.argsort(similarity[target])[::-1]
print("ترتيب التشابه لفضاء:")
for idx in neighbors[1:4]:
    print(items[idx], round(float(similarity[target, idx]), 3))

# ────────────────────────────────────────────────────────────────────────
# [002] content/part12/ch103.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

names = ["خيال", "مغامرة", "تاريخ", "طبخ", "ألغاز", "طبيعة"]
V = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
              [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
seen = np.array([0, 1])
profile = V[seen].mean(axis=0, keepdims=True)
scores = cosine_similarity(profile, V).ravel()
scores[seen] = -np.inf
ranking = np.argsort(scores)[::-1]
print("التوصيات:", [(names[i], round(float(scores[i]), 3))
                   for i in ranking[:3]])

# ────────────────────────────────────────────────────────────────────────
# [003] content/part12/ch103.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = np.array([
    [5, 4, 0, 0, 1],
    [4, 5, 0, 0, 1],
    [0, 0, 5, 4, 0],
    [0, 1, 4, 5, 0],
], dtype=float)
item_similarity = cosine_similarity(ratings.T)
print("تشابه العنصرين 0 و1:", round(float(item_similarity[0, 1]), 3))
print("تشابه العنصرين 2 و3:", round(float(item_similarity[2, 3]), 3))
print("نسبة الخانات المرصودة:", round(float(np.count_nonzero(ratings) / ratings.size), 3))

# ────────────────────────────────────────────────────────────────────────
# [004] content/part12/ch103.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import ndcg_score
import numpy as np

relevance = np.array([[3, 0, 2, 1, 0]])
ranking_a = np.array([[0.9, 0.8, 0.7, 0.2, 0.1]])
ranking_b = np.array([[0.1, 0.2, 0.3, 0.9, 0.8]])
print("NDCG@3 جيد:", round(float(ndcg_score(relevance, ranking_a, k=3)), 4))
print("NDCG@3 ضعيف:", round(float(ndcg_score(relevance, ranking_b, k=3)), 4))

# ────────────────────────────────────────────────────────────────────────
# [005] content/part12/ch103.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

features = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], float)
profile = features[[0, 1]].mean(axis=0, keepdims=True)
scores = cosine_similarity(profile, features).ravel()
scores[[0, 1]] = -np.inf
print("العنصر التالي:", int(np.argmax(scores)))

# ────────────────────────────────────────────────────────────────────────
# [006] content/part12/ch103.md
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import ndcg_score

truth = [[3, 2, 0, 1]]
for scores in ([[0.9, 0.8, 0.2, 0.1]], [[0.1, 0.2, 0.9, 0.8]]):
    print(round(float(ndcg_score(truth, scores, k=3)), 4))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-103-ch103.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

items = ["فضاء", "مغامرة", "مدينة", "طبخ", "ألغاز", "طبيعة"]
features = np.array([
    [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
    [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0],
], dtype=float)
similarity = cosine_similarity(features)
target = items.index("فضاء")
neighbors = np.argsort(similarity[target])[::-1]
print("ترتيب التشابه لفضاء:")
for idx in neighbors[1:4]:
    print(items[idx], round(float(similarity[target, idx]), 3))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-103-ch103.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

names = ["خيال", "مغامرة", "تاريخ", "طبخ", "ألغاز", "طبيعة"]
V = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
              [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
seen = np.array([0, 1])
profile = V[seen].mean(axis=0, keepdims=True)
scores = cosine_similarity(profile, V).ravel()
scores[seen] = -np.inf
ranking = np.argsort(scores)[::-1]
print("التوصيات:", [(names[i], round(float(scores[i]), 3))
                   for i in ranking[:3]])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-103-ch103.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = np.array([
    [5, 4, 0, 0, 1],
    [4, 5, 0, 0, 1],
    [0, 0, 5, 4, 0],
    [0, 1, 4, 5, 0],
], dtype=float)
item_similarity = cosine_similarity(ratings.T)
print("تشابه العنصرين 0 و1:", round(float(item_similarity[0, 1]), 3))
print("تشابه العنصرين 2 و3:", round(float(item_similarity[2, 3]), 3))
print("نسبة الخانات المرصودة:", round(float(np.count_nonzero(ratings) / ratings.size), 3))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-103-ch103.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import ndcg_score
import numpy as np

relevance = np.array([[3, 0, 2, 1, 0]])
ranking_a = np.array([[0.9, 0.8, 0.7, 0.2, 0.1]])
ranking_b = np.array([[0.1, 0.2, 0.3, 0.9, 0.8]])
print("NDCG@3 جيد:", round(float(ndcg_score(relevance, ranking_a, k=3)), 4))
print("NDCG@3 ضعيف:", round(float(ndcg_score(relevance, ranking_b, k=3)), 4))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-103-ch103.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

features = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], float)
profile = features[[0, 1]].mean(axis=0, keepdims=True)
scores = cosine_similarity(profile, features).ravel()
scores[[0, 1]] = -np.inf
print("العنصر التالي:", int(np.argmax(scores)))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-scikit-learn-1-9/notebooks/ch-103-ch103.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from sklearn.metrics import ndcg_score

truth = [[3, 2, 0, 1]]
for scores in ([[0.9, 0.8, 0.2, 0.1]], [[0.1, 0.2, 0.9, 0.8]]):
    print(round(float(ndcg_score(truth, scores, k=3)), 4))
