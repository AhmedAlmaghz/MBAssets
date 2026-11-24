#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق هـ: حلول مختارة للتمارين
Mastering Transformers and Large Language Models — E Exercise Solutions

Book slug : mastering-llms-transformers
Chapter   : app-e-e-exercise-solutions (lesson e)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/app-e-e-exercise-solutions.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق هـ: حلول مختارة للتمارين
Mastering Transformers and Large Language Models — E Exercise Solutions

Book slug : mastering-llms-transformers
Chapter   : app-e-e-exercise-solutions (lesson e)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/app-e-e-exercise-solutions.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق هـ: حلول مختارة للتمارين
Mastering Transformers and Large Language Models — E Exercise Solutions

Book slug : mastering-llms-transformers
Chapter   : app-e-e-exercise-solutions (lesson e)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.2, 0.3, 0.5])
print(kl(P, Q), kl(Q, P))

# ────────────────────────────────────────────────────────────────────────
# [002] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
def fertility(text, tokenizer):
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / max(len(words), 1)

# ────────────────────────────────────────────────────────────────────────
# [003] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
# مفهومي: استخدم دالة rope(x, pos) من الفصل
s1 = (rope(q, 5) * rope(k, 10)).sum()
s2 = (rope(q, 100) * rope(k, 105)).sum()
print(s1, s2)

# ────────────────────────────────────────────────────────────────────────
# [004] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
# مفهومي
base = model.get_output_embeddings().weight.data.clone()
merged = base + (lora_B @ lora_A) * (alpha / r)
assert torch.allclose(merged, expected, atol=1e-5)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.2, 0.3, 0.5])
print(kl(P, Q), kl(Q, P))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def fertility(text, tokenizer):
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / max(len(words), 1)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# مفهومي: استخدم دالة rope(x, pos) من الفصل
s1 = (rope(q, 5) * rope(k, 10)).sum()
s2 = (rope(q, 100) * rope(k, 105)).sum()
print(s1, s2)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# مفهومي
base = model.get_output_embeddings().weight.data.clone()
merged = base + (lora_B @ lora_A) * (alpha / r)
assert torch.allclose(merged, expected, atol=1e-5)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الملحق هـ: حلول مختارة للتمارين
Mastering Transformers and Large Language Models — E Exercise Solutions

Book slug : mastering-llms-transformers
Chapter   : app-e-e-exercise-solutions (lesson e)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.2, 0.3, 0.5])
print(kl(P, Q), kl(Q, P))

# ────────────────────────────────────────────────────────────────────────
# [002] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
def fertility(text, tokenizer):
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / max(len(words), 1)

# ────────────────────────────────────────────────────────────────────────
# [003] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
# مفهومي: استخدم دالة rope(x, pos) من الفصل
s1 = (rope(q, 5) * rope(k, 10)).sum()
s2 = (rope(q, 100) * rope(k, 105)).sum()
print(s1, s2)

# ────────────────────────────────────────────────────────────────────────
# [004] content/appendices/e-exercise-solutions.md
# ────────────────────────────────────────────────────────────────────────
# مفهومي
base = model.get_output_embeddings().weight.data.clone()
merged = base + (lora_B @ lora_A) * (alpha / r)
assert torch.allclose(merged, expected, atol=1e-5)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def kl(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.2, 0.3, 0.5])
print(kl(P, Q), kl(Q, P))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def fertility(text, tokenizer):
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / max(len(words), 1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# مفهومي: استخدم دالة rope(x, pos) من الفصل
s1 = (rope(q, 5) * rope(k, 10)).sum()
s2 = (rope(q, 100) * rope(k, 105)).sum()
print(s1, s2)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/app-e-e-exercise-solutions.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# مفهومي
base = model.get_output_embeddings().weight.data.clone()
merged = base + (lora_B @ lora_A) * (alpha / r)
assert torch.allclose(merged, expected, atol=1e-5)
