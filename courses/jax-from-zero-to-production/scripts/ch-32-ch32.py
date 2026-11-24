#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والثلاثون: الأخلاقيات والخصوصية وخارطة الطريق
JAX: From Zero to Production — Chapter 32 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-32-ch32 (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-32-ch32.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والثلاثون: الأخلاقيات والخصوصية وخارطة الطريق
JAX: From Zero to Production — Chapter 32 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-32-ch32 (lesson 32)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-32-ch32.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والثلاثون: الأخلاقيات والخصوصية وخارطة الطريق
JAX: From Zero to Production — Chapter 32 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-32-ch32 (lesson 32)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def demographic_parity(scores, group):       # group: 0 أو 1
    a = scores[group == 0]
    b = scores[group == 1]
    return float(jnp.mean(a > 0.5) - jnp.mean(b > 0.5))

scores = jax.random.bernoulli(jax.random.key(0), 0.5, (1000)).astype(float)
group = jax.random.bernoulli(jax.random.key(1), 0.5, (1000))
gap = demographic_parity(scores, group)
print(gap)                                   # قرب الصفر = توازن

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def clip_per_sample(grads, c):               # قصّ تدرّج كلّ عيّنة
    norm = jax.tree.map(jnp.linalg.norm, grads)
    return jax.tree.map(lambda gi, n: gi * jnp.minimum(1.0, c / n), grads, norm)

def add_noise(grads, key, sigma, c, batch):
    scale = sigma * c / batch
    noise = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * scale, grads)
    return jax.tree.map(lambda gi, ni: gi + ni, grads, noise)

# في خطوة التدريب:
# clipped = jax.vmap(clip_per_sample)(per_sample_grads)
# noisy = add_noise(jax.tree.map(jnp.mean, clipped), k, sigma, C, B)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def bounded_score(logits, key, eps=0.05):     # إخفاء الثقة الخام
    soft = jax.nn.softmax(logits)
    noise = jax.random.laplace(key, soft.shape) * eps
    return jnp.clip(soft + noise, 0.0, 1.0)

logits = jnp.array([3.0, -1.0, 0.5])
print(bounded_score(logits, jax.random.key(0)))   # درجات مُشفّرة

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)                  # بذرة ثابتة ومعلنة
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (3,))
print(a)                                  # نفس القيمة كل تشغيل

# مفاتيح فرعية لكلّ عيّنة، قابلة لإعادة الإنتاج
for i in range(3):
    ki = jax.random.fold_in(key, i)
    print(float(jax.random.uniform(ki)))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def demographic_parity(scores, group):       # group: 0 أو 1
    a = scores[group == 0]
    b = scores[group == 1]
    return float(jnp.mean(a > 0.5) - jnp.mean(b > 0.5))

scores = jax.random.bernoulli(jax.random.key(0), 0.5, (1000)).astype(float)
group = jax.random.bernoulli(jax.random.key(1), 0.5, (1000))
gap = demographic_parity(scores, group)
print(gap)                                   # قرب الصفر = توازن

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def clip_per_sample(grads, c):               # قصّ تدرّج كلّ عيّنة
    norm = jax.tree.map(jnp.linalg.norm, grads)
    return jax.tree.map(lambda gi, n: gi * jnp.minimum(1.0, c / n), grads, norm)

def add_noise(grads, key, sigma, c, batch):
    scale = sigma * c / batch
    noise = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * scale, grads)
    return jax.tree.map(lambda gi, ni: gi + ni, grads, noise)

# في خطوة التدريب:
# clipped = jax.vmap(clip_per_sample)(per_sample_grads)
# noisy = add_noise(jax.tree.map(jnp.mean, clipped), k, sigma, C, B)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def bounded_score(logits, key, eps=0.05):     # إخفاء الثقة الخام
    soft = jax.nn.softmax(logits)
    noise = jax.random.laplace(key, soft.shape) * eps
    return jnp.clip(soft + noise, 0.0, 1.0)

logits = jnp.array([3.0, -1.0, 0.5])
print(bounded_score(logits, jax.random.key(0)))   # درجات مُشفّرة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)                  # بذرة ثابتة ومعلنة
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (3,))
print(a)                                  # نفس القيمة كل تشغيل

# مفاتيح فرعية لكلّ عيّنة، قابلة لإعادة الإنتاج
for i in range(3):
    ki = jax.random.fold_in(key, i)
    print(float(jax.random.uniform(ki)))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والثلاثون: الأخلاقيات والخصوصية وخارطة الطريق
JAX: From Zero to Production — Chapter 32 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-32-ch32 (lesson 32)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def demographic_parity(scores, group):       # group: 0 أو 1
    a = scores[group == 0]
    b = scores[group == 1]
    return float(jnp.mean(a > 0.5) - jnp.mean(b > 0.5))

scores = jax.random.bernoulli(jax.random.key(0), 0.5, (1000)).astype(float)
group = jax.random.bernoulli(jax.random.key(1), 0.5, (1000))
gap = demographic_parity(scores, group)
print(gap)                                   # قرب الصفر = توازن

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def clip_per_sample(grads, c):               # قصّ تدرّج كلّ عيّنة
    norm = jax.tree.map(jnp.linalg.norm, grads)
    return jax.tree.map(lambda gi, n: gi * jnp.minimum(1.0, c / n), grads, norm)

def add_noise(grads, key, sigma, c, batch):
    scale = sigma * c / batch
    noise = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * scale, grads)
    return jax.tree.map(lambda gi, ni: gi + ni, grads, noise)

# في خطوة التدريب:
# clipped = jax.vmap(clip_per_sample)(per_sample_grads)
# noisy = add_noise(jax.tree.map(jnp.mean, clipped), k, sigma, C, B)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def bounded_score(logits, key, eps=0.05):     # إخفاء الثقة الخام
    soft = jax.nn.softmax(logits)
    noise = jax.random.laplace(key, soft.shape) * eps
    return jnp.clip(soft + noise, 0.0, 1.0)

logits = jnp.array([3.0, -1.0, 0.5])
print(bounded_score(logits, jax.random.key(0)))   # درجات مُشفّرة

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch32.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)                  # بذرة ثابتة ومعلنة
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (3,))
print(a)                                  # نفس القيمة كل تشغيل

# مفاتيح فرعية لكلّ عيّنة، قابلة لإعادة الإنتاج
for i in range(3):
    ki = jax.random.fold_in(key, i)
    print(float(jax.random.uniform(ki)))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def demographic_parity(scores, group):       # group: 0 أو 1
    a = scores[group == 0]
    b = scores[group == 1]
    return float(jnp.mean(a > 0.5) - jnp.mean(b > 0.5))

scores = jax.random.bernoulli(jax.random.key(0), 0.5, (1000)).astype(float)
group = jax.random.bernoulli(jax.random.key(1), 0.5, (1000))
gap = demographic_parity(scores, group)
print(gap)                                   # قرب الصفر = توازن

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def clip_per_sample(grads, c):               # قصّ تدرّج كلّ عيّنة
    norm = jax.tree.map(jnp.linalg.norm, grads)
    return jax.tree.map(lambda gi, n: gi * jnp.minimum(1.0, c / n), grads, norm)

def add_noise(grads, key, sigma, c, batch):
    scale = sigma * c / batch
    noise = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * scale, grads)
    return jax.tree.map(lambda gi, ni: gi + ni, grads, noise)

# في خطوة التدريب:
# clipped = jax.vmap(clip_per_sample)(per_sample_grads)
# noisy = add_noise(jax.tree.map(jnp.mean, clipped), k, sigma, C, B)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def bounded_score(logits, key, eps=0.05):     # إخفاء الثقة الخام
    soft = jax.nn.softmax(logits)
    noise = jax.random.laplace(key, soft.shape) * eps
    return jnp.clip(soft + noise, 0.0, 1.0)

logits = jnp.array([3.0, -1.0, 0.5])
print(bounded_score(logits, jax.random.key(0)))   # درجات مُشفّرة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-32-ch32.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)                  # بذرة ثابتة ومعلنة
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (3,))
print(a)                                  # نفس القيمة كل تشغيل

# مفاتيح فرعية لكلّ عيّنة، قابلة لإعادة الإنتاج
for i in range(3):
    ki = jax.random.fold_in(key, i)
    print(float(jax.random.uniform(ki)))
