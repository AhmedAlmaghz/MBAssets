#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — random: العشوائية الصريحة
JAX: From Zero to Production — Chapter 10 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-10-ch10 (lesson 10)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-10-ch10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — random: العشوائية الصريحة
JAX: From Zero to Production — Chapter 10 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-10-ch10 (lesson 10)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-10-ch10.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — random: العشوائية الصريحة
JAX: From Zero to Production — Chapter 10 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-10-ch10 (lesson 10)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
import jax, numpy as np, jax.numpy as jnp

@jax.jit
def noise():
    return jnp.asarray(np.random.normal(size=3))

print(noise())
print(noise())      # نفس الأرقام تماماً

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
import jax

k = jax.random.key(0)
print(k.dtype, k.shape)          # key ()
print(jax.random.key_data(k))    # uint32[2]

try:
    k + 1
except TypeError as exc:
    print("لا حساب على المفاتيح:", exc)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
import jax

key = jax.random.key(42)
k1, k2 = jax.random.split(key)          # مفتاحان من مفتاح
ka, kb, kc = jax.random.split(k1, 3)    # ثلاثة من الأول

print(jax.random.normal(k1, (2,)))
print(jax.random.normal(k2, (2,)))
print(jax.random.normal(jax.random.fold_in(key, 7), (2,)))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
key = jax.random.key(7)

a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))
print(jnp.allclose(a, b))       # True

k1, k2 = jax.random.split(key)
c = jax.random.normal(k1, (3,))
d = jax.random.normal(k2, (3,))
print(jnp.allclose(c, d))       # False

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)

print(jax.random.uniform(k1, (2, 3), minval=-1.0, maxval=1.0).shape)
print(jax.random.bernoulli(k2, 0.7, (5,)).dtype)
print(jax.random.categorical(k3, jnp.zeros((4, 5))).shape)
print(jax.random.permutation(k4, jnp.arange(6)))

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
def dropout(key, x, rate):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)

@jax.jit
def train_step(key, x, step):
    k = jax.random.fold_in(key, step)      # مفتاح جديد لكل خطوة
    return dropout(k, x, 0.2)

keys = jax.random.split(jax.random.key(0), 8)
masks = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5, (4,)))(keys)
print(masks.shape)      # (8, 4)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, numpy as np, jax.numpy as jnp

@jax.jit
def noise():
    return jnp.asarray(np.random.normal(size=3))

print(noise())
print(noise())      # نفس الأرقام تماماً

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax

k = jax.random.key(0)
print(k.dtype, k.shape)          # key ()
print(jax.random.key_data(k))    # uint32[2]

try:
    k + 1
except TypeError as exc:
    print("لا حساب على المفاتيح:", exc)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax

key = jax.random.key(42)
k1, k2 = jax.random.split(key)          # مفتاحان من مفتاح
ka, kb, kc = jax.random.split(k1, 3)    # ثلاثة من الأول

print(jax.random.normal(k1, (2,)))
print(jax.random.normal(k2, (2,)))
print(jax.random.normal(jax.random.fold_in(key, 7), (2,)))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
key = jax.random.key(7)

a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))
print(jnp.allclose(a, b))       # True

k1, k2 = jax.random.split(key)
c = jax.random.normal(k1, (3,))
d = jax.random.normal(k2, (3,))
print(jnp.allclose(c, d))       # False

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)

print(jax.random.uniform(k1, (2, 3), minval=-1.0, maxval=1.0).shape)
print(jax.random.bernoulli(k2, 0.7, (5,)).dtype)
print(jax.random.categorical(k3, jnp.zeros((4, 5))).shape)
print(jax.random.permutation(k4, jnp.arange(6)))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def dropout(key, x, rate):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)

@jax.jit
def train_step(key, x, step):
    k = jax.random.fold_in(key, step)      # مفتاح جديد لكل خطوة
    return dropout(k, x, 0.2)

keys = jax.random.split(jax.random.key(0), 8)
masks = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5, (4,)))(keys)
print(masks.shape)      # (8, 4)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — random: العشوائية الصريحة
JAX: From Zero to Production — Chapter 10 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-10-ch10 (lesson 10)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
import jax, numpy as np, jax.numpy as jnp

@jax.jit
def noise():
    return jnp.asarray(np.random.normal(size=3))

print(noise())
print(noise())      # نفس الأرقام تماماً

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
import jax

k = jax.random.key(0)
print(k.dtype, k.shape)          # key ()
print(jax.random.key_data(k))    # uint32[2]

try:
    k + 1
except TypeError as exc:
    print("لا حساب على المفاتيح:", exc)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
import jax

key = jax.random.key(42)
k1, k2 = jax.random.split(key)          # مفتاحان من مفتاح
ka, kb, kc = jax.random.split(k1, 3)    # ثلاثة من الأول

print(jax.random.normal(k1, (2,)))
print(jax.random.normal(k2, (2,)))
print(jax.random.normal(jax.random.fold_in(key, 7), (2,)))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
key = jax.random.key(7)

a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))
print(jnp.allclose(a, b))       # True

k1, k2 = jax.random.split(key)
c = jax.random.normal(k1, (3,))
d = jax.random.normal(k2, (3,))
print(jnp.allclose(c, d))       # False

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)

print(jax.random.uniform(k1, (2, 3), minval=-1.0, maxval=1.0).shape)
print(jax.random.bernoulli(k2, 0.7, (5,)).dtype)
print(jax.random.categorical(k3, jnp.zeros((4, 5))).shape)
print(jax.random.permutation(k4, jnp.arange(6)))

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch10.md
# ────────────────────────────────────────────────────────────────────────
def dropout(key, x, rate):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)

@jax.jit
def train_step(key, x, step):
    k = jax.random.fold_in(key, step)      # مفتاح جديد لكل خطوة
    return dropout(k, x, 0.2)

keys = jax.random.split(jax.random.key(0), 8)
masks = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5, (4,)))(keys)
print(masks.shape)      # (8, 4)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, numpy as np, jax.numpy as jnp

@jax.jit
def noise():
    return jnp.asarray(np.random.normal(size=3))

print(noise())
print(noise())      # نفس الأرقام تماماً

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax

k = jax.random.key(0)
print(k.dtype, k.shape)          # key ()
print(jax.random.key_data(k))    # uint32[2]

try:
    k + 1
except TypeError as exc:
    print("لا حساب على المفاتيح:", exc)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax

key = jax.random.key(42)
k1, k2 = jax.random.split(key)          # مفتاحان من مفتاح
ka, kb, kc = jax.random.split(k1, 3)    # ثلاثة من الأول

print(jax.random.normal(k1, (2,)))
print(jax.random.normal(k2, (2,)))
print(jax.random.normal(jax.random.fold_in(key, 7), (2,)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
key = jax.random.key(7)

a = jax.random.normal(key, (3,))
b = jax.random.normal(key, (3,))
print(jnp.allclose(a, b))       # True

k1, k2 = jax.random.split(key)
c = jax.random.normal(k1, (3,))
d = jax.random.normal(k2, (3,))
print(jnp.allclose(c, d))       # False

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)

print(jax.random.uniform(k1, (2, 3), minval=-1.0, maxval=1.0).shape)
print(jax.random.bernoulli(k2, 0.7, (5,)).dtype)
print(jax.random.categorical(k3, jnp.zeros((4, 5))).shape)
print(jax.random.permutation(k4, jnp.arange(6)))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-10-ch10.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def dropout(key, x, rate):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)

@jax.jit
def train_step(key, x, step):
    k = jax.random.fold_in(key, step)      # مفتاح جديد لكل خطوة
    return dropout(k, x, 0.2)

keys = jax.random.split(jax.random.key(0), 8)
masks = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5, (4,)))(keys)
print(masks.shape)      # (8, 4)
