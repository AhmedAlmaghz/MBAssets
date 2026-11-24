#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع عشر: التنظيم والتعميم
JAX: From Zero to Production — Dropout

Book slug : jax-from-zero-to-production
Chapter   : ch-17-ch17 (lesson 17)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-17-ch17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع عشر: التنظيم والتعميم
JAX: From Zero to Production — Dropout

Book slug : jax-from-zero-to-production
Chapter   : ch-17-ch17 (lesson 17)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-17-ch17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع عشر: التنظيم والتعميم
JAX: From Zero to Production — Dropout

Book slug : jax-from-zero-to-production
Chapter   : ch-17-ch17 (lesson 17)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def dropout(x, key, p=0.1, train=True):
    if not train or p == 0.0:
        return x
    mask = random.bernoulli(key, 1.0 - p, shape=x.shape)
    return x * mask / (1.0 - p)        # تطبيع بالإنجاز

k = random.key(0)
x = jnp.ones((4, 8))
out = dropout(x, random.split(k)[0], p=0.5, train=True)
print(float(out.sum()), x.size)        # يقلّ المجموع إحصائياً

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

def decay_only(params, wd=0.01):
    # يطبّق الاضمحلال على الأوزان ذات البعد > 1 فقط
    def f(x):
        return x * (1.0 - wd) if x.ndim > 1 else x
    return jax.tree.map(f, params)

params = {"w": jnp.ones((4, 3)), "b": jnp.full((3,), 0.5)}
new = decay_only(params, wd=0.1)
print(new["b"])        # [0.5 0.5 0.5] لم يتغيّر
print(new["w"][0])     # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)   # تدير الإحصاء
        return nn.relu(x)

model = Block()
key = jax.random.key(1)
x = jnp.ones((16, 32))
variables = model.init(key, x)
print(list(variables.keys()))        # ['params', 'batch_stats']

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def mixup(x, y, key, alpha=0.2):
    k1, k2 = random.split(key)
    lam = random.beta(k1, alpha, alpha)
    perm = random.permutation(k2, x.shape[0])
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]

x = jnp.arange(20.0).reshape(10, 2)
y = jnp.arange(10.0)
xm, ym = mixup(x, y, random.key(3))
print(xm.shape, ym.shape)        # (10, 2) (10,)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def dropout(x, key, p=0.1, train=True):
    if not train or p == 0.0:
        return x
    mask = random.bernoulli(key, 1.0 - p, shape=x.shape)
    return x * mask / (1.0 - p)        # تطبيع بالإنجاز

k = random.key(0)
x = jnp.ones((4, 8))
out = dropout(x, random.split(k)[0], p=0.5, train=True)
print(float(out.sum()), x.size)        # يقلّ المجموع إحصائياً

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

def decay_only(params, wd=0.01):
    # يطبّق الاضمحلال على الأوزان ذات البعد > 1 فقط
    def f(x):
        return x * (1.0 - wd) if x.ndim > 1 else x
    return jax.tree.map(f, params)

params = {"w": jnp.ones((4, 3)), "b": jnp.full((3,), 0.5)}
new = decay_only(params, wd=0.1)
print(new["b"])        # [0.5 0.5 0.5] لم يتغيّر
print(new["w"][0])     # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)   # تدير الإحصاء
        return nn.relu(x)

model = Block()
key = jax.random.key(1)
x = jnp.ones((16, 32))
variables = model.init(key, x)
print(list(variables.keys()))        # ['params', 'batch_stats']

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def mixup(x, y, key, alpha=0.2):
    k1, k2 = random.split(key)
    lam = random.beta(k1, alpha, alpha)
    perm = random.permutation(k2, x.shape[0])
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]

x = jnp.arange(20.0).reshape(10, 2)
y = jnp.arange(10.0)
xm, ym = mixup(x, y, random.key(3))
print(xm.shape, ym.shape)        # (10, 2) (10,)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع عشر: التنظيم والتعميم
JAX: From Zero to Production — Dropout

Book slug : jax-from-zero-to-production
Chapter   : ch-17-ch17 (lesson 17)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def dropout(x, key, p=0.1, train=True):
    if not train or p == 0.0:
        return x
    mask = random.bernoulli(key, 1.0 - p, shape=x.shape)
    return x * mask / (1.0 - p)        # تطبيع بالإنجاز

k = random.key(0)
x = jnp.ones((4, 8))
out = dropout(x, random.split(k)[0], p=0.5, train=True)
print(float(out.sum()), x.size)        # يقلّ المجموع إحصائياً

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

def decay_only(params, wd=0.01):
    # يطبّق الاضمحلال على الأوزان ذات البعد > 1 فقط
    def f(x):
        return x * (1.0 - wd) if x.ndim > 1 else x
    return jax.tree.map(f, params)

params = {"w": jnp.ones((4, 3)), "b": jnp.full((3,), 0.5)}
new = decay_only(params, wd=0.1)
print(new["b"])        # [0.5 0.5 0.5] لم يتغيّر
print(new["w"][0])     # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)   # تدير الإحصاء
        return nn.relu(x)

model = Block()
key = jax.random.key(1)
x = jnp.ones((16, 32))
variables = model.init(key, x)
print(list(variables.keys()))        # ['params', 'batch_stats']

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch17.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def mixup(x, y, key, alpha=0.2):
    k1, k2 = random.split(key)
    lam = random.beta(k1, alpha, alpha)
    perm = random.permutation(k2, x.shape[0])
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]

x = jnp.arange(20.0).reshape(10, 2)
y = jnp.arange(10.0)
xm, ym = mixup(x, y, random.key(3))
print(xm.shape, ym.shape)        # (10, 2) (10,)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def dropout(x, key, p=0.1, train=True):
    if not train or p == 0.0:
        return x
    mask = random.bernoulli(key, 1.0 - p, shape=x.shape)
    return x * mask / (1.0 - p)        # تطبيع بالإنجاز

k = random.key(0)
x = jnp.ones((4, 8))
out = dropout(x, random.split(k)[0], p=0.5, train=True)
print(float(out.sum()), x.size)        # يقلّ المجموع إحصائياً

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

def decay_only(params, wd=0.01):
    # يطبّق الاضمحلال على الأوزان ذات البعد > 1 فقط
    def f(x):
        return x * (1.0 - wd) if x.ndim > 1 else x
    return jax.tree.map(f, params)

params = {"w": jnp.ones((4, 3)), "b": jnp.full((3,), 0.5)}
new = decay_only(params, wd=0.1)
print(new["b"])        # [0.5 0.5 0.5] لم يتغيّر
print(new["w"][0])     # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)   # تدير الإحصاء
        return nn.relu(x)

model = Block()
key = jax.random.key(1)
x = jnp.ones((16, 32))
variables = model.init(key, x)
print(list(variables.keys()))        # ['params', 'batch_stats']

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-17-ch17.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import random

def mixup(x, y, key, alpha=0.2):
    k1, k2 = random.split(key)
    lam = random.beta(k1, alpha, alpha)
    perm = random.permutation(k2, x.shape[0])
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]

x = jnp.arange(20.0).reshape(10, 2)
y = jnp.arange(10.0)
xm, ym = mixup(x, y, random.key(3))
print(xm.shape, ym.shape)        # (10, 2) (10,)
