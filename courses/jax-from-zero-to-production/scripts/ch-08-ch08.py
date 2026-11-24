#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — vmap: الدُفعات بلا حلقات
JAX: From Zero to Production — Chapter 08 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-08-ch08 (lesson 8)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-08-ch08.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — vmap: الدُفعات بلا حلقات
JAX: From Zero to Production — Chapter 08 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-08-ch08 (lesson 8)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-08-ch08.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — vmap: الدُفعات بلا حلقات
JAX: From Zero to Production — Chapter 08 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-08-ch08 (lesson 8)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def score(w, x):                 # عيّنة واحدة
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.arange(32.0).reshape(8, 4)

loop = jnp.stack([score(w, x) for x in batch])   # حلقة بايثون
vec = jax.vmap(score, in_axes=(None, 0))(w, batch)
print(loop.shape, jnp.allclose(loop, vec))       # (8,) True

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(w, x):
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.ones((8, 4))

# الدفعة في البعد الثاني من x، والأوزان مشتركة
xt = jnp.ones((4, 8))
out = jax.vmap(f, in_axes=(None, 1))(w, xt)
print(out.shape)                                  # (8,)

# الدفعة في البعد الأول، والخرج في بعد جديد
out2 = jax.vmap(f, in_axes=(None, 0), out_axes=1)(w, batch)
print(out2.shape)                                 # (8,)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def step(params, batch):
    h = jnp.tanh(batch["x"] @ params["w"] + params["b"])
    return jnp.sum(h * batch["mask"])

params = {"w": jnp.ones((4, 2)) * 0.1, "b": jnp.zeros(2)}
batch = {"x": jnp.ones((8, 4)), "mask": jnp.ones((8, 2))}

out = jax.vmap(step, in_axes=(None, 0))(params, batch)
print(out.shape)      # (8,)

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def noise(key, shape):
    return jax.random.normal(key, shape)

key = jax.random.key(0)
keys = jax.random.split(key, 8)          # ثمانية مفاتيح مستقلة

draws = jax.vmap(noise, in_axes=(0, None))(keys, (3,))
print(draws.shape)                       # (8, 3)
print(jnp.allclose(draws[0], draws[1]))  # False

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.sum((w * x) ** 2)

w = jnp.ones(4)
batch = jnp.ones((8, 4)) * 2.0

per_example = jax.vmap(jax.grad(loss), in_axes=(None, 0))(w, batch)
total = jax.grad(lambda w: jnp.mean(jax.vmap(loss, in_axes=(None, 0))(w, batch)))(w)

print(per_example.shape)   # (8, 4)
print(total.shape)         # (4,)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def score(w, x):                 # عيّنة واحدة
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.arange(32.0).reshape(8, 4)

loop = jnp.stack([score(w, x) for x in batch])   # حلقة بايثون
vec = jax.vmap(score, in_axes=(None, 0))(w, batch)
print(loop.shape, jnp.allclose(loop, vec))       # (8,) True

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(w, x):
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.ones((8, 4))

# الدفعة في البعد الثاني من x، والأوزان مشتركة
xt = jnp.ones((4, 8))
out = jax.vmap(f, in_axes=(None, 1))(w, xt)
print(out.shape)                                  # (8,)

# الدفعة في البعد الأول، والخرج في بعد جديد
out2 = jax.vmap(f, in_axes=(None, 0), out_axes=1)(w, batch)
print(out2.shape)                                 # (8,)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def step(params, batch):
    h = jnp.tanh(batch["x"] @ params["w"] + params["b"])
    return jnp.sum(h * batch["mask"])

params = {"w": jnp.ones((4, 2)) * 0.1, "b": jnp.zeros(2)}
batch = {"x": jnp.ones((8, 4)), "mask": jnp.ones((8, 2))}

out = jax.vmap(step, in_axes=(None, 0))(params, batch)
print(out.shape)      # (8,)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def noise(key, shape):
    return jax.random.normal(key, shape)

key = jax.random.key(0)
keys = jax.random.split(key, 8)          # ثمانية مفاتيح مستقلة

draws = jax.vmap(noise, in_axes=(0, None))(keys, (3,))
print(draws.shape)                       # (8, 3)
print(jnp.allclose(draws[0], draws[1]))  # False

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.sum((w * x) ** 2)

w = jnp.ones(4)
batch = jnp.ones((8, 4)) * 2.0

per_example = jax.vmap(jax.grad(loss), in_axes=(None, 0))(w, batch)
total = jax.grad(lambda w: jnp.mean(jax.vmap(loss, in_axes=(None, 0))(w, batch)))(w)

print(per_example.shape)   # (8, 4)
print(total.shape)         # (4,)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — vmap: الدُفعات بلا حلقات
JAX: From Zero to Production — Chapter 08 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-08-ch08 (lesson 8)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def score(w, x):                 # عيّنة واحدة
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.arange(32.0).reshape(8, 4)

loop = jnp.stack([score(w, x) for x in batch])   # حلقة بايثون
vec = jax.vmap(score, in_axes=(None, 0))(w, batch)
print(loop.shape, jnp.allclose(loop, vec))       # (8,) True

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(w, x):
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.ones((8, 4))

# الدفعة في البعد الثاني من x، والأوزان مشتركة
xt = jnp.ones((4, 8))
out = jax.vmap(f, in_axes=(None, 1))(w, xt)
print(out.shape)                                  # (8,)

# الدفعة في البعد الأول، والخرج في بعد جديد
out2 = jax.vmap(f, in_axes=(None, 0), out_axes=1)(w, batch)
print(out2.shape)                                 # (8,)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def step(params, batch):
    h = jnp.tanh(batch["x"] @ params["w"] + params["b"])
    return jnp.sum(h * batch["mask"])

params = {"w": jnp.ones((4, 2)) * 0.1, "b": jnp.zeros(2)}
batch = {"x": jnp.ones((8, 4)), "mask": jnp.ones((8, 2))}

out = jax.vmap(step, in_axes=(None, 0))(params, batch)
print(out.shape)      # (8,)

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def noise(key, shape):
    return jax.random.normal(key, shape)

key = jax.random.key(0)
keys = jax.random.split(key, 8)          # ثمانية مفاتيح مستقلة

draws = jax.vmap(noise, in_axes=(0, None))(keys, (3,))
print(draws.shape)                       # (8, 3)
print(jnp.allclose(draws[0], draws[1]))  # False

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch08.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.sum((w * x) ** 2)

w = jnp.ones(4)
batch = jnp.ones((8, 4)) * 2.0

per_example = jax.vmap(jax.grad(loss), in_axes=(None, 0))(w, batch)
total = jax.grad(lambda w: jnp.mean(jax.vmap(loss, in_axes=(None, 0))(w, batch)))(w)

print(per_example.shape)   # (8, 4)
print(total.shape)         # (4,)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def score(w, x):                 # عيّنة واحدة
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.arange(32.0).reshape(8, 4)

loop = jnp.stack([score(w, x) for x in batch])   # حلقة بايثون
vec = jax.vmap(score, in_axes=(None, 0))(w, batch)
print(loop.shape, jnp.allclose(loop, vec))       # (8,) True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(w, x):
    return jnp.dot(w, x)

w = jnp.ones(4)
batch = jnp.ones((8, 4))

# الدفعة في البعد الثاني من x، والأوزان مشتركة
xt = jnp.ones((4, 8))
out = jax.vmap(f, in_axes=(None, 1))(w, xt)
print(out.shape)                                  # (8,)

# الدفعة في البعد الأول، والخرج في بعد جديد
out2 = jax.vmap(f, in_axes=(None, 0), out_axes=1)(w, batch)
print(out2.shape)                                 # (8,)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def step(params, batch):
    h = jnp.tanh(batch["x"] @ params["w"] + params["b"])
    return jnp.sum(h * batch["mask"])

params = {"w": jnp.ones((4, 2)) * 0.1, "b": jnp.zeros(2)}
batch = {"x": jnp.ones((8, 4)), "mask": jnp.ones((8, 2))}

out = jax.vmap(step, in_axes=(None, 0))(params, batch)
print(out.shape)      # (8,)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def noise(key, shape):
    return jax.random.normal(key, shape)

key = jax.random.key(0)
keys = jax.random.split(key, 8)          # ثمانية مفاتيح مستقلة

draws = jax.vmap(noise, in_axes=(0, None))(keys, (3,))
print(draws.shape)                       # (8, 3)
print(jnp.allclose(draws[0], draws[1]))  # False

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-08-ch08.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.sum((w * x) ** 2)

w = jnp.ones(4)
batch = jnp.ones((8, 4)) * 2.0

per_example = jax.vmap(jax.grad(loss), in_axes=(None, 0))(w, batch)
total = jax.grad(lambda w: jnp.mean(jax.vmap(loss, in_axes=(None, 0))(w, batch)))(w)

print(per_example.shape)   # (8, 4)
print(total.shape)         # (4,)
