#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jit: التجميع الذي يغيّر كل شيء
JAX: From Zero to Production — Chapter 06 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-06-ch06 (lesson 6)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-06-ch06.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jit: التجميع الذي يغيّر كل شيء
JAX: From Zero to Production — Chapter 06 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-06-ch06 (lesson 6)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-06-ch06.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jit: التجميع الذي يغيّر كل شيء
JAX: From Zero to Production — Chapter 06 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-06-ch06 (lesson 6)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import time, jax, jax.numpy as jnp

def block(x, w):
    for _ in range(20):
        x = jnp.tanh(x @ w)          # حلقة بايثون تُفكّ عند التتبّع
    return jnp.sum(x)

fast = jax.jit(block)
x = w = jnp.ones((256, 256)) * 0.01
t0 = time.perf_counter(); fast(x, w).block_until_ready()
t1 = time.perf_counter(); fast(x, w).block_until_ready()
print(f"first {(t1-t0)*1e3:.0f} ms, steady {(time.perf_counter()-t1)*1e3:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    print("tracing with", x.shape)  # مرّة واحدة فقط
    return jnp.sum(x * 2.0)

f(jnp.ones(3)); f(jnp.ones(3)); f(jnp.ones(3))

@jax.jit
def g(x):
    jax.debug.print("runtime: {v}", v=jnp.sum(x))   # وقت التنفيذ فعلاً
    return x

g(jnp.arange(4.0)); g(jnp.arange(4.0))

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def mlp(x, w, depth, act):
    for i in range(depth):          # depth ثابت ⇒ الحلقة تُفكّ
        x = act(x @ w[i])
    return x

fast = jax.jit(mlp, static_argnums=(2, 3))
ws = [jnp.ones((4, 4)) * 0.5 for _ in range(3)]

fast(jnp.ones((2, 4)), ws, 3, jnp.tanh)
print(fast._cache_size())           # 1
fast(jnp.ones((2, 4)), ws, 2, jnp.tanh)
print(fast._cache_size())           # 2

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

BUCKETS = (16, 32, 64, 128)

def pad_batch(t, buckets=BUCKETS):
    L = next(b for b in buckets if t.shape[1] <= b)
    return jnp.pad(t, ((0, 0), (0, L - t.shape[1])))

@jax.jit
def total(t):
    return jnp.sum(t.astype(jnp.float32))

for raw in (jnp.ones((2, 11)), jnp.ones((2, 30)), jnp.ones((2, 60))):
    print(raw.shape, "->", pad_batch(raw).shape)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=(0,))
def step(w, x):
    return w - 0.01 * (w @ x)

w = x = jnp.ones((512, 512)) * 0.001
w2 = step(w, x)          # w استُهلكت: لا تقرأها بعد الآن

# AOT: ترجمة قبل أن تصل البيانات
compiled = jax.jit(lambda a, b: a - 0.01 * (a @ b)).lower(w2, x).compile()
print(compiled(w2, x).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import time, jax, jax.numpy as jnp

def block(x, w):
    for _ in range(20):
        x = jnp.tanh(x @ w)          # حلقة بايثون تُفكّ عند التتبّع
    return jnp.sum(x)

fast = jax.jit(block)
x = w = jnp.ones((256, 256)) * 0.01
t0 = time.perf_counter(); fast(x, w).block_until_ready()
t1 = time.perf_counter(); fast(x, w).block_until_ready()
print(f"first {(t1-t0)*1e3:.0f} ms, steady {(time.perf_counter()-t1)*1e3:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    print("tracing with", x.shape)  # مرّة واحدة فقط
    return jnp.sum(x * 2.0)

f(jnp.ones(3)); f(jnp.ones(3)); f(jnp.ones(3))

@jax.jit
def g(x):
    jax.debug.print("runtime: {v}", v=jnp.sum(x))   # وقت التنفيذ فعلاً
    return x

g(jnp.arange(4.0)); g(jnp.arange(4.0))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def mlp(x, w, depth, act):
    for i in range(depth):          # depth ثابت ⇒ الحلقة تُفكّ
        x = act(x @ w[i])
    return x

fast = jax.jit(mlp, static_argnums=(2, 3))
ws = [jnp.ones((4, 4)) * 0.5 for _ in range(3)]

fast(jnp.ones((2, 4)), ws, 3, jnp.tanh)
print(fast._cache_size())           # 1
fast(jnp.ones((2, 4)), ws, 2, jnp.tanh)
print(fast._cache_size())           # 2

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

BUCKETS = (16, 32, 64, 128)

def pad_batch(t, buckets=BUCKETS):
    L = next(b for b in buckets if t.shape[1] <= b)
    return jnp.pad(t, ((0, 0), (0, L - t.shape[1])))

@jax.jit
def total(t):
    return jnp.sum(t.astype(jnp.float32))

for raw in (jnp.ones((2, 11)), jnp.ones((2, 30)), jnp.ones((2, 60))):
    print(raw.shape, "->", pad_batch(raw).shape)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=(0,))
def step(w, x):
    return w - 0.01 * (w @ x)

w = x = jnp.ones((512, 512)) * 0.001
w2 = step(w, x)          # w استُهلكت: لا تقرأها بعد الآن

# AOT: ترجمة قبل أن تصل البيانات
compiled = jax.jit(lambda a, b: a - 0.01 * (a @ b)).lower(w2, x).compile()
print(compiled(w2, x).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jit: التجميع الذي يغيّر كل شيء
JAX: From Zero to Production — Chapter 06 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-06-ch06 (lesson 6)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import time, jax, jax.numpy as jnp

def block(x, w):
    for _ in range(20):
        x = jnp.tanh(x @ w)          # حلقة بايثون تُفكّ عند التتبّع
    return jnp.sum(x)

fast = jax.jit(block)
x = w = jnp.ones((256, 256)) * 0.01
t0 = time.perf_counter(); fast(x, w).block_until_ready()
t1 = time.perf_counter(); fast(x, w).block_until_ready()
print(f"first {(t1-t0)*1e3:.0f} ms, steady {(time.perf_counter()-t1)*1e3:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    print("tracing with", x.shape)  # مرّة واحدة فقط
    return jnp.sum(x * 2.0)

f(jnp.ones(3)); f(jnp.ones(3)); f(jnp.ones(3))

@jax.jit
def g(x):
    jax.debug.print("runtime: {v}", v=jnp.sum(x))   # وقت التنفيذ فعلاً
    return x

g(jnp.arange(4.0)); g(jnp.arange(4.0))

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def mlp(x, w, depth, act):
    for i in range(depth):          # depth ثابت ⇒ الحلقة تُفكّ
        x = act(x @ w[i])
    return x

fast = jax.jit(mlp, static_argnums=(2, 3))
ws = [jnp.ones((4, 4)) * 0.5 for _ in range(3)]

fast(jnp.ones((2, 4)), ws, 3, jnp.tanh)
print(fast._cache_size())           # 1
fast(jnp.ones((2, 4)), ws, 2, jnp.tanh)
print(fast._cache_size())           # 2

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

BUCKETS = (16, 32, 64, 128)

def pad_batch(t, buckets=BUCKETS):
    L = next(b for b in buckets if t.shape[1] <= b)
    return jnp.pad(t, ((0, 0), (0, L - t.shape[1])))

@jax.jit
def total(t):
    return jnp.sum(t.astype(jnp.float32))

for raw in (jnp.ones((2, 11)), jnp.ones((2, 30)), jnp.ones((2, 60))):
    print(raw.shape, "->", pad_batch(raw).shape)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch06.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=(0,))
def step(w, x):
    return w - 0.01 * (w @ x)

w = x = jnp.ones((512, 512)) * 0.001
w2 = step(w, x)          # w استُهلكت: لا تقرأها بعد الآن

# AOT: ترجمة قبل أن تصل البيانات
compiled = jax.jit(lambda a, b: a - 0.01 * (a @ b)).lower(w2, x).compile()
print(compiled(w2, x).shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time, jax, jax.numpy as jnp

def block(x, w):
    for _ in range(20):
        x = jnp.tanh(x @ w)          # حلقة بايثون تُفكّ عند التتبّع
    return jnp.sum(x)

fast = jax.jit(block)
x = w = jnp.ones((256, 256)) * 0.01
t0 = time.perf_counter(); fast(x, w).block_until_ready()
t1 = time.perf_counter(); fast(x, w).block_until_ready()
print(f"first {(t1-t0)*1e3:.0f} ms, steady {(time.perf_counter()-t1)*1e3:.1f} ms")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    print("tracing with", x.shape)  # مرّة واحدة فقط
    return jnp.sum(x * 2.0)

f(jnp.ones(3)); f(jnp.ones(3)); f(jnp.ones(3))

@jax.jit
def g(x):
    jax.debug.print("runtime: {v}", v=jnp.sum(x))   # وقت التنفيذ فعلاً
    return x

g(jnp.arange(4.0)); g(jnp.arange(4.0))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def mlp(x, w, depth, act):
    for i in range(depth):          # depth ثابت ⇒ الحلقة تُفكّ
        x = act(x @ w[i])
    return x

fast = jax.jit(mlp, static_argnums=(2, 3))
ws = [jnp.ones((4, 4)) * 0.5 for _ in range(3)]

fast(jnp.ones((2, 4)), ws, 3, jnp.tanh)
print(fast._cache_size())           # 1
fast(jnp.ones((2, 4)), ws, 2, jnp.tanh)
print(fast._cache_size())           # 2

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

BUCKETS = (16, 32, 64, 128)

def pad_batch(t, buckets=BUCKETS):
    L = next(b for b in buckets if t.shape[1] <= b)
    return jnp.pad(t, ((0, 0), (0, L - t.shape[1])))

@jax.jit
def total(t):
    return jnp.sum(t.astype(jnp.float32))

for raw in (jnp.ones((2, 11)), jnp.ones((2, 30)), jnp.ones((2, 60))):
    print(raw.shape, "->", pad_batch(raw).shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-06-ch06.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=(0,))
def step(w, x):
    return w - 0.01 * (w @ x)

w = x = jnp.ones((512, 512)) * 0.001
w2 = step(w, x)          # w استُهلكت: لا تقرأها بعد الآن

# AOT: ترجمة قبل أن تصل البيانات
compiled = jax.jit(lambda a, b: a - 0.01 * (a @ b)).lower(w2, x).compile()
print(compiled(w2, x).shape)
