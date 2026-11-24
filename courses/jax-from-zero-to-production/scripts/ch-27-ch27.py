#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع والعشرون: الذاكرة والأداء
JAX: From Zero to Production — remat

Book slug : jax-from-zero-to-production
Chapter   : ch-27-ch27 (lesson 27)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-27-ch27.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع والعشرون: الذاكرة والأداء
JAX: From Zero to Production — remat

Book slug : jax-from-zero-to-production
Chapter   : ch-27-ch27 (lesson 27)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-27-ch27.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع والعشرون: الذاكرة والأداء
JAX: From Zero to Production — remat

Book slug : jax-from-zero-to-production
Chapter   : ch-27-ch27 (lesson 27)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def block(x):
    for _ in range(4):
        x = jnp.tanh(x @ x.T)        # كتلة مكلفة ذاكرياً
    return x

# أعد حساب الكتلة بدل حفظ مخرجاتها الأربع
recomputed = jax.remat(block)
x = jnp.ones((512, 512))
g = jax.grad(lambda z: recomputed(z).sum())(x)
print(g.shape)                       # (512, 512)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x, w):
    y = x @ w                  # x يُتبرّع به هنا
    return y

x = jnp.ones((1024, 1024))
w = jnp.ones((1024, 64))
out = step(x, w)
print(out.shape, isinstance(out, jax.Array))     # (1024, 64) True

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# إفراغ موتر كبير إلى المضيف ثم استدعاؤه لاحقاً
big = jnp.ones((1 << 20,), dtype=jnp.float32)      # نحو 4MB
host = jax.device_put(big, jax.devices("cpu")[0])
cpu_arr = jax.device_get(host)                      # يُعاد للمضيف
print(cpu_arr.shape, type(cpu_arr))                 # (1048576,) ndarray

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, time

@jax.jit
def train_step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
train_step(x).block_until_ready()           # تسخين: استبعد التجميع

t0 = time.perf_counter()
for _ in range(30):
    train_step(x).block_until_ready()
ms = (time.perf_counter() - t0) / 30 * 1e3
print(f"{ms:.3f} ms/step")                   # زمن التشغيل الصافي

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax
from jax import memory

# تقدير ذروة تقريبي عبر حجز موتر مرجعي
ref = jax.numpy.ones((1024, 1024), dtype=jax.numpy.float32)
peak = ref.size * 4            # بايت لكل f32
print(f"peak block ~ {peak/1024:.1f} KB")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def block(x):
    for _ in range(4):
        x = jnp.tanh(x @ x.T)        # كتلة مكلفة ذاكرياً
    return x

# أعد حساب الكتلة بدل حفظ مخرجاتها الأربع
recomputed = jax.remat(block)
x = jnp.ones((512, 512))
g = jax.grad(lambda z: recomputed(z).sum())(x)
print(g.shape)                       # (512, 512)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x, w):
    y = x @ w                  # x يُتبرّع به هنا
    return y

x = jnp.ones((1024, 1024))
w = jnp.ones((1024, 64))
out = step(x, w)
print(out.shape, isinstance(out, jax.Array))     # (1024, 64) True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# إفراغ موتر كبير إلى المضيف ثم استدعاؤه لاحقاً
big = jnp.ones((1 << 20,), dtype=jnp.float32)      # نحو 4MB
host = jax.device_put(big, jax.devices("cpu")[0])
cpu_arr = jax.device_get(host)                      # يُعاد للمضيف
print(cpu_arr.shape, type(cpu_arr))                 # (1048576,) ndarray

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, time

@jax.jit
def train_step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
train_step(x).block_until_ready()           # تسخين: استبعد التجميع

t0 = time.perf_counter()
for _ in range(30):
    train_step(x).block_until_ready()
ms = (time.perf_counter() - t0) / 30 * 1e3
print(f"{ms:.3f} ms/step")                   # زمن التشغيل الصافي

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax
from jax import memory

# تقدير ذروة تقريبي عبر حجز موتر مرجعي
ref = jax.numpy.ones((1024, 1024), dtype=jax.numpy.float32)
peak = ref.size * 4            # بايت لكل f32
print(f"peak block ~ {peak/1024:.1f} KB")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السابع والعشرون: الذاكرة والأداء
JAX: From Zero to Production — remat

Book slug : jax-from-zero-to-production
Chapter   : ch-27-ch27 (lesson 27)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def block(x):
    for _ in range(4):
        x = jnp.tanh(x @ x.T)        # كتلة مكلفة ذاكرياً
    return x

# أعد حساب الكتلة بدل حفظ مخرجاتها الأربع
recomputed = jax.remat(block)
x = jnp.ones((512, 512))
g = jax.grad(lambda z: recomputed(z).sum())(x)
print(g.shape)                       # (512, 512)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x, w):
    y = x @ w                  # x يُتبرّع به هنا
    return y

x = jnp.ones((1024, 1024))
w = jnp.ones((1024, 64))
out = step(x, w)
print(out.shape, isinstance(out, jax.Array))     # (1024, 64) True

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# إفراغ موتر كبير إلى المضيف ثم استدعاؤه لاحقاً
big = jnp.ones((1 << 20,), dtype=jnp.float32)      # نحو 4MB
host = jax.device_put(big, jax.devices("cpu")[0])
cpu_arr = jax.device_get(host)                      # يُعاد للمضيف
print(cpu_arr.shape, type(cpu_arr))                 # (1048576,) ndarray

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, time

@jax.jit
def train_step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
train_step(x).block_until_ready()           # تسخين: استبعد التجميع

t0 = time.perf_counter()
for _ in range(30):
    train_step(x).block_until_ready()
ms = (time.perf_counter() - t0) / 30 * 1e3
print(f"{ms:.3f} ms/step")                   # زمن التشغيل الصافي

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch27.md
# ────────────────────────────────────────────────────────────────────────
import jax
from jax import memory

# تقدير ذروة تقريبي عبر حجز موتر مرجعي
ref = jax.numpy.ones((1024, 1024), dtype=jax.numpy.float32)
peak = ref.size * 4            # بايت لكل f32
print(f"peak block ~ {peak/1024:.1f} KB")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def block(x):
    for _ in range(4):
        x = jnp.tanh(x @ x.T)        # كتلة مكلفة ذاكرياً
    return x

# أعد حساب الكتلة بدل حفظ مخرجاتها الأربع
recomputed = jax.remat(block)
x = jnp.ones((512, 512))
g = jax.grad(lambda z: recomputed(z).sum())(x)
print(g.shape)                       # (512, 512)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x, w):
    y = x @ w                  # x يُتبرّع به هنا
    return y

x = jnp.ones((1024, 1024))
w = jnp.ones((1024, 64))
out = step(x, w)
print(out.shape, isinstance(out, jax.Array))     # (1024, 64) True

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# إفراغ موتر كبير إلى المضيف ثم استدعاؤه لاحقاً
big = jnp.ones((1 << 20,), dtype=jnp.float32)      # نحو 4MB
host = jax.device_put(big, jax.devices("cpu")[0])
cpu_arr = jax.device_get(host)                      # يُعاد للمضيف
print(cpu_arr.shape, type(cpu_arr))                 # (1048576,) ndarray

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, time

@jax.jit
def train_step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
train_step(x).block_until_ready()           # تسخين: استبعد التجميع

t0 = time.perf_counter()
for _ in range(30):
    train_step(x).block_until_ready()
ms = (time.perf_counter() - t0) / 30 * 1e3
print(f"{ms:.3f} ms/step")                   # زمن التشغيل الصافي

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-27-ch27.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax
from jax import memory

# تقدير ذروة تقريبي عبر حجز موتر مرجعي
ref = jax.numpy.ones((1024, 1024), dtype=jax.numpy.float32)
peak = ref.size * 4            # بايت لكل f32
print(f"peak block ~ {peak/1024:.1f} KB")
