#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الاختبار والحتمية وتشخيص الأخطاء
JAX: From Zero to Production — Chapter 30 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-30-ch30 (lesson 30)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-30-ch30.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الاختبار والحتمية وتشخيص الأخطاء
JAX: From Zero to Production — Chapter 30 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-30-ch30 (lesson 30)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-30-ch30.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الاختبار والحتمية وتشخيص الأخطاء
JAX: From Zero to Production — Chapter 30 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-30-ch30 (lesson 30)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax
# فرض حتمية أكبر على حسابات الطليعة، مع ثمن أداء محتمل
jax.config.update("jax_deterministic", True)

key = jax.random.key(0)
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (1000,))
b = jax.random.normal(k2, (1000,))
print(a[0], b[0])                 # نفس القيمتين في كل تشغيل

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)
# صحيح: اقسم ثم استخدم الفروع
k_drop, k_init = jax.random.split(key)
mask = jax.random.bernoulli(k_drop, p=0.1, shape=(8,))
w = jax.random.normal(k_init, (8, 8))

# خاطئ: إعادة نفس المفتاح لمهمتين مختلفتين
bad = jax.random.normal(key, (8, 8))      # لا تفعل هذا بعد split أعلاه

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
keys = jax.random.split(key, 4)
for i in range(4):
    ki = jax.random.fold_in(key, i)        # مستقلّ لكل i، بلا شجرة split
    print(jax.random.uniform(ki).item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import test_util as jtu

def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

def test_loss_shape_and_value():
    params = {"w": jnp.ones((3, 1)), "b": jnp.zeros((1,))}
    x = jnp.ones((4, 3))
    y = jnp.ones((4, 1))
    out = loss(params, x, y)              # (4,1) قبل المتوسط
    jtu.check_close(out, jnp.zeros((4, 1)))   # كل تنبؤ يساوي الهدف
    assert loss(params, x, y).shape == ()      # الخسارة عدد مفرد

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("x shape={x.shape} min={xmin}", x=x, xmin=jnp.min(x))
    return x * 2

f(jnp.ones((3,)))            # تطبع الشكل والحدّ الأدنى الملموس

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, jax.profiler

@jax.jit
def step(x):
    jax.debug.print("nan? {n}", n=jnp.any(jnp.isnan(x)))
    return jnp.tanh(x @ x.T)

x = jnp.ones((512, 512))
step(x).block_until_ready()          # تسخين

jax.profiler.start_trace("/tmp/trace")
for _ in range(10):
    step(x).block_until_ready()
jax.profiler.stop_trace()            # افتح الأثر في xprof

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
# فرض حتمية أكبر على حسابات الطليعة، مع ثمن أداء محتمل
jax.config.update("jax_deterministic", True)

key = jax.random.key(0)
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (1000,))
b = jax.random.normal(k2, (1000,))
print(a[0], b[0])                 # نفس القيمتين في كل تشغيل

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)
# صحيح: اقسم ثم استخدم الفروع
k_drop, k_init = jax.random.split(key)
mask = jax.random.bernoulli(k_drop, p=0.1, shape=(8,))
w = jax.random.normal(k_init, (8, 8))

# خاطئ: إعادة نفس المفتاح لمهمتين مختلفتين
bad = jax.random.normal(key, (8, 8))      # لا تفعل هذا بعد split أعلاه

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
keys = jax.random.split(key, 4)
for i in range(4):
    ki = jax.random.fold_in(key, i)        # مستقلّ لكل i، بلا شجرة split
    print(jax.random.uniform(ki).item())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import test_util as jtu

def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

def test_loss_shape_and_value():
    params = {"w": jnp.ones((3, 1)), "b": jnp.zeros((1,))}
    x = jnp.ones((4, 3))
    y = jnp.ones((4, 1))
    out = loss(params, x, y)              # (4,1) قبل المتوسط
    jtu.check_close(out, jnp.zeros((4, 1)))   # كل تنبؤ يساوي الهدف
    assert loss(params, x, y).shape == ()      # الخسارة عدد مفرد

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("x shape={x.shape} min={xmin}", x=x, xmin=jnp.min(x))
    return x * 2

f(jnp.ones((3,)))            # تطبع الشكل والحدّ الأدنى الملموس

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, jax.profiler

@jax.jit
def step(x):
    jax.debug.print("nan? {n}", n=jnp.any(jnp.isnan(x)))
    return jnp.tanh(x @ x.T)

x = jnp.ones((512, 512))
step(x).block_until_ready()          # تسخين

jax.profiler.start_trace("/tmp/trace")
for _ in range(10):
    step(x).block_until_ready()
jax.profiler.stop_trace()            # افتح الأثر في xprof

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الاختبار والحتمية وتشخيص الأخطاء
JAX: From Zero to Production — Chapter 30 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-30-ch30 (lesson 30)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax
# فرض حتمية أكبر على حسابات الطليعة، مع ثمن أداء محتمل
jax.config.update("jax_deterministic", True)

key = jax.random.key(0)
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (1000,))
b = jax.random.normal(k2, (1000,))
print(a[0], b[0])                 # نفس القيمتين في كل تشغيل

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)
# صحيح: اقسم ثم استخدم الفروع
k_drop, k_init = jax.random.split(key)
mask = jax.random.bernoulli(k_drop, p=0.1, shape=(8,))
w = jax.random.normal(k_init, (8, 8))

# خاطئ: إعادة نفس المفتاح لمهمتين مختلفتين
bad = jax.random.normal(key, (8, 8))      # لا تفعل هذا بعد split أعلاه

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
keys = jax.random.split(key, 4)
for i in range(4):
    ki = jax.random.fold_in(key, i)        # مستقلّ لكل i، بلا شجرة split
    print(jax.random.uniform(ki).item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import test_util as jtu

def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

def test_loss_shape_and_value():
    params = {"w": jnp.ones((3, 1)), "b": jnp.zeros((1,))}
    x = jnp.ones((4, 3))
    y = jnp.ones((4, 1))
    out = loss(params, x, y)              # (4,1) قبل المتوسط
    jtu.check_close(out, jnp.zeros((4, 1)))   # كل تنبؤ يساوي الهدف
    assert loss(params, x, y).shape == ()      # الخسارة عدد مفرد

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("x shape={x.shape} min={xmin}", x=x, xmin=jnp.min(x))
    return x * 2

f(jnp.ones((3,)))            # تطبع الشكل والحدّ الأدنى الملموس

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch30.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, jax.profiler

@jax.jit
def step(x):
    jax.debug.print("nan? {n}", n=jnp.any(jnp.isnan(x)))
    return jnp.tanh(x @ x.T)

x = jnp.ones((512, 512))
step(x).block_until_ready()          # تسخين

jax.profiler.start_trace("/tmp/trace")
for _ in range(10):
    step(x).block_until_ready()
jax.profiler.stop_trace()            # افتح الأثر في xprof

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
# فرض حتمية أكبر على حسابات الطليعة، مع ثمن أداء محتمل
jax.config.update("jax_deterministic", True)

key = jax.random.key(0)
k1, k2 = jax.random.split(key)
a = jax.random.normal(k1, (1000,))
b = jax.random.normal(k2, (1000,))
print(a[0], b[0])                 # نفس القيمتين في كل تشغيل

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

key = jax.random.key(42)
# صحيح: اقسم ثم استخدم الفروع
k_drop, k_init = jax.random.split(key)
mask = jax.random.bernoulli(k_drop, p=0.1, shape=(8,))
w = jax.random.normal(k_init, (8, 8))

# خاطئ: إعادة نفس المفتاح لمهمتين مختلفتين
bad = jax.random.normal(key, (8, 8))      # لا تفعل هذا بعد split أعلاه

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
keys = jax.random.split(key, 4)
for i in range(4):
    ki = jax.random.fold_in(key, i)        # مستقلّ لكل i، بلا شجرة split
    print(jax.random.uniform(ki).item())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import test_util as jtu

def loss(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

def test_loss_shape_and_value():
    params = {"w": jnp.ones((3, 1)), "b": jnp.zeros((1,))}
    x = jnp.ones((4, 3))
    y = jnp.ones((4, 1))
    out = loss(params, x, y)              # (4,1) قبل المتوسط
    jtu.check_close(out, jnp.zeros((4, 1)))   # كل تنبؤ يساوي الهدف
    assert loss(params, x, y).shape == ()      # الخسارة عدد مفرد

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def f(x):
    jax.debug.print("x shape={x.shape} min={xmin}", x=x, xmin=jnp.min(x))
    return x * 2

f(jnp.ones((3,)))            # تطبع الشكل والحدّ الأدنى الملموس

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-30-ch30.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, jax.profiler

@jax.jit
def step(x):
    jax.debug.print("nan? {n}", n=jnp.any(jnp.isnan(x)))
    return jnp.tanh(x @ x.T)

x = jnp.ones((512, 512))
step(x).block_until_ready()          # تسخين

jax.profiler.start_trace("/tmp/trace")
for _ in range(10):
    step(x).block_until_ready()
jax.profiler.stop_trace()            # افتح الأثر في xprof
