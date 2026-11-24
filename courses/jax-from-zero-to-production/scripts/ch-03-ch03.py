#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jax.numpy: المصفوفات كما لم تعرفها
JAX: From Zero to Production — Chapter 03 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-03-ch03 (lesson 3)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-03-ch03.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jax.numpy: المصفوفات كما لم تعرفها
JAX: From Zero to Production — Chapter 03 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-03-ch03 (lesson 3)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-03-ch03.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jax.numpy: المصفوفات كما لم تعرفها
JAX: From Zero to Production — Chapter 03 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-03-ch03 (lesson 3)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

print(jnp.array(1.0).dtype)                    # float32 لا float64
print((jnp.ones(3, jnp.float32) + 1.0).dtype)  # float32: العدد الضعيف تبع المصفوفة
print((jnp.ones(3, jnp.float16) + 1.0).dtype)  # float16: بقي على عرضه
print(jnp.result_type(1.0, jnp.float32))       # float32
print(jnp.result_type(jnp.bfloat16, jnp.float32))  # float32

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.zeros(5)
y = x.at[0].set(5.0).at[2].add(1.0)

print(y)   # [5. 0. 1. 0. 0.]
print(x)   # [0. 0. 0. 0. 0.]  الأصل لم يُمسّ

idx = jnp.array([1, 3])
print(x.at[idx].set(jnp.array([7.0, 9.0])))   # تحديث بعدّة مواضع دفعةً

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

a = np.zeros(3)
np.add(a, 1.0, out=a)          # يعمل: الكتابة في المكان

b = jnp.zeros(3)
try:
    jnp.add(b, 1.0, out=b)     # لا معامل out في JAX
except TypeError as e:
    print("out:", e)

key = jax.random.key(0)        # لا حالة عالمية، بل مفتاح صريح
print(jax.random.normal(key, (2,)))
print(jax.random.normal(key, (2,)))   # القيم نفسها تماماً

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

def f(a):
    return jnp.tanh(a) * a + 1.0

x = jnp.ones((64, 64))
f(x).block_until_ready()          # تسخين: أوّل نداء يترجم

def bench(fn):
    t0 = time.perf_counter()
    for _ in range(1000):
        fn(x)
    jax.block_until_ready(fn(x))  # لا تقِس قبل أن ينتهي العمل فعلاً
    return round(time.perf_counter() - t0, 3)

print("eager:", bench(f))
print("jit  :", bench(jax.jit(f)))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

print(jnp.array(1.0).dtype)                    # float32 لا float64
print((jnp.ones(3, jnp.float32) + 1.0).dtype)  # float32: العدد الضعيف تبع المصفوفة
print((jnp.ones(3, jnp.float16) + 1.0).dtype)  # float16: بقي على عرضه
print(jnp.result_type(1.0, jnp.float32))       # float32
print(jnp.result_type(jnp.bfloat16, jnp.float32))  # float32

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.zeros(5)
y = x.at[0].set(5.0).at[2].add(1.0)

print(y)   # [5. 0. 1. 0. 0.]
print(x)   # [0. 0. 0. 0. 0.]  الأصل لم يُمسّ

idx = jnp.array([1, 3])
print(x.at[idx].set(jnp.array([7.0, 9.0])))   # تحديث بعدّة مواضع دفعةً

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

a = np.zeros(3)
np.add(a, 1.0, out=a)          # يعمل: الكتابة في المكان

b = jnp.zeros(3)
try:
    jnp.add(b, 1.0, out=b)     # لا معامل out في JAX
except TypeError as e:
    print("out:", e)

key = jax.random.key(0)        # لا حالة عالمية، بل مفتاح صريح
print(jax.random.normal(key, (2,)))
print(jax.random.normal(key, (2,)))   # القيم نفسها تماماً

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

def f(a):
    return jnp.tanh(a) * a + 1.0

x = jnp.ones((64, 64))
f(x).block_until_ready()          # تسخين: أوّل نداء يترجم

def bench(fn):
    t0 = time.perf_counter()
    for _ in range(1000):
        fn(x)
    jax.block_until_ready(fn(x))  # لا تقِس قبل أن ينتهي العمل فعلاً
    return round(time.perf_counter() - t0, 3)

print("eager:", bench(f))
print("jit  :", bench(jax.jit(f)))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — jax.numpy: المصفوفات كما لم تعرفها
JAX: From Zero to Production — Chapter 03 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-03-ch03 (lesson 3)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

print(jnp.array(1.0).dtype)                    # float32 لا float64
print((jnp.ones(3, jnp.float32) + 1.0).dtype)  # float32: العدد الضعيف تبع المصفوفة
print((jnp.ones(3, jnp.float16) + 1.0).dtype)  # float16: بقي على عرضه
print(jnp.result_type(1.0, jnp.float32))       # float32
print(jnp.result_type(jnp.bfloat16, jnp.float32))  # float32

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.zeros(5)
y = x.at[0].set(5.0).at[2].add(1.0)

print(y)   # [5. 0. 1. 0. 0.]
print(x)   # [0. 0. 0. 0. 0.]  الأصل لم يُمسّ

idx = jnp.array([1, 3])
print(x.at[idx].set(jnp.array([7.0, 9.0])))   # تحديث بعدّة مواضع دفعةً

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

a = np.zeros(3)
np.add(a, 1.0, out=a)          # يعمل: الكتابة في المكان

b = jnp.zeros(3)
try:
    jnp.add(b, 1.0, out=b)     # لا معامل out في JAX
except TypeError as e:
    print("out:", e)

key = jax.random.key(0)        # لا حالة عالمية، بل مفتاح صريح
print(jax.random.normal(key, (2,)))
print(jax.random.normal(key, (2,)))   # القيم نفسها تماماً

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch03.md
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

def f(a):
    return jnp.tanh(a) * a + 1.0

x = jnp.ones((64, 64))
f(x).block_until_ready()          # تسخين: أوّل نداء يترجم

def bench(fn):
    t0 = time.perf_counter()
    for _ in range(1000):
        fn(x)
    jax.block_until_ready(fn(x))  # لا تقِس قبل أن ينتهي العمل فعلاً
    return round(time.perf_counter() - t0, 3)

print("eager:", bench(f))
print("jit  :", bench(jax.jit(f)))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

print(jnp.array(1.0).dtype)                    # float32 لا float64
print((jnp.ones(3, jnp.float32) + 1.0).dtype)  # float32: العدد الضعيف تبع المصفوفة
print((jnp.ones(3, jnp.float16) + 1.0).dtype)  # float16: بقي على عرضه
print(jnp.result_type(1.0, jnp.float32))       # float32
print(jnp.result_type(jnp.bfloat16, jnp.float32))  # float32

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.zeros(5)
y = x.at[0].set(5.0).at[2].add(1.0)

print(y)   # [5. 0. 1. 0. 0.]
print(x)   # [0. 0. 0. 0. 0.]  الأصل لم يُمسّ

idx = jnp.array([1, 3])
print(x.at[idx].set(jnp.array([7.0, 9.0])))   # تحديث بعدّة مواضع دفعةً

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

a = np.zeros(3)
np.add(a, 1.0, out=a)          # يعمل: الكتابة في المكان

b = jnp.zeros(3)
try:
    jnp.add(b, 1.0, out=b)     # لا معامل out في JAX
except TypeError as e:
    print("out:", e)

key = jax.random.key(0)        # لا حالة عالمية، بل مفتاح صريح
print(jax.random.normal(key, (2,)))
print(jax.random.normal(key, (2,)))   # القيم نفسها تماماً

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-03-ch03.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

def f(a):
    return jnp.tanh(a) * a + 1.0

x = jnp.ones((64, 64))
f(x).block_until_ready()          # تسخين: أوّل نداء يترجم

def bench(fn):
    t0 = time.perf_counter()
    for _ in range(1000):
        fn(x)
    jax.block_until_ready(fn(x))  # لا تقِس قبل أن ينتهي العمل فعلاً
    return round(time.perf_counter() - t0, 3)

print("eager:", bench(f))
print("jit  :", bench(jax.jit(f)))
