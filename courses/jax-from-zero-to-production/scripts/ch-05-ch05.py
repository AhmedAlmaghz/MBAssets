#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأجهزة والذاكرة: jax.Array
JAX: From Zero to Production — Chapter 05 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-05-ch05 (lesson 5)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-05-ch05.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأجهزة والذاكرة: jax.Array
JAX: From Zero to Production — Chapter 05 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-05-ch05 (lesson 5)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-05-ch05.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأجهزة والذاكرة: jax.Array
JAX: From Zero to Production — Chapter 05 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-05-ch05 (lesson 5)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 8), dtype=jnp.bfloat16)
print(x.shape, x.dtype, x.nbytes)   # (4, 8) bfloat16 64
print(x.device)                     # cuda:0
print(x.sharding)                   # SingleDeviceSharding
print(x.aval)                       # ShapedArray(bfloat16[4,8], weak_type=False)
print(jax.default_backend(), jax.devices())

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

host = np.ones((4096, 4096), dtype=np.float32)   # 64 MiB على المضيف
x = jax.device_put(host, jax.devices()[0])       # نسخة واحدة صريحة

y = jnp.ones((4096, 4096))                       # تُنشأ على الجهاز
print(x.devices(), y.devices())
print(jax.device_get(y).shape)                   # عودة إلى المضيف

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

x = jnp.ones((4096, 4096))
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready()          # تسخين قبل أي قياس

t0 = time.perf_counter()
f(x)                              # يعود فوراً
print("بلا انتظار:", round(time.perf_counter() - t0, 4))

t0 = time.perf_counter()
f(x).block_until_ready()          # هنا يحدث الحساب فعلاً
print("مع انتظار :", round(time.perf_counter() - t0, 4))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=0)
def step(w, g):
    return w - 0.01 * g          # الخرج يأخذ مخزن w

w = jnp.ones(1_000_000)
w = step(w, jnp.ones(1_000_000)) # إعادة الإسناد آمنة
print(w.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 8), dtype=jnp.bfloat16)
print(x.shape, x.dtype, x.nbytes)   # (4, 8) bfloat16 64
print(x.device)                     # cuda:0
print(x.sharding)                   # SingleDeviceSharding
print(x.aval)                       # ShapedArray(bfloat16[4,8], weak_type=False)
print(jax.default_backend(), jax.devices())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

host = np.ones((4096, 4096), dtype=np.float32)   # 64 MiB على المضيف
x = jax.device_put(host, jax.devices()[0])       # نسخة واحدة صريحة

y = jnp.ones((4096, 4096))                       # تُنشأ على الجهاز
print(x.devices(), y.devices())
print(jax.device_get(y).shape)                   # عودة إلى المضيف

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

x = jnp.ones((4096, 4096))
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready()          # تسخين قبل أي قياس

t0 = time.perf_counter()
f(x)                              # يعود فوراً
print("بلا انتظار:", round(time.perf_counter() - t0, 4))

t0 = time.perf_counter()
f(x).block_until_ready()          # هنا يحدث الحساب فعلاً
print("مع انتظار :", round(time.perf_counter() - t0, 4))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=0)
def step(w, g):
    return w - 0.01 * g          # الخرج يأخذ مخزن w

w = jnp.ones(1_000_000)
w = step(w, jnp.ones(1_000_000)) # إعادة الإسناد آمنة
print(w.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأجهزة والذاكرة: jax.Array
JAX: From Zero to Production — Chapter 05 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-05-ch05 (lesson 5)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 8), dtype=jnp.bfloat16)
print(x.shape, x.dtype, x.nbytes)   # (4, 8) bfloat16 64
print(x.device)                     # cuda:0
print(x.sharding)                   # SingleDeviceSharding
print(x.aval)                       # ShapedArray(bfloat16[4,8], weak_type=False)
print(jax.default_backend(), jax.devices())

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

host = np.ones((4096, 4096), dtype=np.float32)   # 64 MiB على المضيف
x = jax.device_put(host, jax.devices()[0])       # نسخة واحدة صريحة

y = jnp.ones((4096, 4096))                       # تُنشأ على الجهاز
print(x.devices(), y.devices())
print(jax.device_get(y).shape)                   # عودة إلى المضيف

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

x = jnp.ones((4096, 4096))
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready()          # تسخين قبل أي قياس

t0 = time.perf_counter()
f(x)                              # يعود فوراً
print("بلا انتظار:", round(time.perf_counter() - t0, 4))

t0 = time.perf_counter()
f(x).block_until_ready()          # هنا يحدث الحساب فعلاً
print("مع انتظار :", round(time.perf_counter() - t0, 4))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch05.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=0)
def step(w, g):
    return w - 0.01 * g          # الخرج يأخذ مخزن w

w = jnp.ones(1_000_000)
w = step(w, jnp.ones(1_000_000)) # إعادة الإسناد آمنة
print(w.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 8), dtype=jnp.bfloat16)
print(x.shape, x.dtype, x.nbytes)   # (4, 8) bfloat16 64
print(x.device)                     # cuda:0
print(x.sharding)                   # SingleDeviceSharding
print(x.aval)                       # ShapedArray(bfloat16[4,8], weak_type=False)
print(jax.default_backend(), jax.devices())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import jax, jax.numpy as jnp

host = np.ones((4096, 4096), dtype=np.float32)   # 64 MiB على المضيف
x = jax.device_put(host, jax.devices()[0])       # نسخة واحدة صريحة

y = jnp.ones((4096, 4096))                       # تُنشأ على الجهاز
print(x.devices(), y.devices())
print(jax.device_get(y).shape)                   # عودة إلى المضيف

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
import jax, jax.numpy as jnp

x = jnp.ones((4096, 4096))
f = jax.jit(lambda a: a @ a)
f(x).block_until_ready()          # تسخين قبل أي قياس

t0 = time.perf_counter()
f(x)                              # يعود فوراً
print("بلا انتظار:", round(time.perf_counter() - t0, 4))

t0 = time.perf_counter()
f(x).block_until_ready()          # هنا يحدث الحساب فعلاً
print("مع انتظار :", round(time.perf_counter() - t0, 4))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-05-ch05.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit(donate_argnums=0)
def step(w, g):
    return w - 0.01 * g          # الخرج يأخذ مخزن w

w = jnp.ones(1_000_000)
w = step(w, jnp.ones(1_000_000)) # إعادة الإسناد آمنة
print(w.shape)
