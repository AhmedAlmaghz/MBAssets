#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن والعشرون: الدقة العددية والتكميم
JAX: From Zero to Production — bf16

Book slug : jax-from-zero-to-production
Chapter   : ch-28-ch28 (lesson 28)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-28-ch28.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن والعشرون: الدقة العددية والتكميم
JAX: From Zero to Production — bf16

Book slug : jax-from-zero-to-production
Chapter   : ch-28-ch28 (lesson 28)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-28-ch28.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن والعشرون: الدقة العددية والتكميم
JAX: From Zero to Production — bf16

Book slug : jax-from-zero-to-production
Chapter   : ch-28-ch28 (lesson 28)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 4), dtype=jnp.bfloat16)
y = x @ x.T                          # ضرب bf16 على البطاقة
print(y.dtype, y.shape)              # bfloat16 (4, 4)

# تُحسب الخسارة عادةً بـ fp32 ثم تُخفَض
loss = jnp.sum(y.astype(jnp.float32))
print(loss.dtype)                    # float32

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def to_fp8(x, scale):
    # محاكاة تكميم E4M3 مبسّطة بقصّ النطاق
    q = jnp.clip(jnp.round(x / scale), -7.0, 7.0) * scale
    return q

g = jnp.array([0.001, 0.5, 2.3, -1.7], dtype=jnp.float32)
scaled = to_fp8(g, scale=0.25)
print(scaled)                        # القيم المكمّمة

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def quantize(w, bits=8):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (w.max() - w.min()) / (qmax - qmin)
    q = jnp.clip(jnp.round(w / scale), qmin, qmax)
    return q, scale

w = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32)
q, s = quantize(w)
deq = q * s
print(float(jnp.max(jnp.abs(w - deq))))    # خطأ إزالة التكميم

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def safe_softmax(z):
    m = jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z - m)
    return e / jnp.sum(e, axis=-1, keepdims=True)

z = jnp.array([1000.0, 1001.0, 1002.0])     # بلا طرح max سيُفيض
print(safe_softmax(z))                       # توزيع مستقرّ

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 4), dtype=jnp.bfloat16)
y = x @ x.T                          # ضرب bf16 على البطاقة
print(y.dtype, y.shape)              # bfloat16 (4, 4)

# تُحسب الخسارة عادةً بـ fp32 ثم تُخفَض
loss = jnp.sum(y.astype(jnp.float32))
print(loss.dtype)                    # float32

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def to_fp8(x, scale):
    # محاكاة تكميم E4M3 مبسّطة بقصّ النطاق
    q = jnp.clip(jnp.round(x / scale), -7.0, 7.0) * scale
    return q

g = jnp.array([0.001, 0.5, 2.3, -1.7], dtype=jnp.float32)
scaled = to_fp8(g, scale=0.25)
print(scaled)                        # القيم المكمّمة

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def quantize(w, bits=8):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (w.max() - w.min()) / (qmax - qmin)
    q = jnp.clip(jnp.round(w / scale), qmin, qmax)
    return q, scale

w = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32)
q, s = quantize(w)
deq = q * s
print(float(jnp.max(jnp.abs(w - deq))))    # خطأ إزالة التكميم

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def safe_softmax(z):
    m = jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z - m)
    return e / jnp.sum(e, axis=-1, keepdims=True)

z = jnp.array([1000.0, 1001.0, 1002.0])     # بلا طرح max سيُفيض
print(safe_softmax(z))                       # توزيع مستقرّ

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن والعشرون: الدقة العددية والتكميم
JAX: From Zero to Production — bf16

Book slug : jax-from-zero-to-production
Chapter   : ch-28-ch28 (lesson 28)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 4), dtype=jnp.bfloat16)
y = x @ x.T                          # ضرب bf16 على البطاقة
print(y.dtype, y.shape)              # bfloat16 (4, 4)

# تُحسب الخسارة عادةً بـ fp32 ثم تُخفَض
loss = jnp.sum(y.astype(jnp.float32))
print(loss.dtype)                    # float32

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def to_fp8(x, scale):
    # محاكاة تكميم E4M3 مبسّطة بقصّ النطاق
    q = jnp.clip(jnp.round(x / scale), -7.0, 7.0) * scale
    return q

g = jnp.array([0.001, 0.5, 2.3, -1.7], dtype=jnp.float32)
scaled = to_fp8(g, scale=0.25)
print(scaled)                        # القيم المكمّمة

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def quantize(w, bits=8):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (w.max() - w.min()) / (qmax - qmin)
    q = jnp.clip(jnp.round(w / scale), qmin, qmax)
    return q, scale

w = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32)
q, s = quantize(w)
deq = q * s
print(float(jnp.max(jnp.abs(w - deq))))    # خطأ إزالة التكميم

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch28.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def safe_softmax(z):
    m = jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z - m)
    return e / jnp.sum(e, axis=-1, keepdims=True)

z = jnp.array([1000.0, 1001.0, 1002.0])     # بلا طرح max سيُفيض
print(safe_softmax(z))                       # توزيع مستقرّ

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.ones((4, 4), dtype=jnp.bfloat16)
y = x @ x.T                          # ضرب bf16 على البطاقة
print(y.dtype, y.shape)              # bfloat16 (4, 4)

# تُحسب الخسارة عادةً بـ fp32 ثم تُخفَض
loss = jnp.sum(y.astype(jnp.float32))
print(loss.dtype)                    # float32

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def to_fp8(x, scale):
    # محاكاة تكميم E4M3 مبسّطة بقصّ النطاق
    q = jnp.clip(jnp.round(x / scale), -7.0, 7.0) * scale
    return q

g = jnp.array([0.001, 0.5, 2.3, -1.7], dtype=jnp.float32)
scaled = to_fp8(g, scale=0.25)
print(scaled)                        # القيم المكمّمة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def quantize(w, bits=8):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (w.max() - w.min()) / (qmax - qmin)
    q = jnp.clip(jnp.round(w / scale), qmin, qmax)
    return q, scale

w = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32)
q, s = quantize(w)
deq = q * s
print(float(jnp.max(jnp.abs(w - deq))))    # خطأ إزالة التكميم

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-28-ch28.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

def safe_softmax(z):
    m = jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z - m)
    return e / jnp.sum(e, axis=-1, keepdims=True)

z = jnp.array([1000.0, 1001.0, 1002.0])     # بلا طرح max سيُفيض
print(safe_softmax(z))                       # توزيع مستقرّ
