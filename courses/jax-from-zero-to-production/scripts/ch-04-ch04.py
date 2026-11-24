#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأشكال والبثّ والفهرسة المتقدّمة
JAX: From Zero to Production — einsum

Book slug : jax-from-zero-to-production
Chapter   : ch-04-ch04 (lesson 4)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-04-ch04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأشكال والبثّ والفهرسة المتقدّمة
JAX: From Zero to Production — einsum

Book slug : jax-from-zero-to-production
Chapter   : ch-04-ch04 (lesson 4)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-04-ch04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأشكال والبثّ والفهرسة المتقدّمة
JAX: From Zero to Production — einsum

Book slug : jax-from-zero-to-production
Chapter   : ch-04-ch04 (lesson 4)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

a = jnp.ones((3, 1, 4))
b = jnp.ones((2, 4))
print((a + b).shape)        # (3, 2, 4)

c = jnp.ones((3, 4))
d = jnp.ones((3,))
try:
    c + d                   # 4 مقابل 3: لا توافق
except ValueError as e:
    print("broadcast:", e)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.arange(12).reshape(3, 4)
print(x.T.shape)                      # (4, 3) تبديل
print(jnp.moveaxis(x, 0, 1).shape)    # (4, 3) نقل محور
print(jnp.expand_dims(x, 0).shape)    # (1, 3, 4) إضافة محور

y = jnp.ones((2, 3, 4))
print(y.transpose(0, 2, 1).shape)     # (2, 4, 3)
print(y.reshape(2, 12).shape)         # (2, 12) نفس ترتيب الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

A, B = jnp.ones((2, 3)), jnp.ones((3, 4))
print(jnp.einsum("ik,kj->ij", A, B).shape)      # (2, 4)

X, Y = jnp.ones((5, 2, 3)), jnp.ones((5, 3, 4))
print(jnp.einsum("bik,bkj->bij", X, Y).shape)   # (5, 2, 4) دفعة كاملة

M = jnp.arange(9).reshape(3, 3)
print(jnp.einsum("ii->", M))                    # الأثر: مجموع القطر
print(jnp.einsum("ii->i", M))                   # القطر نفسه

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([10.0, 20.0, 30.0, 40.0])
idx = jnp.array([2, 0, 2])
print(jnp.take(x, idx))          # [30. 10. 30.]
print(x.at[idx].add(1.0))        # [11. 20. 32. 40.] تراكم على المتكرّر

# بيانات غير مستوية: ثلاث مجموعات بأطوال مختلفة
vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
seg = jnp.array([0, 0, 1, 2, 2])
print(jax.ops.segment_sum(vals, seg, num_segments=3))   # [3. 3. 9.]

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([-1.0, 2.0, -3.0, 4.0])
mask = x > 0

print(x[mask])                    # [2. 4.] خارج jit: شكل ديناميكي
print(jnp.where(mask, x, 0.0))    # [0. 2. 0. 4.] شكل ثابت
print(jnp.sum(jnp.where(mask, x, 0.0)))   # 6.0 مجموع الموجب فقط

f = jax.jit(lambda v: jnp.sum(jnp.where(v > 0, v, 0.0)))
print(f(x))                       # يعمل: الشكل معروف

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

a = jnp.ones((3, 1, 4))
b = jnp.ones((2, 4))
print((a + b).shape)        # (3, 2, 4)

c = jnp.ones((3, 4))
d = jnp.ones((3,))
try:
    c + d                   # 4 مقابل 3: لا توافق
except ValueError as e:
    print("broadcast:", e)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.arange(12).reshape(3, 4)
print(x.T.shape)                      # (4, 3) تبديل
print(jnp.moveaxis(x, 0, 1).shape)    # (4, 3) نقل محور
print(jnp.expand_dims(x, 0).shape)    # (1, 3, 4) إضافة محور

y = jnp.ones((2, 3, 4))
print(y.transpose(0, 2, 1).shape)     # (2, 4, 3)
print(y.reshape(2, 12).shape)         # (2, 12) نفس ترتيب الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

A, B = jnp.ones((2, 3)), jnp.ones((3, 4))
print(jnp.einsum("ik,kj->ij", A, B).shape)      # (2, 4)

X, Y = jnp.ones((5, 2, 3)), jnp.ones((5, 3, 4))
print(jnp.einsum("bik,bkj->bij", X, Y).shape)   # (5, 2, 4) دفعة كاملة

M = jnp.arange(9).reshape(3, 3)
print(jnp.einsum("ii->", M))                    # الأثر: مجموع القطر
print(jnp.einsum("ii->i", M))                   # القطر نفسه

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([10.0, 20.0, 30.0, 40.0])
idx = jnp.array([2, 0, 2])
print(jnp.take(x, idx))          # [30. 10. 30.]
print(x.at[idx].add(1.0))        # [11. 20. 32. 40.] تراكم على المتكرّر

# بيانات غير مستوية: ثلاث مجموعات بأطوال مختلفة
vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
seg = jnp.array([0, 0, 1, 2, 2])
print(jax.ops.segment_sum(vals, seg, num_segments=3))   # [3. 3. 9.]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([-1.0, 2.0, -3.0, 4.0])
mask = x > 0

print(x[mask])                    # [2. 4.] خارج jit: شكل ديناميكي
print(jnp.where(mask, x, 0.0))    # [0. 2. 0. 4.] شكل ثابت
print(jnp.sum(jnp.where(mask, x, 0.0)))   # 6.0 مجموع الموجب فقط

f = jax.jit(lambda v: jnp.sum(jnp.where(v > 0, v, 0.0)))
print(f(x))                       # يعمل: الشكل معروف

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الأشكال والبثّ والفهرسة المتقدّمة
JAX: From Zero to Production — einsum

Book slug : jax-from-zero-to-production
Chapter   : ch-04-ch04 (lesson 4)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

a = jnp.ones((3, 1, 4))
b = jnp.ones((2, 4))
print((a + b).shape)        # (3, 2, 4)

c = jnp.ones((3, 4))
d = jnp.ones((3,))
try:
    c + d                   # 4 مقابل 3: لا توافق
except ValueError as e:
    print("broadcast:", e)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.arange(12).reshape(3, 4)
print(x.T.shape)                      # (4, 3) تبديل
print(jnp.moveaxis(x, 0, 1).shape)    # (4, 3) نقل محور
print(jnp.expand_dims(x, 0).shape)    # (1, 3, 4) إضافة محور

y = jnp.ones((2, 3, 4))
print(y.transpose(0, 2, 1).shape)     # (2, 4, 3)
print(y.reshape(2, 12).shape)         # (2, 12) نفس ترتيب الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

A, B = jnp.ones((2, 3)), jnp.ones((3, 4))
print(jnp.einsum("ik,kj->ij", A, B).shape)      # (2, 4)

X, Y = jnp.ones((5, 2, 3)), jnp.ones((5, 3, 4))
print(jnp.einsum("bik,bkj->bij", X, Y).shape)   # (5, 2, 4) دفعة كاملة

M = jnp.arange(9).reshape(3, 3)
print(jnp.einsum("ii->", M))                    # الأثر: مجموع القطر
print(jnp.einsum("ii->i", M))                   # القطر نفسه

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([10.0, 20.0, 30.0, 40.0])
idx = jnp.array([2, 0, 2])
print(jnp.take(x, idx))          # [30. 10. 30.]
print(x.at[idx].add(1.0))        # [11. 20. 32. 40.] تراكم على المتكرّر

# بيانات غير مستوية: ثلاث مجموعات بأطوال مختلفة
vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
seg = jnp.array([0, 0, 1, 2, 2])
print(jax.ops.segment_sum(vals, seg, num_segments=3))   # [3. 3. 9.]

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch04.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([-1.0, 2.0, -3.0, 4.0])
mask = x > 0

print(x[mask])                    # [2. 4.] خارج jit: شكل ديناميكي
print(jnp.where(mask, x, 0.0))    # [0. 2. 0. 4.] شكل ثابت
print(jnp.sum(jnp.where(mask, x, 0.0)))   # 6.0 مجموع الموجب فقط

f = jax.jit(lambda v: jnp.sum(jnp.where(v > 0, v, 0.0)))
print(f(x))                       # يعمل: الشكل معروف

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

a = jnp.ones((3, 1, 4))
b = jnp.ones((2, 4))
print((a + b).shape)        # (3, 2, 4)

c = jnp.ones((3, 4))
d = jnp.ones((3,))
try:
    c + d                   # 4 مقابل 3: لا توافق
except ValueError as e:
    print("broadcast:", e)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

x = jnp.arange(12).reshape(3, 4)
print(x.T.shape)                      # (4, 3) تبديل
print(jnp.moveaxis(x, 0, 1).shape)    # (4, 3) نقل محور
print(jnp.expand_dims(x, 0).shape)    # (1, 3, 4) إضافة محور

y = jnp.ones((2, 3, 4))
print(y.transpose(0, 2, 1).shape)     # (2, 4, 3)
print(y.reshape(2, 12).shape)         # (2, 12) نفس ترتيب الذاكرة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

A, B = jnp.ones((2, 3)), jnp.ones((3, 4))
print(jnp.einsum("ik,kj->ij", A, B).shape)      # (2, 4)

X, Y = jnp.ones((5, 2, 3)), jnp.ones((5, 3, 4))
print(jnp.einsum("bik,bkj->bij", X, Y).shape)   # (5, 2, 4) دفعة كاملة

M = jnp.arange(9).reshape(3, 3)
print(jnp.einsum("ii->", M))                    # الأثر: مجموع القطر
print(jnp.einsum("ii->i", M))                   # القطر نفسه

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([10.0, 20.0, 30.0, 40.0])
idx = jnp.array([2, 0, 2])
print(jnp.take(x, idx))          # [30. 10. 30.]
print(x.at[idx].add(1.0))        # [11. 20. 32. 40.] تراكم على المتكرّر

# بيانات غير مستوية: ثلاث مجموعات بأطوال مختلفة
vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
seg = jnp.array([0, 0, 1, 2, 2])
print(jax.ops.segment_sum(vals, seg, num_segments=3))   # [3. 3. 9.]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-04-ch04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

x = jnp.array([-1.0, 2.0, -3.0, 4.0])
mask = x > 0

print(x[mask])                    # [2. 4.] خارج jit: شكل ديناميكي
print(jnp.where(mask, x, 0.0))    # [0. 2. 0. 4.] شكل ثابت
print(jnp.sum(jnp.where(mask, x, 0.0)))   # 6.0 مجموع الموجب فقط

f = jax.jit(lambda v: jnp.sum(jnp.where(v > 0, v, 0.0)))
print(f(x))                       # يعمل: الشكل معروف
