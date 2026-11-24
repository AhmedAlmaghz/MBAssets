#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — بيئة عمل احترافية على Python 3.14
JAX: From Zero to Production — Chapter 02 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-02-ch02 (lesson 2)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-02-ch02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — بيئة عمل احترافية على Python 3.14
JAX: From Zero to Production — Chapter 02 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-02-ch02 (lesson 2)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-02-ch02.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — بيئة عمل احترافية على Python 3.14
JAX: From Zero to Production — Chapter 02 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-02-ch02 (lesson 2)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import jaxlib

print("jax     :", jax.__version__)
print("jaxlib  :", jaxlib.__version__)
print("backend :", jax.default_backend())
print("devices :", jax.devices())

# اختبار وظيفي صغير: جمع + تجميع + تدرّج
@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) ** 2)

x = jnp.arange(8.0)
print("f(x)    :", f(x))
print("grad    :", jax.grad(f)(x))

# تحقّق من أن التجميع يعمل فعلاً على الجهاز المطلوب
print("lowered :", f.lower(x).compile().devices())

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
import jax
# اكتشاف NaN فور حدوثه بدل البحث عنه لاحقاً
jax.config.update("jax_debug_nans", True)
# تفعيل الأعداد 64-بت (معطّل افتراضياً لأسباب أداء)
jax.config.update("jax_enable_x64", True)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-02-ch02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import jaxlib

print("jax     :", jax.__version__)
print("jaxlib  :", jaxlib.__version__)
print("backend :", jax.default_backend())
print("devices :", jax.devices())

# اختبار وظيفي صغير: جمع + تجميع + تدرّج
@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) ** 2)

x = jnp.arange(8.0)
print("f(x)    :", f(x))
print("grad    :", jax.grad(f)(x))

# تحقّق من أن التجميع يعمل فعلاً على الجهاز المطلوب
print("lowered :", f.lower(x).compile().devices())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-02-ch02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
# اكتشاف NaN فور حدوثه بدل البحث عنه لاحقاً
jax.config.update("jax_debug_nans", True)
# تفعيل الأعداد 64-بت (معطّل افتراضياً لأسباب أداء)
jax.config.update("jax_enable_x64", True)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-02-ch02.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — بيئة عمل احترافية على Python 3.14
JAX: From Zero to Production — Chapter 02 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-02-ch02 (lesson 2)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import jaxlib

print("jax     :", jax.__version__)
print("jaxlib  :", jaxlib.__version__)
print("backend :", jax.default_backend())
print("devices :", jax.devices())

# اختبار وظيفي صغير: جمع + تجميع + تدرّج
@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) ** 2)

x = jnp.arange(8.0)
print("f(x)    :", f(x))
print("grad    :", jax.grad(f)(x))

# تحقّق من أن التجميع يعمل فعلاً على الجهاز المطلوب
print("lowered :", f.lower(x).compile().devices())

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch02.md
# ────────────────────────────────────────────────────────────────────────
import jax
# اكتشاف NaN فور حدوثه بدل البحث عنه لاحقاً
jax.config.update("jax_debug_nans", True)
# تفعيل الأعداد 64-بت (معطّل افتراضياً لأسباب أداء)
jax.config.update("jax_enable_x64", True)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-02-ch02.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import jaxlib

print("jax     :", jax.__version__)
print("jaxlib  :", jaxlib.__version__)
print("backend :", jax.default_backend())
print("devices :", jax.devices())

# اختبار وظيفي صغير: جمع + تجميع + تدرّج
@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) ** 2)

x = jnp.arange(8.0)
print("f(x)    :", f(x))
print("grad    :", jax.grad(f)(x))

# تحقّق من أن التجميع يعمل فعلاً على الجهاز المطلوب
print("lowered :", f.lower(x).compile().devices())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-02-ch02.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
# اكتشاف NaN فور حدوثه بدل البحث عنه لاحقاً
jax.config.update("jax_debug_nans", True)
# تفعيل الأعداد 64-بت (معطّل افتراضياً لأسباب أداء)
jax.config.update("jax_enable_x64", True)
