#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع والعشرون: XLA من الداخل
JAX: From Zero to Production — HLO

Book slug : jax-from-zero-to-production
Chapter   : ch-24-ch24 (lesson 24)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-24-ch24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع والعشرون: XLA من الداخل
JAX: From Zero to Production — HLO

Book slug : jax-from-zero-to-production
Chapter   : ch-24-ch24 (lesson 24)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-24-ch24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع والعشرون: XLA من الداخل
JAX: From Zero to Production — HLO

Book slug : jax-from-zero-to-production
Chapter   : ch-24-ch24 (lesson 24)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch24.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) * x)

x = jnp.ones((8, 8), dtype=jnp.float32)

lowered = f.lower(x)
print(lowered.as_text())              # HLO قبل التحسين
exe = lowered.compile()
print(exe.as_text())                  # HLO بعد التحسين

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch24.md
# ────────────────────────────────────────────────────────────────────────
import jax

jax.config.update("jax_compilation_cache_dir", "/var/cache/jax")
# الثابت الافتراضي يتجاهل الدوال سريعة التجميع؛ ارفعه لتخزين كل شيء
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch24.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
step(x).block_until_ready()          # تسخين: استبعد زمن التجميع

jax.profiler.start_trace("/tmp/trace")
for _ in range(20):
    step(x).block_until_ready()
jax.profiler.stop_trace()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) * x)

x = jnp.ones((8, 8), dtype=jnp.float32)

lowered = f.lower(x)
print(lowered.as_text())              # HLO قبل التحسين
exe = lowered.compile()
print(exe.as_text())                  # HLO بعد التحسين

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax

jax.config.update("jax_compilation_cache_dir", "/var/cache/jax")
# الثابت الافتراضي يتجاهل الدوال سريعة التجميع؛ ارفعه لتخزين كل شيء
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
step(x).block_until_ready()          # تسخين: استبعد زمن التجميع

jax.profiler.start_trace("/tmp/trace")
for _ in range(20):
    step(x).block_until_ready()
jax.profiler.stop_trace()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع والعشرون: XLA من الداخل
JAX: From Zero to Production — HLO

Book slug : jax-from-zero-to-production
Chapter   : ch-24-ch24 (lesson 24)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch24.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) * x)

x = jnp.ones((8, 8), dtype=jnp.float32)

lowered = f.lower(x)
print(lowered.as_text())              # HLO قبل التحسين
exe = lowered.compile()
print(exe.as_text())                  # HLO بعد التحسين

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch24.md
# ────────────────────────────────────────────────────────────────────────
import jax

jax.config.update("jax_compilation_cache_dir", "/var/cache/jax")
# الثابت الافتراضي يتجاهل الدوال سريعة التجميع؛ ارفعه لتخزين كل شيء
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch24.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
step(x).block_until_ready()          # تسخين: استبعد زمن التجميع

jax.profiler.start_trace("/tmp/trace")
for _ in range(20):
    step(x).block_until_ready()
jax.profiler.stop_trace()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sum(jnp.tanh(x) * x)

x = jnp.ones((8, 8), dtype=jnp.float32)

lowered = f.lower(x)
print(lowered.as_text())              # HLO قبل التحسين
exe = lowered.compile()
print(exe.as_text())                  # HLO بعد التحسين

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax

jax.config.update("jax_compilation_cache_dir", "/var/cache/jax")
# الثابت الافتراضي يتجاهل الدوال سريعة التجميع؛ ارفعه لتخزين كل شيء
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-24-ch24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def step(x):
    return jnp.tanh(x @ x.T).sum()

x = jnp.ones((2048, 2048))
step(x).block_until_ready()          # تسخين: استبعد زمن التجميع

jax.profiler.start_trace("/tmp/trace")
for _ in range(20):
    step(x).block_until_ready()
jax.profiler.stop_trace()
