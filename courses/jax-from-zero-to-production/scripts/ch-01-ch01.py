#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — القصة: لماذا وُلدت JAX؟
JAX: From Zero to Production — Chapter 01 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-01-ch01 (lesson 1)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-01-ch01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — القصة: لماذا وُلدت JAX؟
JAX: From Zero to Production — Chapter 01 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-01-ch01 (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-01-ch01.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — القصة: لماذا وُلدت JAX؟
JAX: From Zero to Production — Chapter 01 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-01-ch01 (lesson 1)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.randn(4096, 4096).astype("float32")
W = np.random.randn(4096, 4096).astype("float32")
b = np.random.randn(4096).astype("float32")

h = x @ W          # مصفوفة وسيطة 1
h = h + b          # مصفوفة وسيطة 2
h = np.maximum(h, 0)   # مصفوفة وسيطة 3
print(h.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch01.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def block(x, W, b):
    return jax.nn.relu(x @ W + b)

x = jax.random.normal(jax.random.key(0), (4096, 4096))
W = jax.random.normal(jax.random.key(1), (4096, 4096))
b = jnp.zeros(4096)

h = block(x, W, b)     # نواة واحدة، لا مصفوفات وسيطة

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch01.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def bad(x):
    if x[0] > 0:          # قرار يعتمد على قيمة!
        return x * 2
    return x + 1

bad(jnp.array([1.0, 2.0]))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.randn(4096, 4096).astype("float32")
W = np.random.randn(4096, 4096).astype("float32")
b = np.random.randn(4096).astype("float32")

h = x @ W          # مصفوفة وسيطة 1
h = h + b          # مصفوفة وسيطة 2
h = np.maximum(h, 0)   # مصفوفة وسيطة 3
print(h.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def block(x, W, b):
    return jax.nn.relu(x @ W + b)

x = jax.random.normal(jax.random.key(0), (4096, 4096))
W = jax.random.normal(jax.random.key(1), (4096, 4096))
b = jnp.zeros(4096)

h = block(x, W, b)     # نواة واحدة، لا مصفوفات وسيطة

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def bad(x):
    if x[0] > 0:          # قرار يعتمد على قيمة!
        return x * 2
    return x + 1

bad(jnp.array([1.0, 2.0]))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — القصة: لماذا وُلدت JAX؟
JAX: From Zero to Production — Chapter 01 — Foundations: Why JAX

Book slug : jax-from-zero-to-production
Chapter   : ch-01-ch01 (lesson 1)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch01.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.randn(4096, 4096).astype("float32")
W = np.random.randn(4096, 4096).astype("float32")
b = np.random.randn(4096).astype("float32")

h = x @ W          # مصفوفة وسيطة 1
h = h + b          # مصفوفة وسيطة 2
h = np.maximum(h, 0)   # مصفوفة وسيطة 3
print(h.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch01.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def block(x, W, b):
    return jax.nn.relu(x @ W + b)

x = jax.random.normal(jax.random.key(0), (4096, 4096))
W = jax.random.normal(jax.random.key(1), (4096, 4096))
b = jnp.zeros(4096)

h = block(x, W, b)     # نواة واحدة، لا مصفوفات وسيطة

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch01.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def bad(x):
    if x[0] > 0:          # قرار يعتمد على قيمة!
        return x * 2
    return x + 1

bad(jnp.array([1.0, 2.0]))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.randn(4096, 4096).astype("float32")
W = np.random.randn(4096, 4096).astype("float32")
b = np.random.randn(4096).astype("float32")

h = x @ W          # مصفوفة وسيطة 1
h = h + b          # مصفوفة وسيطة 2
h = np.maximum(h, 0)   # مصفوفة وسيطة 3
print(h.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def block(x, W, b):
    return jax.nn.relu(x @ W + b)

x = jax.random.normal(jax.random.key(0), (4096, 4096))
W = jax.random.normal(jax.random.key(1), (4096, 4096))
b = jnp.zeros(4096)

h = block(x, W, b)     # نواة واحدة، لا مصفوفات وسيطة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-01-ch01.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.jit
def bad(x):
    if x[0] > 0:          # قرار يعتمد على قيمة!
        return x * 2
    return x + 1

bad(jnp.array([1.0, 2.0]))
