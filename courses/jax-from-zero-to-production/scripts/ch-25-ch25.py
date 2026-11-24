#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس والعشرون: التقطيع: Mesh وNamedSharding
JAX: From Zero to Production — GSPMD

Book slug : jax-from-zero-to-production
Chapter   : ch-25-ch25 (lesson 25)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-25-ch25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس والعشرون: التقطيع: Mesh وNamedSharding
JAX: From Zero to Production — GSPMD

Book slug : jax-from-zero-to-production
Chapter   : ch-25-ch25 (lesson 25)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-25-ch25.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس والعشرون: التقطيع: Mesh وNamedSharding
JAX: From Zero to Production — GSPMD

Book slug : jax-from-zero-to-production
Chapter   : ch-25-ch25 (lesson 25)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
import jax
import numpy as np

devices = np.array(jax.devices()).reshape(2, 4)   # صفّان، أربعة أعمدة
mesh = jax.make_mesh((2, 4), axis_names=("x", "y"), devices=devices)

print(mesh.shape)        # {'x': 2, 'y': 4}
print(mesh.devices[0, 3])

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import NamedSharding, PartitionSpec as P

data_par = NamedSharding(mesh, P("x", None))   # الدفعة مقسومة، الميزات كاملة
model_par = NamedSharding(mesh, P())           # نسخة كاملة على كل جهاز
tensor_par = NamedSharding(mesh, P(None, "y")) # الأوزان مقسومة على y

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((8,), axis_names=("x",))
shard_batch = NamedSharding(mesh, P("x"))
replicate   = NamedSharding(mesh, P())

def loss(w, xb, yb):
    pred = jnp.tanh(xb @ w)
    return jnp.mean((pred - yb) ** 2)

step = jax.jit(
    jax.grad(loss),
    in_shardings=(replicate, shard_batch, shard_batch),
    out_shardings=replicate,
)

w  = jnp.ones((64, 8)) / 8
xb = jnp.ones((1024, 64))
yb = jnp.zeros((1024, 8))

g = step(w, xb, yb)
print(g.sharding)     # نسخة كاملة من التدرّج على كل جهاز

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np, jax

def sharded_batch(step_idx, batch=1024):
    host = np.load(f"batch_{step_idx:05d}.npy")     # (1024, 64) على المضيف
    parts = np.split(host, 8, axis=0)               # ثماني قطع متساوية
    return jax.device_put(parts[0], shard_batch)    # كل قطعة إلى جهازها

xb = sharded_batch(0)
for i in range(1, 100):
    nxt = sharded_batch(i)                          # تحميل مسبق
    g = step(w, xb, yb)
    xb = nxt

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
import numpy as np

devices = np.array(jax.devices()).reshape(2, 4)   # صفّان، أربعة أعمدة
mesh = jax.make_mesh((2, 4), axis_names=("x", "y"), devices=devices)

print(mesh.shape)        # {'x': 2, 'y': 4}
print(mesh.devices[0, 3])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import NamedSharding, PartitionSpec as P

data_par = NamedSharding(mesh, P("x", None))   # الدفعة مقسومة، الميزات كاملة
model_par = NamedSharding(mesh, P())           # نسخة كاملة على كل جهاز
tensor_par = NamedSharding(mesh, P(None, "y")) # الأوزان مقسومة على y

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((8,), axis_names=("x",))
shard_batch = NamedSharding(mesh, P("x"))
replicate   = NamedSharding(mesh, P())

def loss(w, xb, yb):
    pred = jnp.tanh(xb @ w)
    return jnp.mean((pred - yb) ** 2)

step = jax.jit(
    jax.grad(loss),
    in_shardings=(replicate, shard_batch, shard_batch),
    out_shardings=replicate,
)

w  = jnp.ones((64, 8)) / 8
xb = jnp.ones((1024, 64))
yb = jnp.zeros((1024, 8))

g = step(w, xb, yb)
print(g.sharding)     # نسخة كاملة من التدرّج على كل جهاز

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np, jax

def sharded_batch(step_idx, batch=1024):
    host = np.load(f"batch_{step_idx:05d}.npy")     # (1024, 64) على المضيف
    parts = np.split(host, 8, axis=0)               # ثماني قطع متساوية
    return jax.device_put(parts[0], shard_batch)    # كل قطعة إلى جهازها

xb = sharded_batch(0)
for i in range(1, 100):
    nxt = sharded_batch(i)                          # تحميل مسبق
    g = step(w, xb, yb)
    xb = nxt

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس والعشرون: التقطيع: Mesh وNamedSharding
JAX: From Zero to Production — GSPMD

Book slug : jax-from-zero-to-production
Chapter   : ch-25-ch25 (lesson 25)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
import jax
import numpy as np

devices = np.array(jax.devices()).reshape(2, 4)   # صفّان، أربعة أعمدة
mesh = jax.make_mesh((2, 4), axis_names=("x", "y"), devices=devices)

print(mesh.shape)        # {'x': 2, 'y': 4}
print(mesh.devices[0, 3])

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import NamedSharding, PartitionSpec as P

data_par = NamedSharding(mesh, P("x", None))   # الدفعة مقسومة، الميزات كاملة
model_par = NamedSharding(mesh, P())           # نسخة كاملة على كل جهاز
tensor_par = NamedSharding(mesh, P(None, "y")) # الأوزان مقسومة على y

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((8,), axis_names=("x",))
shard_batch = NamedSharding(mesh, P("x"))
replicate   = NamedSharding(mesh, P())

def loss(w, xb, yb):
    pred = jnp.tanh(xb @ w)
    return jnp.mean((pred - yb) ** 2)

step = jax.jit(
    jax.grad(loss),
    in_shardings=(replicate, shard_batch, shard_batch),
    out_shardings=replicate,
)

w  = jnp.ones((64, 8)) / 8
xb = jnp.ones((1024, 64))
yb = jnp.zeros((1024, 8))

g = step(w, xb, yb)
print(g.sharding)     # نسخة كاملة من التدرّج على كل جهاز

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch25.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np, jax

def sharded_batch(step_idx, batch=1024):
    host = np.load(f"batch_{step_idx:05d}.npy")     # (1024, 64) على المضيف
    parts = np.split(host, 8, axis=0)               # ثماني قطع متساوية
    return jax.device_put(parts[0], shard_batch)    # كل قطعة إلى جهازها

xb = sharded_batch(0)
for i in range(1, 100):
    nxt = sharded_batch(i)                          # تحميل مسبق
    g = step(w, xb, yb)
    xb = nxt

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import numpy as np

devices = np.array(jax.devices()).reshape(2, 4)   # صفّان، أربعة أعمدة
mesh = jax.make_mesh((2, 4), axis_names=("x", "y"), devices=devices)

print(mesh.shape)        # {'x': 2, 'y': 4}
print(mesh.devices[0, 3])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import NamedSharding, PartitionSpec as P

data_par = NamedSharding(mesh, P("x", None))   # الدفعة مقسومة، الميزات كاملة
model_par = NamedSharding(mesh, P())           # نسخة كاملة على كل جهاز
tensor_par = NamedSharding(mesh, P(None, "y")) # الأوزان مقسومة على y

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((8,), axis_names=("x",))
shard_batch = NamedSharding(mesh, P("x"))
replicate   = NamedSharding(mesh, P())

def loss(w, xb, yb):
    pred = jnp.tanh(xb @ w)
    return jnp.mean((pred - yb) ** 2)

step = jax.jit(
    jax.grad(loss),
    in_shardings=(replicate, shard_batch, shard_batch),
    out_shardings=replicate,
)

w  = jnp.ones((64, 8)) / 8
xb = jnp.ones((1024, 64))
yb = jnp.zeros((1024, 8))

g = step(w, xb, yb)
print(g.sharding)     # نسخة كاملة من التدرّج على كل جهاز

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-25-ch25.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np, jax

def sharded_batch(step_idx, batch=1024):
    host = np.load(f"batch_{step_idx:05d}.npy")     # (1024, 64) على المضيف
    parts = np.split(host, 8, axis=0)               # ثماني قطع متساوية
    return jax.device_put(parts[0], shard_batch)    # كل قطعة إلى جهازها

xb = sharded_batch(0)
for i in range(1, 100):
    nxt = sharded_batch(i)                          # تحميل مسبق
    g = step(w, xb, yb)
    xb = nxt
