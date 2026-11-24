#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي عشر: من pmap إلى shard_map: التوازي
JAX: From Zero to Production — Chapter 11 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-11-ch11 (lesson 11)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-11-ch11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي عشر: من pmap إلى shard_map: التوازي
JAX: From Zero to Production — Chapter 11 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-11-ch11 (lesson 11)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-11-ch11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي عشر: من pmap إلى shard_map: التوازي
JAX: From Zero to Production — Chapter 11 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-11-ch11 (lesson 11)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devices = jax.devices()                        # ثمانية أجهزة على مضيف واحد
assert len(devices) == 8
mesh = jax.make_mesh((8,), ("x",), devices=devices)
sharding = NamedSharding(mesh, P("x"))

x = jax.device_put(jnp.arange(64.0).reshape(8, 8), sharding)
print(x.sharding)                              # P('x',)
print(x.addressable_shards[0].data.shape)      # (1, 8)

@jax.jit
def col_sum(a):
    return a.sum(axis=0)

print(col_sum(x).shape)                        # (8,) على محور مكرّر

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
from functools import partial
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=P("x"))
def local_center(x):
    return x - jax.lax.pmean(x.mean(), "x")

y = local_center(x)
print(y.shape)      # (8, 8) عالمي

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-11-ch11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devices = jax.devices()                        # ثمانية أجهزة على مضيف واحد
assert len(devices) == 8
mesh = jax.make_mesh((8,), ("x",), devices=devices)
sharding = NamedSharding(mesh, P("x"))

x = jax.device_put(jnp.arange(64.0).reshape(8, 8), sharding)
print(x.sharding)                              # P('x',)
print(x.addressable_shards[0].data.shape)      # (1, 8)

@jax.jit
def col_sum(a):
    return a.sum(axis=0)

print(col_sum(x).shape)                        # (8,) على محور مكرّر

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-11-ch11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from functools import partial
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=P("x"))
def local_center(x):
    return x - jax.lax.pmean(x.mean(), "x")

y = local_center(x)
print(y.shape)      # (8, 8) عالمي

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-11-ch11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي عشر: من pmap إلى shard_map: التوازي
JAX: From Zero to Production — Chapter 11 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-11-ch11 (lesson 11)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devices = jax.devices()                        # ثمانية أجهزة على مضيف واحد
assert len(devices) == 8
mesh = jax.make_mesh((8,), ("x",), devices=devices)
sharding = NamedSharding(mesh, P("x"))

x = jax.device_put(jnp.arange(64.0).reshape(8, 8), sharding)
print(x.sharding)                              # P('x',)
print(x.addressable_shards[0].data.shape)      # (1, 8)

@jax.jit
def col_sum(a):
    return a.sum(axis=0)

print(col_sum(x).shape)                        # (8,) على محور مكرّر

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
from functools import partial
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=P("x"))
def local_center(x):
    return x - jax.lax.pmean(x.mean(), "x")

y = local_center(x)
print(y.shape)      # (8, 8) عالمي

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-11-ch11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devices = jax.devices()                        # ثمانية أجهزة على مضيف واحد
assert len(devices) == 8
mesh = jax.make_mesh((8,), ("x",), devices=devices)
sharding = NamedSharding(mesh, P("x"))

x = jax.device_put(jnp.arange(64.0).reshape(8, 8), sharding)
print(x.sharding)                              # P('x',)
print(x.addressable_shards[0].data.shape)      # (1, 8)

@jax.jit
def col_sum(a):
    return a.sum(axis=0)

print(col_sum(x).shape)                        # (8,) على محور مكرّر

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-11-ch11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from functools import partial
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh, in_specs=(P("x"),), out_specs=P("x"))
def local_center(x):
    return x - jax.lax.pmean(x.mean(), "x")

y = local_center(x)
print(y.shape)      # (8, 8) عالمي
