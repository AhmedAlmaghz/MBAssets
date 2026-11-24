#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — النقاء والحالة وأشجار pytree
JAX: From Zero to Production — Chapter 09 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-09-ch09 (lesson 9)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-09-ch09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — النقاء والحالة وأشجار pytree
JAX: From Zero to Production — Chapter 09 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-09-ch09 (lesson 9)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-09-ch09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — النقاء والحالة وأشجار pytree
JAX: From Zero to Production — Chapter 09 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-09-ch09 (lesson 9)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

count = 0

@jax.jit
def step(x):
    global count
    count += 1          # ينفّذ عند التتبّع، لا عند التشغيل
    return x + 1.0

for _ in range(3):
    step(jnp.zeros(()))
print("count =", count)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}
grads  = {"w": jnp.full((2, 3), 0.5), "b": jnp.ones(3)}

updated = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

print(len(jax.tree.leaves(params)))                               # 2
print(jax.tree.structure(updated) == jax.tree.structure(params))  # True

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class Dense:
    w: jax.Array
    b: jax.Array

def flatten(d):
    return (d.w, d.b), None          # الأبناء، ثم البيانات الثابتة

def unflatten(aux, children):
    return Dense(*children)

jax.tree_util.register_pytree_node(Dense, flatten, unflatten)

layer = Dense(jnp.ones((3, 4)), jnp.zeros(3))
print(jax.tree.leaves(layer))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def step(state, x):
    count, total = state
    return (count + 1, total + x), x * 2.0

state = (jnp.array(0), jnp.array(0.0))
step_jit = jax.jit(step)

for i in range(3):
    state, y = step_jit(state, jnp.array(float(i)))

print(state)    # (3, 3.0)
print(y)        # 4.0

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(8)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)

model = Block()
x = jnp.ones((4, 8))
variables = model.init(jax.random.key(0), x, train=True)

y, updates = model.apply(
    {"params": variables["params"], "batch_stats": variables["batch_stats"]},
    x, train=True, mutable=["batch_stats"])

print(y.shape, updates["batch_stats"]["mean"].shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
from flax import nnx

class Block(nnx.Module):
    def __init__(self, rngs: nnx.RNGs):
        self.linear = nnx.Linear(8, 8, rngs=rngs)
        self.norm = nnx.BatchNorm(8, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.norm(self.linear(x)))

model = Block(nnx.RNGs(0))
graphdef, state = nnx.split(model)      # الشجرة عند الحدود
print(len(jax.tree.leaves(state)))
model = nnx.merge(graphdef, state)      # عودة إلى كائن حيّ

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

count = 0

@jax.jit
def step(x):
    global count
    count += 1          # ينفّذ عند التتبّع، لا عند التشغيل
    return x + 1.0

for _ in range(3):
    step(jnp.zeros(()))
print("count =", count)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}
grads  = {"w": jnp.full((2, 3), 0.5), "b": jnp.ones(3)}

updated = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

print(len(jax.tree.leaves(params)))                               # 2
print(jax.tree.structure(updated) == jax.tree.structure(params))  # True

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class Dense:
    w: jax.Array
    b: jax.Array

def flatten(d):
    return (d.w, d.b), None          # الأبناء، ثم البيانات الثابتة

def unflatten(aux, children):
    return Dense(*children)

jax.tree_util.register_pytree_node(Dense, flatten, unflatten)

layer = Dense(jnp.ones((3, 4)), jnp.zeros(3))
print(jax.tree.leaves(layer))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def step(state, x):
    count, total = state
    return (count + 1, total + x), x * 2.0

state = (jnp.array(0), jnp.array(0.0))
step_jit = jax.jit(step)

for i in range(3):
    state, y = step_jit(state, jnp.array(float(i)))

print(state)    # (3, 3.0)
print(y)        # 4.0

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(8)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)

model = Block()
x = jnp.ones((4, 8))
variables = model.init(jax.random.key(0), x, train=True)

y, updates = model.apply(
    {"params": variables["params"], "batch_stats": variables["batch_stats"]},
    x, train=True, mutable=["batch_stats"])

print(y.shape, updates["batch_stats"]["mean"].shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax
from flax import nnx

class Block(nnx.Module):
    def __init__(self, rngs: nnx.RNGs):
        self.linear = nnx.Linear(8, 8, rngs=rngs)
        self.norm = nnx.BatchNorm(8, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.norm(self.linear(x)))

model = Block(nnx.RNGs(0))
graphdef, state = nnx.split(model)      # الشجرة عند الحدود
print(len(jax.tree.leaves(state)))
model = nnx.merge(graphdef, state)      # عودة إلى كائن حيّ

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — النقاء والحالة وأشجار pytree
JAX: From Zero to Production — Chapter 09 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-09-ch09 (lesson 9)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

count = 0

@jax.jit
def step(x):
    global count
    count += 1          # ينفّذ عند التتبّع، لا عند التشغيل
    return x + 1.0

for _ in range(3):
    step(jnp.zeros(()))
print("count =", count)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}
grads  = {"w": jnp.full((2, 3), 0.5), "b": jnp.ones(3)}

updated = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

print(len(jax.tree.leaves(params)))                               # 2
print(jax.tree.structure(updated) == jax.tree.structure(params))  # True

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class Dense:
    w: jax.Array
    b: jax.Array

def flatten(d):
    return (d.w, d.b), None          # الأبناء، ثم البيانات الثابتة

def unflatten(aux, children):
    return Dense(*children)

jax.tree_util.register_pytree_node(Dense, flatten, unflatten)

layer = Dense(jnp.ones((3, 4)), jnp.zeros(3))
print(jax.tree.leaves(layer))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def step(state, x):
    count, total = state
    return (count + 1, total + x), x * 2.0

state = (jnp.array(0), jnp.array(0.0))
step_jit = jax.jit(step)

for i in range(3):
    state, y = step_jit(state, jnp.array(float(i)))

print(state)    # (3, 3.0)
print(y)        # 4.0

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(8)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)

model = Block()
x = jnp.ones((4, 8))
variables = model.init(jax.random.key(0), x, train=True)

y, updates = model.apply(
    {"params": variables["params"], "batch_stats": variables["batch_stats"]},
    x, train=True, mutable=["batch_stats"])

print(y.shape, updates["batch_stats"]["mean"].shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch09.md
# ────────────────────────────────────────────────────────────────────────
import jax
from flax import nnx

class Block(nnx.Module):
    def __init__(self, rngs: nnx.RNGs):
        self.linear = nnx.Linear(8, 8, rngs=rngs)
        self.norm = nnx.BatchNorm(8, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.norm(self.linear(x)))

model = Block(nnx.RNGs(0))
graphdef, state = nnx.split(model)      # الشجرة عند الحدود
print(len(jax.tree.leaves(state)))
model = nnx.merge(graphdef, state)      # عودة إلى كائن حيّ

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

count = 0

@jax.jit
def step(x):
    global count
    count += 1          # ينفّذ عند التتبّع، لا عند التشغيل
    return x + 1.0

for _ in range(3):
    step(jnp.zeros(()))
print("count =", count)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

params = {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)}
grads  = {"w": jnp.full((2, 3), 0.5), "b": jnp.ones(3)}

updated = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

print(len(jax.tree.leaves(params)))                               # 2
print(jax.tree.structure(updated) == jax.tree.structure(params))  # True

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class Dense:
    w: jax.Array
    b: jax.Array

def flatten(d):
    return (d.w, d.b), None          # الأبناء، ثم البيانات الثابتة

def unflatten(aux, children):
    return Dense(*children)

jax.tree_util.register_pytree_node(Dense, flatten, unflatten)

layer = Dense(jnp.ones((3, 4)), jnp.zeros(3))
print(jax.tree.leaves(layer))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def step(state, x):
    count, total = state
    return (count + 1, total + x), x * 2.0

state = (jnp.array(0), jnp.array(0.0))
step_jit = jax.jit(step)

for i in range(3):
    state, y = step_jit(state, jnp.array(float(i)))

print(state)    # (3, 3.0)
print(y)        # 4.0

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(8)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x)

model = Block()
x = jnp.ones((4, 8))
variables = model.init(jax.random.key(0), x, train=True)

y, updates = model.apply(
    {"params": variables["params"], "batch_stats": variables["batch_stats"]},
    x, train=True, mutable=["batch_stats"])

print(y.shape, updates["batch_stats"]["mean"].shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-09-ch09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import jax
from flax import nnx

class Block(nnx.Module):
    def __init__(self, rngs: nnx.RNGs):
        self.linear = nnx.Linear(8, 8, rngs=rngs)
        self.norm = nnx.BatchNorm(8, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.norm(self.linear(x)))

model = Block(nnx.RNGs(0))
graphdef, state = nnx.split(model)      # الشجرة عند الحدود
print(len(jax.tree.leaves(state)))
model = nnx.merge(graphdef, state)      # عودة إلى كائن حيّ
