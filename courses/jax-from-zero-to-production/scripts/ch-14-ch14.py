#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع عشر: Flax: من Linen إلى NNX
JAX: From Zero to Production — Chapter 14 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-14-ch14 (lesson 14)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-14-ch14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع عشر: Flax: من Linen إلى NNX
JAX: From Zero to Production — Chapter 14 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-14-ch14 (lesson 14)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-14-ch14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع عشر: Flax: من Linen إلى NNX
JAX: From Zero to Production — Chapter 14 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-14-ch14 (lesson 14)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(2)(x)

model = MLP()
params = model.init(jax.random.key(0), jnp.ones((1, 2)))
print(model.apply(params, jnp.ones((8, 2))).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

for path, leaf in jax.tree.leaves_with_path(params):
    print(path, leaf.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

model = Net()
v = model.init(jax.random.key(0), jnp.ones((1, 28, 28, 1)), train=True)
print(sorted(v.keys()))            # ['batch_stats', 'params']

logits, new = model.apply(v, x, train=True, mutable=["batch_stats"])
v = {**v, "batch_stats": new["batch_stats"]}

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = nnx.Linear(2, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 32, rngs=rngs)
        self.fc3 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)

model = MLP(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-2))

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(2)(x)

model = MLP()
params = model.init(jax.random.key(0), jnp.ones((1, 2)))
print(model.apply(params, jnp.ones((8, 2))).shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

for path, leaf in jax.tree.leaves_with_path(params):
    print(path, leaf.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

model = Net()
v = model.init(jax.random.key(0), jnp.ones((1, 28, 28, 1)), train=True)
print(sorted(v.keys()))            # ['batch_stats', 'params']

logits, new = model.apply(v, x, train=True, mutable=["batch_stats"])
v = {**v, "batch_stats": new["batch_stats"]}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = nnx.Linear(2, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 32, rngs=rngs)
        self.fc3 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)

model = MLP(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-2))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الرابع عشر: Flax: من Linen إلى NNX
JAX: From Zero to Production — Chapter 14 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-14-ch14 (lesson 14)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(2)(x)

model = MLP()
params = model.init(jax.random.key(0), jnp.ones((1, 2)))
print(model.apply(params, jnp.ones((8, 2))).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

for path, leaf in jax.tree.leaves_with_path(params):
    print(path, leaf.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

model = Net()
v = model.init(jax.random.key(0), jnp.ones((1, 28, 28, 1)), train=True)
print(sorted(v.keys()))            # ['batch_stats', 'params']

logits, new = model.apply(v, x, train=True, mutable=["batch_stats"])
v = {**v, "batch_stats": new["batch_stats"]}

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = nnx.Linear(2, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 32, rngs=rngs)
        self.fc3 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)

model = MLP(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-2))

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch14.md
# ────────────────────────────────────────────────────────────────────────
@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(2)(x)

model = MLP()
params = model.init(jax.random.key(0), jnp.ones((1, 2)))
print(model.apply(params, jnp.ones((8, 2))).shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp

for path, leaf in jax.tree.leaves_with_path(params):
    print(path, leaf.shape)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

model = Net()
v = model.init(jax.random.key(0), jnp.ones((1, 28, 28, 1)), train=True)
print(sorted(v.keys()))            # ['batch_stats', 'params']

logits, new = model.apply(v, x, train=True, mutable=["batch_stats"])
v = {**v, "batch_stats": new["batch_stats"]}

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from flax import nnx
import optax

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = nnx.Linear(2, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, 32, rngs=rngs)
        self.fc3 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)

model = MLP(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-2))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-14-ch14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss
