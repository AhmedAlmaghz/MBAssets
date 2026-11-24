#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع عشر: الشبكات التلافيفية في JAX
JAX: From Zero to Production — Chapter 19 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-19-ch19 (lesson 19)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-19-ch19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع عشر: الشبكات التلافيفية في JAX
JAX: From Zero to Production — Chapter 19 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-19-ch19 (lesson 19)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-19-ch19.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع عشر: الشبكات التلافيفية في JAX
JAX: From Zero to Production — Chapter 19 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-19-ch19 (lesson 19)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch19.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

x = jnp.arange(25.0).reshape(1, 5, 5, 1)      # NHWC
w = jnp.ones((3, 3, 1, 4)) / 9.0              # HWIO

y = lax.conv_general_dilated(
    x, w,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
print(y.shape)          # (1, 5, 5, 4): الحجم محفوظ، القنوات أربع
print(y[0, 2, 2, 0])    # 12.0 — متوسط النافذة المركزية

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch19.md
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn

class Block(nn.Module):
    ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(self.ch, (3, 3), (self.stride, self.stride),
                    padding="SAME", use_bias=False)(x)
        y = nn.relu(nn.BatchNorm(use_running_average=not train)(y))
        y = nn.Conv(self.ch, (3, 3), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        if self.stride != 1 or x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), (self.stride, self.stride),
                        use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + y)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch19.md
# ────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = jnp.mean(x, axis=(1, 2))          # تجميع عام: يمحو المكان
        return nn.Dense(10)(x)

def loss_fn(params, x, y):
    logits = model.apply({"params": params}, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optim.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

x = jnp.arange(25.0).reshape(1, 5, 5, 1)      # NHWC
w = jnp.ones((3, 3, 1, 4)) / 9.0              # HWIO

y = lax.conv_general_dilated(
    x, w,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
print(y.shape)          # (1, 5, 5, 4): الحجم محفوظ، القنوات أربع
print(y[0, 2, 2, 0])    # 12.0 — متوسط النافذة المركزية

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn

class Block(nn.Module):
    ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(self.ch, (3, 3), (self.stride, self.stride),
                    padding="SAME", use_bias=False)(x)
        y = nn.relu(nn.BatchNorm(use_running_average=not train)(y))
        y = nn.Conv(self.ch, (3, 3), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        if self.stride != 1 or x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), (self.stride, self.stride),
                        use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + y)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = jnp.mean(x, axis=(1, 2))          # تجميع عام: يمحو المكان
        return nn.Dense(10)(x)

def loss_fn(params, x, y):
    logits = model.apply({"params": params}, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optim.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع عشر: الشبكات التلافيفية في JAX
JAX: From Zero to Production — Chapter 19 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-19-ch19 (lesson 19)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch19.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

x = jnp.arange(25.0).reshape(1, 5, 5, 1)      # NHWC
w = jnp.ones((3, 3, 1, 4)) / 9.0              # HWIO

y = lax.conv_general_dilated(
    x, w,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
print(y.shape)          # (1, 5, 5, 4): الحجم محفوظ، القنوات أربع
print(y[0, 2, 2, 0])    # 12.0 — متوسط النافذة المركزية

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch19.md
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn

class Block(nn.Module):
    ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(self.ch, (3, 3), (self.stride, self.stride),
                    padding="SAME", use_bias=False)(x)
        y = nn.relu(nn.BatchNorm(use_running_average=not train)(y))
        y = nn.Conv(self.ch, (3, 3), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        if self.stride != 1 or x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), (self.stride, self.stride),
                        use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + y)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch19.md
# ────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = jnp.mean(x, axis=(1, 2))          # تجميع عام: يمحو المكان
        return nn.Dense(10)(x)

def loss_fn(params, x, y):
    logits = model.apply({"params": params}, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optim.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

x = jnp.arange(25.0).reshape(1, 5, 5, 1)      # NHWC
w = jnp.ones((3, 3, 1, 4)) / 9.0              # HWIO

y = lax.conv_general_dilated(
    x, w,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
print(y.shape)          # (1, 5, 5, 4): الحجم محفوظ، القنوات أربع
print(y[0, 2, 2, 0])    # 12.0 — متوسط النافذة المركزية

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn

class Block(nn.Module):
    ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(self.ch, (3, 3), (self.stride, self.stride),
                    padding="SAME", use_bias=False)(x)
        y = nn.relu(nn.BatchNorm(use_running_average=not train)(y))
        y = nn.Conv(self.ch, (3, 3), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        if self.stride != 1 or x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), (self.stride, self.stride),
                        use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + y)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-19-ch19.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = jnp.mean(x, axis=(1, 2))          # تجميع عام: يمحو المكان
        return nn.Dense(10)(x)

def loss_fn(params, x, y):
    logits = model.apply({"params": params}, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optim.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
