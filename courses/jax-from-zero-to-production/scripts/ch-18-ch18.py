#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن عشر: Equinox وjaxtyping: النمط البايثوني
JAX: From Zero to Production — Chapter 18 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-18-ch18 (lesson 18)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-18-ch18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن عشر: Equinox وjaxtyping: النمط البايثوني
JAX: From Zero to Production — Chapter 18 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-18-ch18 (lesson 18)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-18-ch18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن عشر: Equinox وjaxtyping: النمط البايثوني
JAX: From Zero to Production — Chapter 18 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-18-ch18 (lesson 18)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, equinox as eqx

class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    drop: float = eqx.field(static=True)   # ليست مصفوفة: جزء من البنية

    def __init__(self, key, din=784, dh=128, dout=10):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(din, dh, key=k1)
        self.l2 = eqx.nn.Linear(dh, dout, key=k2)
        self.drop = 0.1

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))

model = MLP(jax.random.key(0))
print(len(jax.tree.leaves(model)))   # 4: وزن وانحياز لكل طبقة

sizes = jax.tree.map(lambda a: a.size if eqx.is_array(a) else 0, model)
print(sum(jax.tree.leaves(sizes)))   # 101770

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
class Config(eqx.Module):
    w: jax.Array
    scale: float = eqx.field(static=True)

cfg = Config(w=jnp.ones((3,)), scale=2.0)
print(jax.tree.leaves(cfg))            # [Array([1., 1., 1.])]

@eqx.filter_jit
def f(c, x):
    return c.scale * (x @ c.w)

@eqx.filter_grad
def total(c, x):
    return jnp.sum(f(c, x))

g = total(cfg, jnp.arange(3.0))
print(g.w)        # [0. 2. 4.]
print(g.scale)    # None

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

@jaxtyped(typechecker=beartype)
def scores(
    q: Float[Array, "batch heads qlen dim"],
    k: Float[Array, "batch heads klen dim"],
) -> Float[Array, "batch heads qlen klen"]:
    return jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])

q = jnp.ones((2, 4, 8, 16))
print(scores(q, q).shape)              # (2, 4, 8, 8)
scores(q, jnp.ones((2, 4, 7, 16)))     # خطأ فوري

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
import optax

optim = optax.adamw(learning_rate=3e-4, weight_decay=1e-4)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

def loss_fn(m, x, y):
    logits = jax.vmap(m)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@eqx.filter_jit
def train_step(m, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(m, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(m, eqx.is_array))
    m = eqx.apply_updates(m, updates)
    return m, opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, equinox as eqx

class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    drop: float = eqx.field(static=True)   # ليست مصفوفة: جزء من البنية

    def __init__(self, key, din=784, dh=128, dout=10):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(din, dh, key=k1)
        self.l2 = eqx.nn.Linear(dh, dout, key=k2)
        self.drop = 0.1

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))

model = MLP(jax.random.key(0))
print(len(jax.tree.leaves(model)))   # 4: وزن وانحياز لكل طبقة

sizes = jax.tree.map(lambda a: a.size if eqx.is_array(a) else 0, model)
print(sum(jax.tree.leaves(sizes)))   # 101770

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class Config(eqx.Module):
    w: jax.Array
    scale: float = eqx.field(static=True)

cfg = Config(w=jnp.ones((3,)), scale=2.0)
print(jax.tree.leaves(cfg))            # [Array([1., 1., 1.])]

@eqx.filter_jit
def f(c, x):
    return c.scale * (x @ c.w)

@eqx.filter_grad
def total(c, x):
    return jnp.sum(f(c, x))

g = total(cfg, jnp.arange(3.0))
print(g.w)        # [0. 2. 4.]
print(g.scale)    # None

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

@jaxtyped(typechecker=beartype)
def scores(
    q: Float[Array, "batch heads qlen dim"],
    k: Float[Array, "batch heads klen dim"],
) -> Float[Array, "batch heads qlen klen"]:
    return jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])

q = jnp.ones((2, 4, 8, 16))
print(scores(q, q).shape)              # (2, 4, 8, 8)
scores(q, jnp.ones((2, 4, 7, 16)))     # خطأ فوري

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import optax

optim = optax.adamw(learning_rate=3e-4, weight_decay=1e-4)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

def loss_fn(m, x, y):
    logits = jax.vmap(m)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@eqx.filter_jit
def train_step(m, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(m, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(m, eqx.is_array))
    m = eqx.apply_updates(m, updates)
    return m, opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثامن عشر: Equinox وjaxtyping: النمط البايثوني
JAX: From Zero to Production — Chapter 18 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-18-ch18 (lesson 18)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, equinox as eqx

class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    drop: float = eqx.field(static=True)   # ليست مصفوفة: جزء من البنية

    def __init__(self, key, din=784, dh=128, dout=10):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(din, dh, key=k1)
        self.l2 = eqx.nn.Linear(dh, dout, key=k2)
        self.drop = 0.1

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))

model = MLP(jax.random.key(0))
print(len(jax.tree.leaves(model)))   # 4: وزن وانحياز لكل طبقة

sizes = jax.tree.map(lambda a: a.size if eqx.is_array(a) else 0, model)
print(sum(jax.tree.leaves(sizes)))   # 101770

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
class Config(eqx.Module):
    w: jax.Array
    scale: float = eqx.field(static=True)

cfg = Config(w=jnp.ones((3,)), scale=2.0)
print(jax.tree.leaves(cfg))            # [Array([1., 1., 1.])]

@eqx.filter_jit
def f(c, x):
    return c.scale * (x @ c.w)

@eqx.filter_grad
def total(c, x):
    return jnp.sum(f(c, x))

g = total(cfg, jnp.arange(3.0))
print(g.w)        # [0. 2. 4.]
print(g.scale)    # None

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

@jaxtyped(typechecker=beartype)
def scores(
    q: Float[Array, "batch heads qlen dim"],
    k: Float[Array, "batch heads klen dim"],
) -> Float[Array, "batch heads qlen klen"]:
    return jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])

q = jnp.ones((2, 4, 8, 16))
print(scores(q, q).shape)              # (2, 4, 8, 8)
scores(q, jnp.ones((2, 4, 7, 16)))     # خطأ فوري

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch18.md
# ────────────────────────────────────────────────────────────────────────
import optax

optim = optax.adamw(learning_rate=3e-4, weight_decay=1e-4)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

def loss_fn(m, x, y):
    logits = jax.vmap(m)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@eqx.filter_jit
def train_step(m, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(m, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(m, eqx.is_array))
    m = eqx.apply_updates(m, updates)
    return m, opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, equinox as eqx

class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    drop: float = eqx.field(static=True)   # ليست مصفوفة: جزء من البنية

    def __init__(self, key, din=784, dh=128, dout=10):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(din, dh, key=k1)
        self.l2 = eqx.nn.Linear(dh, dout, key=k2)
        self.drop = 0.1

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))

model = MLP(jax.random.key(0))
print(len(jax.tree.leaves(model)))   # 4: وزن وانحياز لكل طبقة

sizes = jax.tree.map(lambda a: a.size if eqx.is_array(a) else 0, model)
print(sum(jax.tree.leaves(sizes)))   # 101770

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class Config(eqx.Module):
    w: jax.Array
    scale: float = eqx.field(static=True)

cfg = Config(w=jnp.ones((3,)), scale=2.0)
print(jax.tree.leaves(cfg))            # [Array([1., 1., 1.])]

@eqx.filter_jit
def f(c, x):
    return c.scale * (x @ c.w)

@eqx.filter_grad
def total(c, x):
    return jnp.sum(f(c, x))

g = total(cfg, jnp.arange(3.0))
print(g.w)        # [0. 2. 4.]
print(g.scale)    # None

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

@jaxtyped(typechecker=beartype)
def scores(
    q: Float[Array, "batch heads qlen dim"],
    k: Float[Array, "batch heads klen dim"],
) -> Float[Array, "batch heads qlen klen"]:
    return jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])

q = jnp.ones((2, 4, 8, 16))
print(scores(q, q).shape)              # (2, 4, 8, 8)
scores(q, jnp.ones((2, 4, 7, 16)))     # خطأ فوري

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-18-ch18.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import optax

optim = optax.adamw(learning_rate=3e-4, weight_decay=1e-4)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

def loss_fn(m, x, y):
    logits = jax.vmap(m)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@eqx.filter_jit
def train_step(m, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(m, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(m, eqx.is_array))
    m = eqx.apply_updates(m, updates)
    return m, opt_state, loss
