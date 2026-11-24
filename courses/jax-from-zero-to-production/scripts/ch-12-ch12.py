#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني عشر: تركيب التبديلات: القواعد الذهبية
JAX: From Zero to Production — jit(grad(vmap(f)))

Book slug : jax-from-zero-to-production
Chapter   : ch-12-ch12 (lesson 12)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-12-ch12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني عشر: تركيب التبديلات: القواعد الذهبية
JAX: From Zero to Production — jit(grad(vmap(f)))

Book slug : jax-from-zero-to-production
Chapter   : ch-12-ch12 (lesson 12)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-12-ch12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني عشر: تركيب التبديلات: القواعد الذهبية
JAX: From Zero to Production — jit(grad(vmap(f)))

Book slug : jax-from-zero-to-production
Chapter   : ch-12-ch12 (lesson 12)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

jax.config.update("jax_log_compiles", True)  # سطر عند كل تجميع

def f(x):
    return jnp.tanh(x) @ jnp.ones((4, 1))

A = jax.jit(jax.vmap(f))   # الدفعة كلها داخل التجميع
B = jax.vmap(jax.jit(f))   # f وحدها داخل التجميع

for n in (8, 16, 32):
    x = jnp.ones((n, 4))
    print("A", n, A(x).shape, "B", n, B(x).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def loss(params, x, y):
    logits = x @ params["w"] + params["b"]
    l = jnp.mean((logits - y) ** 2)
    return l, logits          # الخسارة + ملحق

params = {"w": jnp.zeros((4, 1)), "b": jnp.zeros(1)}
x, y = jnp.ones((2, 4)), jnp.ones((2, 1))

per_example = jax.vmap(jax.grad(loss, has_aux=True), in_axes=(None, 0, 0))
grads, logits = per_example(params, x, y)
print(jax.tree.map(lambda g: g.shape, grads))
print(logits.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.custom_vjp
def softplus(x):
    return jnp.logaddexp(0.0, x)      # أمامي مستقرّ

def softplus_fwd(x):
    y = jnp.logaddexp(0.0, x)
    return y, y                        # نُخزّن y للعكسي

def softplus_bwd(y, g):
    # d/dx softplus(x) = sigmoid(x) = 1 - exp(-softplus(x))
    return (g * (1.0 - jnp.exp(-y)),)

softplus.defvjp(softplus_fwd, softplus_bwd)

print(softplus(30.0), jax.grad(softplus)(30.0))   # 30.0 1.0
print(jax.grad(softplus)(-30.0))                   # ~1.8e-13

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

@custom_partitioning
def row_softmax(x):
    return jax.nn.softmax(x, axis=-1)

def _partition(mesh, arg_shapes, result_shape):
    s = NamedSharding(mesh, P(None, "tp"))
    return mesh, row_softmax, s, s   # الشبكة، الدالّة، تقطيع المدخل، تقطيع الخرج

row_softmax.def_partition(partition=_partition)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

jax.config.update("jax_log_compiles", True)  # سطر عند كل تجميع

def f(x):
    return jnp.tanh(x) @ jnp.ones((4, 1))

A = jax.jit(jax.vmap(f))   # الدفعة كلها داخل التجميع
B = jax.vmap(jax.jit(f))   # f وحدها داخل التجميع

for n in (8, 16, 32):
    x = jnp.ones((n, 4))
    print("A", n, A(x).shape, "B", n, B(x).shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def loss(params, x, y):
    logits = x @ params["w"] + params["b"]
    l = jnp.mean((logits - y) ** 2)
    return l, logits          # الخسارة + ملحق

params = {"w": jnp.zeros((4, 1)), "b": jnp.zeros(1)}
x, y = jnp.ones((2, 4)), jnp.ones((2, 1))

per_example = jax.vmap(jax.grad(loss, has_aux=True), in_axes=(None, 0, 0))
grads, logits = per_example(params, x, y)
print(jax.tree.map(lambda g: g.shape, grads))
print(logits.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.custom_vjp
def softplus(x):
    return jnp.logaddexp(0.0, x)      # أمامي مستقرّ

def softplus_fwd(x):
    y = jnp.logaddexp(0.0, x)
    return y, y                        # نُخزّن y للعكسي

def softplus_bwd(y, g):
    # d/dx softplus(x) = sigmoid(x) = 1 - exp(-softplus(x))
    return (g * (1.0 - jnp.exp(-y)),)

softplus.defvjp(softplus_fwd, softplus_bwd)

print(softplus(30.0), jax.grad(softplus)(30.0))   # 30.0 1.0
print(jax.grad(softplus)(-30.0))                   # ~1.8e-13

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

@custom_partitioning
def row_softmax(x):
    return jax.nn.softmax(x, axis=-1)

def _partition(mesh, arg_shapes, result_shape):
    s = NamedSharding(mesh, P(None, "tp"))
    return mesh, row_softmax, s, s   # الشبكة، الدالّة، تقطيع المدخل، تقطيع الخرج

row_softmax.def_partition(partition=_partition)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني عشر: تركيب التبديلات: القواعد الذهبية
JAX: From Zero to Production — jit(grad(vmap(f)))

Book slug : jax-from-zero-to-production
Chapter   : ch-12-ch12 (lesson 12)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

jax.config.update("jax_log_compiles", True)  # سطر عند كل تجميع

def f(x):
    return jnp.tanh(x) @ jnp.ones((4, 1))

A = jax.jit(jax.vmap(f))   # الدفعة كلها داخل التجميع
B = jax.vmap(jax.jit(f))   # f وحدها داخل التجميع

for n in (8, 16, 32):
    x = jnp.ones((n, 4))
    print("A", n, A(x).shape, "B", n, B(x).shape)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def loss(params, x, y):
    logits = x @ params["w"] + params["b"]
    l = jnp.mean((logits - y) ** 2)
    return l, logits          # الخسارة + ملحق

params = {"w": jnp.zeros((4, 1)), "b": jnp.zeros(1)}
x, y = jnp.ones((2, 4)), jnp.ones((2, 1))

per_example = jax.vmap(jax.grad(loss, has_aux=True), in_axes=(None, 0, 0))
grads, logits = per_example(params, x, y)
print(jax.tree.map(lambda g: g.shape, grads))
print(logits.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.custom_vjp
def softplus(x):
    return jnp.logaddexp(0.0, x)      # أمامي مستقرّ

def softplus_fwd(x):
    y = jnp.logaddexp(0.0, x)
    return y, y                        # نُخزّن y للعكسي

def softplus_bwd(y, g):
    # d/dx softplus(x) = sigmoid(x) = 1 - exp(-softplus(x))
    return (g * (1.0 - jnp.exp(-y)),)

softplus.defvjp(softplus_fwd, softplus_bwd)

print(softplus(30.0), jax.grad(softplus)(30.0))   # 30.0 1.0
print(jax.grad(softplus)(-30.0))                   # ~1.8e-13

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch12.md
# ────────────────────────────────────────────────────────────────────────
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

@custom_partitioning
def row_softmax(x):
    return jax.nn.softmax(x, axis=-1)

def _partition(mesh, arg_shapes, result_shape):
    s = NamedSharding(mesh, P(None, "tp"))
    return mesh, row_softmax, s, s   # الشبكة، الدالّة، تقطيع المدخل، تقطيع الخرج

row_softmax.def_partition(partition=_partition)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

jax.config.update("jax_log_compiles", True)  # سطر عند كل تجميع

def f(x):
    return jnp.tanh(x) @ jnp.ones((4, 1))

A = jax.jit(jax.vmap(f))   # الدفعة كلها داخل التجميع
B = jax.vmap(jax.jit(f))   # f وحدها داخل التجميع

for n in (8, 16, 32):
    x = jnp.ones((n, 4))
    print("A", n, A(x).shape, "B", n, B(x).shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def loss(params, x, y):
    logits = x @ params["w"] + params["b"]
    l = jnp.mean((logits - y) ** 2)
    return l, logits          # الخسارة + ملحق

params = {"w": jnp.zeros((4, 1)), "b": jnp.zeros(1)}
x, y = jnp.ones((2, 4)), jnp.ones((2, 1))

per_example = jax.vmap(jax.grad(loss, has_aux=True), in_axes=(None, 0, 0))
grads, logits = per_example(params, x, y)
print(jax.tree.map(lambda g: g.shape, grads))
print(logits.shape)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

@jax.custom_vjp
def softplus(x):
    return jnp.logaddexp(0.0, x)      # أمامي مستقرّ

def softplus_fwd(x):
    y = jnp.logaddexp(0.0, x)
    return y, y                        # نُخزّن y للعكسي

def softplus_bwd(y, g):
    # d/dx softplus(x) = sigmoid(x) = 1 - exp(-softplus(x))
    return (g * (1.0 - jnp.exp(-y)),)

softplus.defvjp(softplus_fwd, softplus_bwd)

print(softplus(30.0), jax.grad(softplus)(30.0))   # 30.0 1.0
print(jax.grad(softplus)(-30.0))                   # ~1.8e-13

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-12-ch12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

@custom_partitioning
def row_softmax(x):
    return jax.nn.softmax(x, axis=-1)

def _partition(mesh, arg_shapes, result_shape):
    s = NamedSharding(mesh, P(None, "tp"))
    return mesh, row_softmax, s, s   # الشبكة، الدالّة، تقطيع المدخل، تقطيع الخرج

row_softmax.def_partition(partition=_partition)
