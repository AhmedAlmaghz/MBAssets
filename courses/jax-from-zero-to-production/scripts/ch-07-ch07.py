#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — grad: التفاضل التلقائي بلا سحر
JAX: From Zero to Production — Chapter 07 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-07-ch07 (lesson 7)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-07-ch07.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — grad: التفاضل التلقائي بلا سحر
JAX: From Zero to Production — Chapter 07 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-07-ch07 (lesson 7)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-07-ch07.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — grad: التفاضل التلقائي بلا سحر
JAX: From Zero to Production — Chapter 07 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-07-ch07 (lesson 7)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

f = lambda x: x ** 3 + 2 * x
print(jax.grad(f)(2.0))            # 14.0
print(jax.grad(jax.grad(f))(2.0))  # 12.0

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[0])])

x = jnp.array([2.0, 3.0])
y, pullback = jax.vjp(f, x)
print(y)                                   # [6.  0.909]
print(pullback(jnp.array([1.0, 0.0]))[0])  # [3. 2.]

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def g(x):
    return jnp.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

y, y_dot = jax.jvp(g, (jnp.array([2.0, 3.0]),), (jnp.array([1.0, 0.0]),))
print(y, y_dot)     # [4. 9. 6.] [4. 0. 3.]

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.mean(jnp.tanh(x @ w) ** 2)

w, x = jnp.ones((4, 4)) * 0.1, jnp.ones((8, 4))
value, grads = jax.value_and_grad(loss)(w, x)
print(value, grads.shape)

# معلومات إضافية بلا اشتقاق
def loss_aux(w, x):
    h = jnp.tanh(x @ w)
    return jnp.mean(h ** 2), h

(v, h), grads = jax.value_and_grad(loss_aux, has_aux=True)(w, x)
print(v, h.shape)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jnp.mean((h @ params["w2"]) ** 2)

params = {"w1": jnp.ones((4, 8)) * 0.1, "b1": jnp.zeros(8), "w2": jnp.ones((8, 1)) * 0.1}
g = jax.grad(loss)(params, jnp.ones((16, 4)))
print(g["b1"].shape)                     # (8,)
print(len(jax.tree.leaves(g)))           # 3

gx, gp = jax.grad(loss, argnums=(1, 0))(params, jnp.ones((16, 4)))
print(gx.shape, gp["w2"].shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

h = lambda v: jnp.array([v[0] ** 2, v[0] * v[1]])
v = jnp.array([2.0, 3.0])

print(jax.jacrev(h)(v))                        # [[3. 2.] [1. 1.]]
print(jax.jacfwd(h)(v))                        # النتيجة نفسها
print(jax.hessian(lambda z: jnp.sum(z ** 3))(v))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

f = lambda x: x ** 3 + 2 * x
print(jax.grad(f)(2.0))            # 14.0
print(jax.grad(jax.grad(f))(2.0))  # 12.0

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[0])])

x = jnp.array([2.0, 3.0])
y, pullback = jax.vjp(f, x)
print(y)                                   # [6.  0.909]
print(pullback(jnp.array([1.0, 0.0]))[0])  # [3. 2.]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def g(x):
    return jnp.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

y, y_dot = jax.jvp(g, (jnp.array([2.0, 3.0]),), (jnp.array([1.0, 0.0]),))
print(y, y_dot)     # [4. 9. 6.] [4. 0. 3.]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.mean(jnp.tanh(x @ w) ** 2)

w, x = jnp.ones((4, 4)) * 0.1, jnp.ones((8, 4))
value, grads = jax.value_and_grad(loss)(w, x)
print(value, grads.shape)

# معلومات إضافية بلا اشتقاق
def loss_aux(w, x):
    h = jnp.tanh(x @ w)
    return jnp.mean(h ** 2), h

(v, h), grads = jax.value_and_grad(loss_aux, has_aux=True)(w, x)
print(v, h.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jnp.mean((h @ params["w2"]) ** 2)

params = {"w1": jnp.ones((4, 8)) * 0.1, "b1": jnp.zeros(8), "w2": jnp.ones((8, 1)) * 0.1}
g = jax.grad(loss)(params, jnp.ones((16, 4)))
print(g["b1"].shape)                     # (8,)
print(len(jax.tree.leaves(g)))           # 3

gx, gp = jax.grad(loss, argnums=(1, 0))(params, jnp.ones((16, 4)))
print(gx.shape, gp["w2"].shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

h = lambda v: jnp.array([v[0] ** 2, v[0] * v[1]])
v = jnp.array([2.0, 3.0])

print(jax.jacrev(h)(v))                        # [[3. 2.] [1. 1.]]
print(jax.jacfwd(h)(v))                        # النتيجة نفسها
print(jax.hessian(lambda z: jnp.sum(z ** 3))(v))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — grad: التفاضل التلقائي بلا سحر
JAX: From Zero to Production — Chapter 07 — The Four Transformations

Book slug : jax-from-zero-to-production
Chapter   : ch-07-ch07 (lesson 7)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

f = lambda x: x ** 3 + 2 * x
print(jax.grad(f)(2.0))            # 14.0
print(jax.grad(jax.grad(f))(2.0))  # 12.0

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[0])])

x = jnp.array([2.0, 3.0])
y, pullback = jax.vjp(f, x)
print(y)                                   # [6.  0.909]
print(pullback(jnp.array([1.0, 0.0]))[0])  # [3. 2.]

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def g(x):
    return jnp.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

y, y_dot = jax.jvp(g, (jnp.array([2.0, 3.0]),), (jnp.array([1.0, 0.0]),))
print(y, y_dot)     # [4. 9. 6.] [4. 0. 3.]

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.mean(jnp.tanh(x @ w) ** 2)

w, x = jnp.ones((4, 4)) * 0.1, jnp.ones((8, 4))
value, grads = jax.value_and_grad(loss)(w, x)
print(value, grads.shape)

# معلومات إضافية بلا اشتقاق
def loss_aux(w, x):
    h = jnp.tanh(x @ w)
    return jnp.mean(h ** 2), h

(v, h), grads = jax.value_and_grad(loss_aux, has_aux=True)(w, x)
print(v, h.shape)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jnp.mean((h @ params["w2"]) ** 2)

params = {"w1": jnp.ones((4, 8)) * 0.1, "b1": jnp.zeros(8), "w2": jnp.ones((8, 1)) * 0.1}
g = jax.grad(loss)(params, jnp.ones((16, 4)))
print(g["b1"].shape)                     # (8,)
print(len(jax.tree.leaves(g)))           # 3

gx, gp = jax.grad(loss, argnums=(1, 0))(params, jnp.ones((16, 4)))
print(gx.shape, gp["w2"].shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch07.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

h = lambda v: jnp.array([v[0] ** 2, v[0] * v[1]])
v = jnp.array([2.0, 3.0])

print(jax.jacrev(h)(v))                        # [[3. 2.] [1. 1.]]
print(jax.jacfwd(h)(v))                        # النتيجة نفسها
print(jax.hessian(lambda z: jnp.sum(z ** 3))(v))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

f = lambda x: x ** 3 + 2 * x
print(jax.grad(f)(2.0))            # 14.0
print(jax.grad(jax.grad(f))(2.0))  # 12.0

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def f(x):
    return jnp.array([x[0] * x[1], jnp.sin(x[0])])

x = jnp.array([2.0, 3.0])
y, pullback = jax.vjp(f, x)
print(y)                                   # [6.  0.909]
print(pullback(jnp.array([1.0, 0.0]))[0])  # [3. 2.]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def g(x):
    return jnp.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

y, y_dot = jax.jvp(g, (jnp.array([2.0, 3.0]),), (jnp.array([1.0, 0.0]),))
print(y, y_dot)     # [4. 9. 6.] [4. 0. 3.]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(w, x):
    return jnp.mean(jnp.tanh(x @ w) ** 2)

w, x = jnp.ones((4, 4)) * 0.1, jnp.ones((8, 4))
value, grads = jax.value_and_grad(loss)(w, x)
print(value, grads.shape)

# معلومات إضافية بلا اشتقاق
def loss_aux(w, x):
    h = jnp.tanh(x @ w)
    return jnp.mean(h ** 2), h

(v, h), grads = jax.value_and_grad(loss_aux, has_aux=True)(w, x)
print(v, h.shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def loss(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jnp.mean((h @ params["w2"]) ** 2)

params = {"w1": jnp.ones((4, 8)) * 0.1, "b1": jnp.zeros(8), "w2": jnp.ones((8, 1)) * 0.1}
g = jax.grad(loss)(params, jnp.ones((16, 4)))
print(g["b1"].shape)                     # (8,)
print(len(jax.tree.leaves(g)))           # 3

gx, gp = jax.grad(loss, argnums=(1, 0))(params, jnp.ones((16, 4)))
print(gx.shape, gp["w2"].shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-07-ch07.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

h = lambda v: jnp.array([v[0] ** 2, v[0] * v[1]])
v = jnp.array([2.0, 3.0])

print(jax.jacrev(h)(v))                        # [[3. 2.] [1. 1.]]
print(jax.jacfwd(h)(v))                        # النتيجة نفسها
print(jax.hessian(lambda z: jnp.sum(z ** 3))(v))
