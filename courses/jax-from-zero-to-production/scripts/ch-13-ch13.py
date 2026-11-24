#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث عشر: شبكة عصبية من الصفر بـ jax.numpy
JAX: From Zero to Production — MLP

Book slug : jax-from-zero-to-production
Chapter   : ch-13-ch13 (lesson 13)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-13-ch13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث عشر: شبكة عصبية من الصفر بـ jax.numpy
JAX: From Zero to Production — MLP

Book slug : jax-from-zero-to-production
Chapter   : ch-13-ch13 (lesson 13)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-13-ch13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث عشر: شبكة عصبية من الصفر بـ jax.numpy
JAX: From Zero to Production — MLP

Book slug : jax-from-zero-to-production
Chapter   : ch-13-ch13 (lesson 13)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

x, y = make_moons(512, jax.random.key(0))
print(x.shape, y.shape, y.mean())

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
def init_params(key, sizes):
    params, key = [], key
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m)   # He
        params.append({"W": W, "b": jnp.zeros(n)})
    return params

params = init_params(jax.random.key(1), [2, 64, 32, 2])
print([p["W"].shape for p in params])

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
def forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import jax.scipy as jsp

def loss_fn(params, x, y):
    logits = forward(params, x)                       # (B, 2)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    picked = logp[jnp.arange(y.shape[0]), y]          # (B,)
    return -jnp.mean(picked)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import optax

optimizer = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params, opt_state, l0 = train_step(params, opt_state, x, y)
print(round(float(l0), 4))

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax, jax.scipy as jsp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), minval=0.0, maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), minval=0.0, maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

def init_params(key, sizes):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append({"W": jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m),
                       "b": jnp.zeros(n)})
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]

def loss_fn(params, x, y):
    logits = forward(params, x)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

x, y = make_moons(512, jax.random.key(0))
params = init_params(jax.random.key(1), [2, 64, 32, 2])
opt = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
state = opt.init(params)

@jax.jit
def step(params, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss

for i in range(301):
    params, state, loss = step(params, state, x, y)
    if i % 50 == 0:
        acc = jnp.mean(jnp.argmax(forward(params, x), -1) == y)
        print(f"step {i:3d}  loss {loss:.4f}  acc {acc:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

x, y = make_moons(512, jax.random.key(0))
print(x.shape, y.shape, y.mean())

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def init_params(key, sizes):
    params, key = [], key
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m)   # He
        params.append({"W": W, "b": jnp.zeros(n)})
    return params

params = init_params(jax.random.key(1), [2, 64, 32, 2])
print([p["W"].shape for p in params])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax.scipy as jsp

def loss_fn(params, x, y):
    logits = forward(params, x)                       # (B, 2)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    picked = logp[jnp.arange(y.shape[0]), y]          # (B,)
    return -jnp.mean(picked)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import optax

optimizer = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params, opt_state, l0 = train_step(params, opt_state, x, y)
print(round(float(l0), 4))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax, jax.scipy as jsp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), minval=0.0, maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), minval=0.0, maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

def init_params(key, sizes):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append({"W": jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m),
                       "b": jnp.zeros(n)})
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]

def loss_fn(params, x, y):
    logits = forward(params, x)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

x, y = make_moons(512, jax.random.key(0))
params = init_params(jax.random.key(1), [2, 64, 32, 2])
opt = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
state = opt.init(params)

@jax.jit
def step(params, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss

for i in range(301):
    params, state, loss = step(params, state, x, y)
    if i % 50 == 0:
        acc = jnp.mean(jnp.argmax(forward(params, x), -1) == y)
        print(f"step {i:3d}  loss {loss:.4f}  acc {acc:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثالث عشر: شبكة عصبية من الصفر بـ jax.numpy
JAX: From Zero to Production — MLP

Book slug : jax-from-zero-to-production
Chapter   : ch-13-ch13 (lesson 13)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

x, y = make_moons(512, jax.random.key(0))
print(x.shape, y.shape, y.mean())

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
def init_params(key, sizes):
    params, key = [], key
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m)   # He
        params.append({"W": W, "b": jnp.zeros(n)})
    return params

params = init_params(jax.random.key(1), [2, 64, 32, 2])
print([p["W"].shape for p in params])

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
def forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import jax.scipy as jsp

def loss_fn(params, x, y):
    logits = forward(params, x)                       # (B, 2)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    picked = logp[jnp.arange(y.shape[0]), y]          # (B,)
    return -jnp.mean(picked)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import optax

optimizer = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params, opt_state, l0 = train_step(params, opt_state, x, y)
print(round(float(l0), 4))

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch13.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax, jax.scipy as jsp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), minval=0.0, maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), minval=0.0, maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

def init_params(key, sizes):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append({"W": jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m),
                       "b": jnp.zeros(n)})
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]

def loss_fn(params, x, y):
    logits = forward(params, x)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

x, y = make_moons(512, jax.random.key(0))
params = init_params(jax.random.key(1), [2, 64, 32, 2])
opt = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
state = opt.init(params)

@jax.jit
def step(params, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss

for i in range(301):
    params, state, loss = step(params, state, x, y)
    if i % 50 == 0:
        acc = jnp.mean(jnp.argmax(forward(params, x), -1) == y)
        print(f"step {i:3d}  loss {loss:.4f}  acc {acc:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

x, y = make_moons(512, jax.random.key(0))
print(x.shape, y.shape, y.mean())

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def init_params(key, sizes):
    params, key = [], key
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m)   # He
        params.append({"W": W, "b": jnp.zeros(n)})
    return params

params = init_params(jax.random.key(1), [2, 64, 32, 2])
print([p["W"].shape for p in params])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax.scipy as jsp

def loss_fn(params, x, y):
    logits = forward(params, x)                       # (B, 2)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    picked = logp[jnp.arange(y.shape[0]), y]          # (B,)
    return -jnp.mean(picked)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import optax

optimizer = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params, opt_state, l0 = train_step(params, opt_state, x, y)
print(round(float(l0), 4))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-13-ch13.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax, jax.scipy as jsp

def make_moons(n, key, noise=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    a = n // 2
    t0 = jax.random.uniform(k1, (a,), minval=0.0, maxval=jnp.pi)
    t1 = jax.random.uniform(k2, (n - a,), minval=0.0, maxval=jnp.pi)
    x0 = jnp.stack([jnp.cos(t0), jnp.sin(t0)], axis=1)
    x1 = jnp.stack([1.0 - jnp.cos(t1), 0.5 - jnp.sin(t1)], axis=1)
    x = jnp.concatenate([x0, x1]) + noise * jax.random.normal(k3, (n, 2))
    y = jnp.concatenate([jnp.zeros(a), jnp.ones(n - a)]).astype(jnp.int32)
    return x, y

def init_params(key, sizes):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append({"W": jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m),
                       "b": jnp.zeros(n)})
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["W"] + layer["b"])
    return x @ params[-1]["W"] + params[-1]["b"]

def loss_fn(params, x, y):
    logits = forward(params, x)
    logp = logits - jsp.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])

x, y = make_moons(512, jax.random.key(0))
params = init_params(jax.random.key(1), [2, 64, 32, 2])
opt = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
state = opt.init(params)

@jax.jit
def step(params, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss

for i in range(301):
    params, state, loss = step(params, state, x, y)
    if i % 50 == 0:
        acc = jnp.mean(jnp.argmax(forward(params, x), -1) == y)
        print(f"step {i:3d}  loss {loss:.4f}  acc {acc:.3f}")
