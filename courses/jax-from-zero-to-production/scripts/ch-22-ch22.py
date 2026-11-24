#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والعشرون: GPT صغير من الصفر
JAX: From Zero to Production — Chapter 22 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-22-ch22 (lesson 22)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-22-ch22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والعشرون: GPT صغير من الصفر
JAX: From Zero to Production — Chapter 22 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-22-ch22 (lesson 22)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-22-ch22.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والعشرون: GPT صغير من الصفر
JAX: From Zero to Production — Chapter 22 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-22-ch22 (lesson 22)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: jnp.array([stoi[c] for c in s], dtype=jnp.int32)
decode = lambda t: "".join(itos[int(i)] for i in t)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn
import jax
import jax.numpy as jnp

class Block(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, x, mask):
        d = x.shape[-1]
        h = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(z.shape[0], z.shape[1], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(d, name="proj")(a.reshape(a.shape[0], a.shape[1], d))
        h = nn.gelu(nn.Dense(4 * d, name="fc")(nn.LayerNorm()(x)))
        return x + nn.Dense(d, name="fc2")(h)

class GPT(nn.Module):
    vocab: int
    ctx: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 5

    @nn.compact
    def __call__(self, idx):
        b, t = idx.shape
        tok = nn.Embed(self.vocab, self.d_model, name="tok")(idx)      # (B,T,D)
        pos = self.param("pos", nn.initializers.normal(0.02), (self.ctx, self.d_model))
        x = tok + pos[:t]
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]      # (1,1,T,T)
        for _ in range(self.n_layers):
            x = Block(self.n_heads)(x, mask)
        return nn.Dense(self.vocab, name="head")(nn.LayerNorm()(x))    # (B,T,V)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
import optax

def loss_fn(params, idx):
    logits = model.apply({"params": params}, idx[:, :-1])
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, idx[:, 1:, None], axis=-1))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-3, warmup_steps=100,
    decay_steps=2000, end_value=3e-4)
tx = optax.chain(optax.clip_by_global_norm(1.0),
                 optax.adamw(schedule, weight_decay=0.01))
opt_state = tx.init(params)

@jax.jit
def step(params, opt_state, idx):
    loss, grads = jax.value_and_grad(loss_fn)(params, idx)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
def generate(params, key, prompt, n_new, temperature=1.0, top_k=None):
    def one(carry, _):
        window, key = carry
        logits = model.apply({"params": params}, window)[:, -1, :] / temperature
        if top_k is not None:
            thr = jnp.sort(logits, -1)[:, -top_k][:, None]
            logits = jnp.where(logits < thr, -jnp.inf, logits)
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits, axis=-1)
        window = jnp.concatenate([window[:, 1:], nxt[:, None]], axis=1)
        return (window, key), nxt
    (_, _), out = jax.lax.scan(one, (prompt, key), None, length=n_new)
    return out.T

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "tp"))
rep, shard_w = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "tp"))

params = jax.tree.map(lambda p: jax.device_put(p, shard_w if p.ndim > 1 else rep), params)
batch = jax.device_put(batch, NamedSharding(mesh, P("dp")))
params, opt_state, loss = jax.jit(step)(params, opt_state, batch)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: jnp.array([stoi[c] for c in s], dtype=jnp.int32)
decode = lambda t: "".join(itos[int(i)] for i in t)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn
import jax
import jax.numpy as jnp

class Block(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, x, mask):
        d = x.shape[-1]
        h = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(z.shape[0], z.shape[1], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(d, name="proj")(a.reshape(a.shape[0], a.shape[1], d))
        h = nn.gelu(nn.Dense(4 * d, name="fc")(nn.LayerNorm()(x)))
        return x + nn.Dense(d, name="fc2")(h)

class GPT(nn.Module):
    vocab: int
    ctx: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 5

    @nn.compact
    def __call__(self, idx):
        b, t = idx.shape
        tok = nn.Embed(self.vocab, self.d_model, name="tok")(idx)      # (B,T,D)
        pos = self.param("pos", nn.initializers.normal(0.02), (self.ctx, self.d_model))
        x = tok + pos[:t]
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]      # (1,1,T,T)
        for _ in range(self.n_layers):
            x = Block(self.n_heads)(x, mask)
        return nn.Dense(self.vocab, name="head")(nn.LayerNorm()(x))    # (B,T,V)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import optax

def loss_fn(params, idx):
    logits = model.apply({"params": params}, idx[:, :-1])
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, idx[:, 1:, None], axis=-1))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-3, warmup_steps=100,
    decay_steps=2000, end_value=3e-4)
tx = optax.chain(optax.clip_by_global_norm(1.0),
                 optax.adamw(schedule, weight_decay=0.01))
opt_state = tx.init(params)

@jax.jit
def step(params, opt_state, idx):
    loss, grads = jax.value_and_grad(loss_fn)(params, idx)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def generate(params, key, prompt, n_new, temperature=1.0, top_k=None):
    def one(carry, _):
        window, key = carry
        logits = model.apply({"params": params}, window)[:, -1, :] / temperature
        if top_k is not None:
            thr = jnp.sort(logits, -1)[:, -top_k][:, None]
            logits = jnp.where(logits < thr, -jnp.inf, logits)
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits, axis=-1)
        window = jnp.concatenate([window[:, 1:], nxt[:, None]], axis=1)
        return (window, key), nxt
    (_, _), out = jax.lax.scan(one, (prompt, key), None, length=n_new)
    return out.T

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "tp"))
rep, shard_w = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "tp"))

params = jax.tree.map(lambda p: jax.device_put(p, shard_w if p.ndim > 1 else rep), params)
batch = jax.device_put(batch, NamedSharding(mesh, P("dp")))
params, opt_state, loss = jax.jit(step)(params, opt_state, batch)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الثاني والعشرون: GPT صغير من الصفر
JAX: From Zero to Production — Chapter 22 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-22-ch22 (lesson 22)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: jnp.array([stoi[c] for c in s], dtype=jnp.int32)
decode = lambda t: "".join(itos[int(i)] for i in t)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn
import jax
import jax.numpy as jnp

class Block(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, x, mask):
        d = x.shape[-1]
        h = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(z.shape[0], z.shape[1], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(d, name="proj")(a.reshape(a.shape[0], a.shape[1], d))
        h = nn.gelu(nn.Dense(4 * d, name="fc")(nn.LayerNorm()(x)))
        return x + nn.Dense(d, name="fc2")(h)

class GPT(nn.Module):
    vocab: int
    ctx: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 5

    @nn.compact
    def __call__(self, idx):
        b, t = idx.shape
        tok = nn.Embed(self.vocab, self.d_model, name="tok")(idx)      # (B,T,D)
        pos = self.param("pos", nn.initializers.normal(0.02), (self.ctx, self.d_model))
        x = tok + pos[:t]
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]      # (1,1,T,T)
        for _ in range(self.n_layers):
            x = Block(self.n_heads)(x, mask)
        return nn.Dense(self.vocab, name="head")(nn.LayerNorm()(x))    # (B,T,V)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
import optax

def loss_fn(params, idx):
    logits = model.apply({"params": params}, idx[:, :-1])
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, idx[:, 1:, None], axis=-1))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-3, warmup_steps=100,
    decay_steps=2000, end_value=3e-4)
tx = optax.chain(optax.clip_by_global_norm(1.0),
                 optax.adamw(schedule, weight_decay=0.01))
opt_state = tx.init(params)

@jax.jit
def step(params, opt_state, idx):
    loss, grads = jax.value_and_grad(loss_fn)(params, idx)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
def generate(params, key, prompt, n_new, temperature=1.0, top_k=None):
    def one(carry, _):
        window, key = carry
        logits = model.apply({"params": params}, window)[:, -1, :] / temperature
        if top_k is not None:
            thr = jnp.sort(logits, -1)[:, -top_k][:, None]
            logits = jnp.where(logits < thr, -jnp.inf, logits)
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits, axis=-1)
        window = jnp.concatenate([window[:, 1:], nxt[:, None]], axis=1)
        return (window, key), nxt
    (_, _), out = jax.lax.scan(one, (prompt, key), None, length=n_new)
    return out.T

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch22.md
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "tp"))
rep, shard_w = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "tp"))

params = jax.tree.map(lambda p: jax.device_put(p, shard_w if p.ndim > 1 else rep), params)
batch = jax.device_put(batch, NamedSharding(mesh, P("dp")))
params, opt_state, loss = jax.jit(step)(params, opt_state, batch)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: jnp.array([stoi[c] for c in s], dtype=jnp.int32)
decode = lambda t: "".join(itos[int(i)] for i in t)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import flax.linen as nn
import jax
import jax.numpy as jnp

class Block(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, x, mask):
        d = x.shape[-1]
        h = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(z.shape[0], z.shape[1], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(d, name="proj")(a.reshape(a.shape[0], a.shape[1], d))
        h = nn.gelu(nn.Dense(4 * d, name="fc")(nn.LayerNorm()(x)))
        return x + nn.Dense(d, name="fc2")(h)

class GPT(nn.Module):
    vocab: int
    ctx: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 5

    @nn.compact
    def __call__(self, idx):
        b, t = idx.shape
        tok = nn.Embed(self.vocab, self.d_model, name="tok")(idx)      # (B,T,D)
        pos = self.param("pos", nn.initializers.normal(0.02), (self.ctx, self.d_model))
        x = tok + pos[:t]
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]      # (1,1,T,T)
        for _ in range(self.n_layers):
            x = Block(self.n_heads)(x, mask)
        return nn.Dense(self.vocab, name="head")(nn.LayerNorm()(x))    # (B,T,V)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import optax

def loss_fn(params, idx):
    logits = model.apply({"params": params}, idx[:, :-1])
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, idx[:, 1:, None], axis=-1))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-3, warmup_steps=100,
    decay_steps=2000, end_value=3e-4)
tx = optax.chain(optax.clip_by_global_norm(1.0),
                 optax.adamw(schedule, weight_decay=0.01))
opt_state = tx.init(params)

@jax.jit
def step(params, opt_state, idx):
    loss, grads = jax.value_and_grad(loss_fn)(params, idx)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def generate(params, key, prompt, n_new, temperature=1.0, top_k=None):
    def one(carry, _):
        window, key = carry
        logits = model.apply({"params": params}, window)[:, -1, :] / temperature
        if top_k is not None:
            thr = jnp.sort(logits, -1)[:, -top_k][:, None]
            logits = jnp.where(logits < thr, -jnp.inf, logits)
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits, axis=-1)
        window = jnp.concatenate([window[:, 1:], nxt[:, None]], axis=1)
        return (window, key), nxt
    (_, _), out = jax.lax.scan(one, (prompt, key), None, length=n_new)
    return out.T

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-22-ch22.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "tp"))
rep, shard_w = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "tp"))

params = jax.tree.map(lambda p: jax.device_put(p, shard_w if p.ndim > 1 else rep), params)
batch = jax.device_put(batch, NamedSharding(mesh, P("dp")))
params, opt_state, loss = jax.jit(step)(params, opt_state, batch)
