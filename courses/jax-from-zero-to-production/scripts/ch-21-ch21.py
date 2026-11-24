#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والعشرون: الانتباه والمحوّلات
JAX: From Zero to Production — Chapter 21 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-21-ch21 (lesson 21)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-21-ch21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والعشرون: الانتباه والمحوّلات
JAX: From Zero to Production — Chapter 21 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-21-ch21 (lesson 21)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-21-ch21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والعشرون: الانتباه والمحوّلات
JAX: From Zero to Production — Chapter 21 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-21-ch21 (lesson 21)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
def attention(q, k, v, mask=None):
    """q: (B,T,H,D)  k,v: (B,S,H,D)  mask: (B,1,T,S)"""
    s = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    if mask is not None:
        s = jnp.where(mask, s, -jnp.inf)     # قبل softmax لا بعده
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", p, v)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
causal = jnp.tril(jnp.ones((t, t), dtype=bool))          # (T,S)
pad = tokens != 0                                        # (B,S)
m = causal[None, None] & pad[:, None, None, :]           # (B,1,T,S)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
def rope(x, offset=0):
    t, d = x.shape[1], x.shape[-1]
    ang = (jnp.arange(t) + offset)[:, None] * (10000.0 ** (-jnp.arange(0, d, 2) / d))[None, :]
    cos, sin = jnp.cos(ang)[:, None], jnp.sin(ang)[:, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        h = nn.LayerNorm()(x)                              # pre-LN
        qkv = nn.Dense(3 * self.d_model, use_bias=False)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(*z.shape[:2], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(self.d_model)(a.reshape(*a.shape[:2], self.d_model))
        h = nn.gelu(nn.Dense(4 * self.d_model)(nn.LayerNorm()(x)))
        return x + nn.Dense(self.d_model)(h)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
def attention(q, k, v, mask=None):
    """q: (B,T,H,D)  k,v: (B,S,H,D)  mask: (B,1,T,S)"""
    s = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    if mask is not None:
        s = jnp.where(mask, s, -jnp.inf)     # قبل softmax لا بعده
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", p, v)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
causal = jnp.tril(jnp.ones((t, t), dtype=bool))          # (T,S)
pad = tokens != 0                                        # (B,S)
m = causal[None, None] & pad[:, None, None, :]           # (B,1,T,S)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def rope(x, offset=0):
    t, d = x.shape[1], x.shape[-1]
    ang = (jnp.arange(t) + offset)[:, None] * (10000.0 ** (-jnp.arange(0, d, 2) / d))[None, :]
    cos, sin = jnp.cos(ang)[:, None], jnp.sin(ang)[:, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        h = nn.LayerNorm()(x)                              # pre-LN
        qkv = nn.Dense(3 * self.d_model, use_bias=False)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(*z.shape[:2], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(self.d_model)(a.reshape(*a.shape[:2], self.d_model))
        h = nn.gelu(nn.Dense(4 * self.d_model)(nn.LayerNorm()(x)))
        return x + nn.Dense(self.d_model)(h)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والعشرون: الانتباه والمحوّلات
JAX: From Zero to Production — Chapter 21 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-21-ch21 (lesson 21)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
def attention(q, k, v, mask=None):
    """q: (B,T,H,D)  k,v: (B,S,H,D)  mask: (B,1,T,S)"""
    s = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    if mask is not None:
        s = jnp.where(mask, s, -jnp.inf)     # قبل softmax لا بعده
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", p, v)

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
causal = jnp.tril(jnp.ones((t, t), dtype=bool))          # (T,S)
pad = tokens != 0                                        # (B,S)
m = causal[None, None] & pad[:, None, None, :]           # (B,1,T,S)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
def rope(x, offset=0):
    t, d = x.shape[1], x.shape[-1]
    ang = (jnp.arange(t) + offset)[:, None] * (10000.0 ** (-jnp.arange(0, d, 2) / d))[None, :]
    cos, sin = jnp.cos(ang)[:, None], jnp.sin(ang)[:, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch21.md
# ────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        h = nn.LayerNorm()(x)                              # pre-LN
        qkv = nn.Dense(3 * self.d_model, use_bias=False)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(*z.shape[:2], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(self.d_model)(a.reshape(*a.shape[:2], self.d_model))
        h = nn.gelu(nn.Dense(4 * self.d_model)(nn.LayerNorm()(x)))
        return x + nn.Dense(self.d_model)(h)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def attention(q, k, v, mask=None):
    """q: (B,T,H,D)  k,v: (B,S,H,D)  mask: (B,1,T,S)"""
    s = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(q.shape[-1])
    if mask is not None:
        s = jnp.where(mask, s, -jnp.inf)     # قبل softmax لا بعده
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhts,bshd->bthd", p, v)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
causal = jnp.tril(jnp.ones((t, t), dtype=bool))          # (T,S)
pad = tokens != 0                                        # (B,S)
m = causal[None, None] & pad[:, None, None, :]           # (B,1,T,S)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def rope(x, offset=0):
    t, d = x.shape[1], x.shape[-1]
    ang = (jnp.arange(t) + offset)[:, None] * (10000.0 ** (-jnp.arange(0, d, 2) / d))[None, :]
    cos, sin = jnp.cos(ang)[:, None], jnp.sin(ang)[:, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-21-ch21.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        h = nn.LayerNorm()(x)                              # pre-LN
        qkv = nn.Dense(3 * self.d_model, use_bias=False)(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (z.reshape(*z.shape[:2], self.n_heads, -1) for z in (q, k, v))
        a = jax.nn.dot_product_attention(q, k, v, mask=mask)
        x = x + nn.Dense(self.d_model)(a.reshape(*a.shape[:2], self.d_model))
        h = nn.gelu(nn.Dense(4 * self.d_model)(nn.LayerNorm()(x)))
        return x + nn.Dense(self.d_model)(h)
