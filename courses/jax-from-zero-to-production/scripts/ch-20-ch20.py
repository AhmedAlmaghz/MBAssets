#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — التسلسلات: RNN وLSTM وGRU عبر scan
JAX: From Zero to Production — Chapter 20 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-20-ch20 (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-20-ch20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — التسلسلات: RNN وLSTM وGRU عبر scan
JAX: From Zero to Production — Chapter 20 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-20-ch20 (lesson 20)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-20-ch20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — التسلسلات: RNN وLSTM وGRU عبر scan
JAX: From Zero to Production — Chapter 20 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-20-ch20 (lesson 20)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

xs = jnp.array([1.0, 2.0, 3.0, 4.0])

def body(c, x):
    c = c + x
    return c, c          # (carry الجديد، مخرج هذه الخطوة)

total, ys = lax.scan(body, 0.0, xs)
print(total, ys)         # 10.0  [1. 3. 6. 10.]

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
def rnn_step(p, h, x):
    h = jnp.tanh(x @ p["Wx"] + h @ p["Wh"] + p["b"])
    return h, h

def run_rnn(p, xs, h0):
    # xs: (T, B, D)   h0: (B, H)   الخرج: (T, B, H)
    return lax.scan(lambda h, x: rnn_step(p, h, x), h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
import jax

def run_rnn_ckpt(p, xs, h0):
    step = jax.checkpoint(lambda h, x: rnn_step(p, h, x))   # إعادة حساب أثناء الرجوع
    return lax.scan(step, h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
def lstm_step(p, carry, x):
    h, c = carry
    z = x @ p["W"] + h @ p["U"] + p["b"]      # (B, 4H)
    i, f, g, o = jnp.split(z, 4, axis=-1)
    i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
    g = jnp.tanh(g)
    c = f * c + i * g
    h = o * jnp.tanh(c)
    return (h, c), h

def run_lstm(p, xs, h0, c0):
    (h, c), hs = lax.scan(lambda carry, x: lstm_step(p, carry, x), (h0, c0), xs)
    return hs, (h, c)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
def masked_step(p, carry, xm):
    x, m = xm                                   # m: (B,) منطقي
    (h, c), _ = lstm_step(p, carry, x)
    h = jnp.where(m[:, None], h, carry[0])      # جمّد الحالة عند الحشو
    c = jnp.where(m[:, None], c, carry[1])
    return (h, c), h

def sequence_loss(p, xs, ys, mask):
    hs, _ = lax.scan(lambda cr, xm: masked_step(p, cr, xm),
                     (jnp.zeros((B, H)), jnp.zeros((B, H))), (xs, mask))
    logits = hs @ p["Wo"] + p["bo"]             # (T, B, V)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, ys)
    loss = jnp.where(mask, loss, 0.0)
    return loss.sum() / mask.sum()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

xs = jnp.array([1.0, 2.0, 3.0, 4.0])

def body(c, x):
    c = c + x
    return c, c          # (carry الجديد، مخرج هذه الخطوة)

total, ys = lax.scan(body, 0.0, xs)
print(total, ys)         # 10.0  [1. 3. 6. 10.]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
def rnn_step(p, h, x):
    h = jnp.tanh(x @ p["Wx"] + h @ p["Wh"] + p["b"])
    return h, h

def run_rnn(p, xs, h0):
    # xs: (T, B, D)   h0: (B, H)   الخرج: (T, B, H)
    return lax.scan(lambda h, x: rnn_step(p, h, x), h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax

def run_rnn_ckpt(p, xs, h0):
    step = jax.checkpoint(lambda h, x: rnn_step(p, h, x))   # إعادة حساب أثناء الرجوع
    return lax.scan(step, h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def lstm_step(p, carry, x):
    h, c = carry
    z = x @ p["W"] + h @ p["U"] + p["b"]      # (B, 4H)
    i, f, g, o = jnp.split(z, 4, axis=-1)
    i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
    g = jnp.tanh(g)
    c = f * c + i * g
    h = o * jnp.tanh(c)
    return (h, c), h

def run_lstm(p, xs, h0, c0):
    (h, c), hs = lax.scan(lambda carry, x: lstm_step(p, carry, x), (h0, c0), xs)
    return hs, (h, c)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def masked_step(p, carry, xm):
    x, m = xm                                   # m: (B,) منطقي
    (h, c), _ = lstm_step(p, carry, x)
    h = jnp.where(m[:, None], h, carry[0])      # جمّد الحالة عند الحشو
    c = jnp.where(m[:, None], c, carry[1])
    return (h, c), h

def sequence_loss(p, xs, ys, mask):
    hs, _ = lax.scan(lambda cr, xm: masked_step(p, cr, xm),
                     (jnp.zeros((B, H)), jnp.zeros((B, H))), (xs, mask))
    logits = hs @ p["Wo"] + p["bo"]             # (T, B, V)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, ys)
    loss = jnp.where(mask, loss, 0.0)
    return loss.sum() / mask.sum()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — التسلسلات: RNN وLSTM وGRU عبر scan
JAX: From Zero to Production — Chapter 20 — Modern Neural Architectures

Book slug : jax-from-zero-to-production
Chapter   : ch-20-ch20 (lesson 20)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

xs = jnp.array([1.0, 2.0, 3.0, 4.0])

def body(c, x):
    c = c + x
    return c, c          # (carry الجديد، مخرج هذه الخطوة)

total, ys = lax.scan(body, 0.0, xs)
print(total, ys)         # 10.0  [1. 3. 6. 10.]

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
def rnn_step(p, h, x):
    h = jnp.tanh(x @ p["Wx"] + h @ p["Wh"] + p["b"])
    return h, h

def run_rnn(p, xs, h0):
    # xs: (T, B, D)   h0: (B, H)   الخرج: (T, B, H)
    return lax.scan(lambda h, x: rnn_step(p, h, x), h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
import jax

def run_rnn_ckpt(p, xs, h0):
    step = jax.checkpoint(lambda h, x: rnn_step(p, h, x))   # إعادة حساب أثناء الرجوع
    return lax.scan(step, h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
def lstm_step(p, carry, x):
    h, c = carry
    z = x @ p["W"] + h @ p["U"] + p["b"]      # (B, 4H)
    i, f, g, o = jnp.split(z, 4, axis=-1)
    i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
    g = jnp.tanh(g)
    c = f * c + i * g
    h = o * jnp.tanh(c)
    return (h, c), h

def run_lstm(p, xs, h0, c0):
    (h, c), hs = lax.scan(lambda carry, x: lstm_step(p, carry, x), (h0, c0), xs)
    return hs, (h, c)

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch20.md
# ────────────────────────────────────────────────────────────────────────
def masked_step(p, carry, xm):
    x, m = xm                                   # m: (B,) منطقي
    (h, c), _ = lstm_step(p, carry, x)
    h = jnp.where(m[:, None], h, carry[0])      # جمّد الحالة عند الحشو
    c = jnp.where(m[:, None], c, carry[1])
    return (h, c), h

def sequence_loss(p, xs, ys, mask):
    hs, _ = lax.scan(lambda cr, xm: masked_step(p, cr, xm),
                     (jnp.zeros((B, H)), jnp.zeros((B, H))), (xs, mask))
    logits = hs @ p["Wo"] + p["bo"]             # (T, B, V)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, ys)
    loss = jnp.where(mask, loss, 0.0)
    return loss.sum() / mask.sum()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax.numpy as jnp
from jax import lax

xs = jnp.array([1.0, 2.0, 3.0, 4.0])

def body(c, x):
    c = c + x
    return c, c          # (carry الجديد، مخرج هذه الخطوة)

total, ys = lax.scan(body, 0.0, xs)
print(total, ys)         # 10.0  [1. 3. 6. 10.]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def rnn_step(p, h, x):
    h = jnp.tanh(x @ p["Wx"] + h @ p["Wh"] + p["b"])
    return h, h

def run_rnn(p, xs, h0):
    # xs: (T, B, D)   h0: (B, H)   الخرج: (T, B, H)
    return lax.scan(lambda h, x: rnn_step(p, h, x), h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax

def run_rnn_ckpt(p, xs, h0):
    step = jax.checkpoint(lambda h, x: rnn_step(p, h, x))   # إعادة حساب أثناء الرجوع
    return lax.scan(step, h0, xs)[1]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def lstm_step(p, carry, x):
    h, c = carry
    z = x @ p["W"] + h @ p["U"] + p["b"]      # (B, 4H)
    i, f, g, o = jnp.split(z, 4, axis=-1)
    i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
    g = jnp.tanh(g)
    c = f * c + i * g
    h = o * jnp.tanh(c)
    return (h, c), h

def run_lstm(p, xs, h0, c0):
    (h, c), hs = lax.scan(lambda carry, x: lstm_step(p, carry, x), (h0, c0), xs)
    return hs, (h, c)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-20-ch20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def masked_step(p, carry, xm):
    x, m = xm                                   # m: (B,) منطقي
    (h, c), _ = lstm_step(p, carry, x)
    h = jnp.where(m[:, None], h, carry[0])      # جمّد الحالة عند الحشو
    c = jnp.where(m[:, None], c, carry[1])
    return (h, c), h

def sequence_loss(p, xs, ys, mask):
    hs, _ = lax.scan(lambda cr, xm: masked_step(p, cr, xm),
                     (jnp.zeros((B, H)), jnp.zeros((B, H))), (xs, mask))
    logits = hs @ p["Wo"] + p["bo"]             # (T, B, V)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, ys)
    loss = jnp.where(mask, loss, 0.0)
    return loss.sum() / mask.sum()
