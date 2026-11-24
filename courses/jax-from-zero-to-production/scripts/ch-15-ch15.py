#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس عشر: Optax: المحسّنات كتحويلات
JAX: From Zero to Production — Chapter 15 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-15-ch15 (lesson 15)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-15-ch15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس عشر: Optax: المحسّنات كتحويلات
JAX: From Zero to Production — Chapter 15 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-15-ch15 (lesson 15)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-15-ch15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس عشر: Optax: المحسّنات كتحويلات
JAX: From Zero to Production — Chapter 15 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-15-ch15 (lesson 15)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

params = {"w": jnp.ones((4, 3)), "b": jnp.zeros((3,))}
tx = optax.sgd(learning_rate=0.1, momentum=0.9)

state = tx.init(params)                 # أصفار بأشكالها
print(state[1].trace["w"].shape)        # (4, 3)

grads = jax.tree.map(jnp.ones_like, params)
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)
print(params["w"][0])                   # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
tx = optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.1)
# المكافئ اليدوي، للفهم لا للاستخدام:
# tx = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(0.1),
#                  optax.scale_by_learning_rate(3e-4))

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
warm = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4, warmup_steps=500,
    decay_steps=20_000, end_value=1e-5)
cos = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=10_000, alpha=0.0)
exp = optax.exponential_decay(init_value=1e-2, transition_steps=1_000, decay_rate=0.5)

for step in (0, 250, 500, 5_000):
    print(step, float(warm(step)))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, weight_decay=0.1)
state = tx.init(params)
state = state._replace(
    hyperparams={**state.hyperparams, "learning_rate": jnp.asarray(2e-4)})

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
tx = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.adamw(learning_rate=3e-4, weight_decay=0.1),
)
# بديل تكيّفي: يقيّد نسبة التحديث إلى حجم المعامل نفسه
tx_agc = optax.chain(
    optax.adaptive_grad_clip(clipping=0.01, eps=1e-3),
    optax.sgd(learning_rate=0.1, momentum=0.9),
)

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
def label(p):
    return jax.tree.map(lambda x: "decay" if x.ndim > 1 else "no_decay", p)

tx = optax.multi_transform(
    {"decay": optax.adamw(1e-3, weight_decay=0.1),
     "no_decay": optax.adamw(1e-3, weight_decay=0.0)},
    label)

state = tx.init(params)      # حالة لكل فرع، مدمجة في شجرة واحدة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

params = {"w": jnp.ones((4, 3)), "b": jnp.zeros((3,))}
tx = optax.sgd(learning_rate=0.1, momentum=0.9)

state = tx.init(params)                 # أصفار بأشكالها
print(state[1].trace["w"].shape)        # (4, 3)

grads = jax.tree.map(jnp.ones_like, params)
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)
print(params["w"][0])                   # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
tx = optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.1)
# المكافئ اليدوي، للفهم لا للاستخدام:
# tx = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(0.1),
#                  optax.scale_by_learning_rate(3e-4))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
warm = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4, warmup_steps=500,
    decay_steps=20_000, end_value=1e-5)
cos = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=10_000, alpha=0.0)
exp = optax.exponential_decay(init_value=1e-2, transition_steps=1_000, decay_rate=0.5)

for step in (0, 250, 500, 5_000):
    print(step, float(warm(step)))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, weight_decay=0.1)
state = tx.init(params)
state = state._replace(
    hyperparams={**state.hyperparams, "learning_rate": jnp.asarray(2e-4)})

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
tx = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.adamw(learning_rate=3e-4, weight_decay=0.1),
)
# بديل تكيّفي: يقيّد نسبة التحديث إلى حجم المعامل نفسه
tx_agc = optax.chain(
    optax.adaptive_grad_clip(clipping=0.01, eps=1e-3),
    optax.sgd(learning_rate=0.1, momentum=0.9),
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def label(p):
    return jax.tree.map(lambda x: "decay" if x.ndim > 1 else "no_decay", p)

tx = optax.multi_transform(
    {"decay": optax.adamw(1e-3, weight_decay=0.1),
     "no_decay": optax.adamw(1e-3, weight_decay=0.0)},
    label)

state = tx.init(params)      # حالة لكل فرع، مدمجة في شجرة واحدة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الخامس عشر: Optax: المحسّنات كتحويلات
JAX: From Zero to Production — Chapter 15 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-15-ch15 (lesson 15)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

params = {"w": jnp.ones((4, 3)), "b": jnp.zeros((3,))}
tx = optax.sgd(learning_rate=0.1, momentum=0.9)

state = tx.init(params)                 # أصفار بأشكالها
print(state[1].trace["w"].shape)        # (4, 3)

grads = jax.tree.map(jnp.ones_like, params)
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)
print(params["w"][0])                   # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
tx = optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.1)
# المكافئ اليدوي، للفهم لا للاستخدام:
# tx = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(0.1),
#                  optax.scale_by_learning_rate(3e-4))

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
warm = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4, warmup_steps=500,
    decay_steps=20_000, end_value=1e-5)
cos = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=10_000, alpha=0.0)
exp = optax.exponential_decay(init_value=1e-2, transition_steps=1_000, decay_rate=0.5)

for step in (0, 250, 500, 5_000):
    print(step, float(warm(step)))

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, weight_decay=0.1)
state = tx.init(params)
state = state._replace(
    hyperparams={**state.hyperparams, "learning_rate": jnp.asarray(2e-4)})

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
tx = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.adamw(learning_rate=3e-4, weight_decay=0.1),
)
# بديل تكيّفي: يقيّد نسبة التحديث إلى حجم المعامل نفسه
tx_agc = optax.chain(
    optax.adaptive_grad_clip(clipping=0.01, eps=1e-3),
    optax.sgd(learning_rate=0.1, momentum=0.9),
)

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
def label(p):
    return jax.tree.map(lambda x: "decay" if x.ndim > 1 else "no_decay", p)

tx = optax.multi_transform(
    {"decay": optax.adamw(1e-3, weight_decay=0.1),
     "no_decay": optax.adamw(1e-3, weight_decay=0.0)},
    label)

state = tx.init(params)      # حالة لكل فرع، مدمجة في شجرة واحدة

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp, optax

params = {"w": jnp.ones((4, 3)), "b": jnp.zeros((3,))}
tx = optax.sgd(learning_rate=0.1, momentum=0.9)

state = tx.init(params)                 # أصفار بأشكالها
print(state[1].trace["w"].shape)        # (4, 3)

grads = jax.tree.map(jnp.ones_like, params)
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)
print(params["w"][0])                   # [0.9 0.9 0.9]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
tx = optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.1)
# المكافئ اليدوي، للفهم لا للاستخدام:
# tx = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(0.1),
#                  optax.scale_by_learning_rate(3e-4))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
warm = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4, warmup_steps=500,
    decay_steps=20_000, end_value=1e-5)
cos = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=10_000, alpha=0.0)
exp = optax.exponential_decay(init_value=1e-2, transition_steps=1_000, decay_rate=0.5)

for step in (0, 250, 500, 5_000):
    print(step, float(warm(step)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
tx = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3, weight_decay=0.1)
state = tx.init(params)
state = state._replace(
    hyperparams={**state.hyperparams, "learning_rate": jnp.asarray(2e-4)})

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
tx = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.adamw(learning_rate=3e-4, weight_decay=0.1),
)
# بديل تكيّفي: يقيّد نسبة التحديث إلى حجم المعامل نفسه
tx_agc = optax.chain(
    optax.adaptive_grad_clip(clipping=0.01, eps=1e-3),
    optax.sgd(learning_rate=0.1, momentum=0.9),
)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-15-ch15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def label(p):
    return jax.tree.map(lambda x: "decay" if x.ndim > 1 else "no_decay", p)

tx = optax.multi_transform(
    {"decay": optax.adamw(1e-3, weight_decay=0.1),
     "no_decay": optax.adamw(1e-3, weight_decay=0.0)},
    label)

state = tx.init(params)      # حالة لكل فرع، مدمجة في شجرة واحدة
