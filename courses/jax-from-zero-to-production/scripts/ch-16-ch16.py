#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس عشر: حلقة تدريب احترافية
JAX: From Zero to Production — Chapter 16 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-16-ch16 (lesson 16)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-16-ch16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس عشر: حلقة تدريب احترافية
JAX: From Zero to Production — Chapter 16 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-16-ch16 (lesson 16)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-16-ch16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس عشر: حلقة تدريب احترافية
JAX: From Zero to Production — Chapter 16 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-16-ch16 (lesson 16)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# متوسط متحرّك داخل الحلقة: لا .item() ولا قوائم
running = {"loss": jnp.zeros(()), "acc": jnp.zeros(())}

def accumulate(running, metrics, decay=0.98):
    return jax.tree.map(lambda r, m: decay * r + (1 - decay) * m, running, metrics)

# الطباعة عند حدود مضاعفات 100 فقط
if step % 100 == 0:
    print(step, {k: float(v) for k, v in running.items()})

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
import orbax.checkpoint as ocp

ckpt = ocp.PyTreeCheckpointer()
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
mngr = ocp.CheckpointManager("artifacts/run01", ckpt, options=options)

mngr.save(step, state)              # الحالة كلها: معاملات ومحسّن ومفتاح
print(mngr.latest_step())           # آخر خطوة محفوظة

restored = mngr.restore(mngr.latest_step(), items=state)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
def ema_update(ema, params, decay=0.999):
    """نسخة بطيئة من المعاملات، تُحدَّث بعد كل خطوة."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params)

# البديل الجاهز في Optax، إن أردت الحالة وإدارة الانحياز:
# ema_tx = optax.ema(0.999)
# ema_params, ema_state = ema_tx.update(params, ema_state)

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
state = mngr.restore(step, items=state)
state = state._replace(rng=jax.random.key(seed))   # لا: يهدم التسلسل
# الصحيح: لا تلمس rng إن كان محفوظاً في الحالة
state = mngr.restore(step, items=state)
loader = loader.skip(state.step * batch_size)      # نفس موضع البيانات

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
import functools, jax, jax.numpy as jnp, optax
import orbax.checkpoint as ocp
from typing import Any, NamedTuple

class TrainState(NamedTuple):
    params: Any          # شجرة أوزان النموذج
    opt_state: Any       # حالة المحسّن: زخم ولحظات وعدّاد
    step: jax.Array      # () int32
    rng: jax.Array       # مفتاح dropout
    ema: Any             # نسخة بطيئة بنفس شكل params

@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({"params": params}, batch["x"], train=True,
                             rngs={"dropout": jax.random.split(state.rng)[1]})
        logp = jax.nn.log_softmax(logits)
        loss = -jnp.mean(jnp.take_along_axis(logp, batch["y"][:, None], axis=1))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = jax.tree.map(lambda e, p: 0.999 * e + 0.001 * p, state.ema, params)
    metrics = {"loss": loss, "acc": jnp.mean(jnp.argmax(logits, -1) == batch["y"])}
    new_state = state._replace(params=params, opt_state=opt_state,
                               step=state.step + 1, rng=jax.random.split(state.rng)[0],
                               ema=ema)
    return new_state, metrics

mngr = ocp.CheckpointManager("artifacts/run01", ocp.PyTreeCheckpointer(),
                             options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
for epoch in range(num_epochs):
    for batch in loader:
        state, metrics = train_step(state, batch)
        running = jax.tree.map(lambda r, m: 0.98 * r + 0.02 * m, running, metrics)
        if int(state.step) % 100 == 0:
            print(int(state.step), {k: round(float(v), 4) for k, v in running.items()})
    mngr.save(int(state.step), state)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# متوسط متحرّك داخل الحلقة: لا .item() ولا قوائم
running = {"loss": jnp.zeros(()), "acc": jnp.zeros(())}

def accumulate(running, metrics, decay=0.98):
    return jax.tree.map(lambda r, m: decay * r + (1 - decay) * m, running, metrics)

# الطباعة عند حدود مضاعفات 100 فقط
if step % 100 == 0:
    print(step, {k: float(v) for k, v in running.items()})

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import orbax.checkpoint as ocp

ckpt = ocp.PyTreeCheckpointer()
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
mngr = ocp.CheckpointManager("artifacts/run01", ckpt, options=options)

mngr.save(step, state)              # الحالة كلها: معاملات ومحسّن ومفتاح
print(mngr.latest_step())           # آخر خطوة محفوظة

restored = mngr.restore(mngr.latest_step(), items=state)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def ema_update(ema, params, decay=0.999):
    """نسخة بطيئة من المعاملات، تُحدَّث بعد كل خطوة."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params)

# البديل الجاهز في Optax، إن أردت الحالة وإدارة الانحياز:
# ema_tx = optax.ema(0.999)
# ema_params, ema_state = ema_tx.update(params, ema_state)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
state = mngr.restore(step, items=state)
state = state._replace(rng=jax.random.key(seed))   # لا: يهدم التسلسل
# الصحيح: لا تلمس rng إن كان محفوظاً في الحالة
state = mngr.restore(step, items=state)
loader = loader.skip(state.step * batch_size)      # نفس موضع البيانات

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import functools, jax, jax.numpy as jnp, optax
import orbax.checkpoint as ocp
from typing import Any, NamedTuple

class TrainState(NamedTuple):
    params: Any          # شجرة أوزان النموذج
    opt_state: Any       # حالة المحسّن: زخم ولحظات وعدّاد
    step: jax.Array      # () int32
    rng: jax.Array       # مفتاح dropout
    ema: Any             # نسخة بطيئة بنفس شكل params

@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({"params": params}, batch["x"], train=True,
                             rngs={"dropout": jax.random.split(state.rng)[1]})
        logp = jax.nn.log_softmax(logits)
        loss = -jnp.mean(jnp.take_along_axis(logp, batch["y"][:, None], axis=1))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = jax.tree.map(lambda e, p: 0.999 * e + 0.001 * p, state.ema, params)
    metrics = {"loss": loss, "acc": jnp.mean(jnp.argmax(logits, -1) == batch["y"])}
    new_state = state._replace(params=params, opt_state=opt_state,
                               step=state.step + 1, rng=jax.random.split(state.rng)[0],
                               ema=ema)
    return new_state, metrics

mngr = ocp.CheckpointManager("artifacts/run01", ocp.PyTreeCheckpointer(),
                             options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
for epoch in range(num_epochs):
    for batch in loader:
        state, metrics = train_step(state, batch)
        running = jax.tree.map(lambda r, m: 0.98 * r + 0.02 * m, running, metrics)
        if int(state.step) % 100 == 0:
            print(int(state.step), {k: round(float(v), 4) for k, v in running.items()})
    mngr.save(int(state.step), state)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل السادس عشر: حلقة تدريب احترافية
JAX: From Zero to Production — Chapter 16 — Building Neural Networks

Book slug : jax-from-zero-to-production
Chapter   : ch-16-ch16 (lesson 16)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# متوسط متحرّك داخل الحلقة: لا .item() ولا قوائم
running = {"loss": jnp.zeros(()), "acc": jnp.zeros(())}

def accumulate(running, metrics, decay=0.98):
    return jax.tree.map(lambda r, m: decay * r + (1 - decay) * m, running, metrics)

# الطباعة عند حدود مضاعفات 100 فقط
if step % 100 == 0:
    print(step, {k: float(v) for k, v in running.items()})

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
import orbax.checkpoint as ocp

ckpt = ocp.PyTreeCheckpointer()
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
mngr = ocp.CheckpointManager("artifacts/run01", ckpt, options=options)

mngr.save(step, state)              # الحالة كلها: معاملات ومحسّن ومفتاح
print(mngr.latest_step())           # آخر خطوة محفوظة

restored = mngr.restore(mngr.latest_step(), items=state)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
def ema_update(ema, params, decay=0.999):
    """نسخة بطيئة من المعاملات، تُحدَّث بعد كل خطوة."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params)

# البديل الجاهز في Optax، إن أردت الحالة وإدارة الانحياز:
# ema_tx = optax.ema(0.999)
# ema_params, ema_state = ema_tx.update(params, ema_state)

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
state = mngr.restore(step, items=state)
state = state._replace(rng=jax.random.key(seed))   # لا: يهدم التسلسل
# الصحيح: لا تلمس rng إن كان محفوظاً في الحالة
state = mngr.restore(step, items=state)
loader = loader.skip(state.step * batch_size)      # نفس موضع البيانات

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch16.md
# ────────────────────────────────────────────────────────────────────────
import functools, jax, jax.numpy as jnp, optax
import orbax.checkpoint as ocp
from typing import Any, NamedTuple

class TrainState(NamedTuple):
    params: Any          # شجرة أوزان النموذج
    opt_state: Any       # حالة المحسّن: زخم ولحظات وعدّاد
    step: jax.Array      # () int32
    rng: jax.Array       # مفتاح dropout
    ema: Any             # نسخة بطيئة بنفس شكل params

@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({"params": params}, batch["x"], train=True,
                             rngs={"dropout": jax.random.split(state.rng)[1]})
        logp = jax.nn.log_softmax(logits)
        loss = -jnp.mean(jnp.take_along_axis(logp, batch["y"][:, None], axis=1))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = jax.tree.map(lambda e, p: 0.999 * e + 0.001 * p, state.ema, params)
    metrics = {"loss": loss, "acc": jnp.mean(jnp.argmax(logits, -1) == batch["y"])}
    new_state = state._replace(params=params, opt_state=opt_state,
                               step=state.step + 1, rng=jax.random.split(state.rng)[0],
                               ema=ema)
    return new_state, metrics

mngr = ocp.CheckpointManager("artifacts/run01", ocp.PyTreeCheckpointer(),
                             options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
for epoch in range(num_epochs):
    for batch in loader:
        state, metrics = train_step(state, batch)
        running = jax.tree.map(lambda r, m: 0.98 * r + 0.02 * m, running, metrics)
        if int(state.step) % 100 == 0:
            print(int(state.step), {k: round(float(v), 4) for k, v in running.items()})
    mngr.save(int(state.step), state)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

# متوسط متحرّك داخل الحلقة: لا .item() ولا قوائم
running = {"loss": jnp.zeros(()), "acc": jnp.zeros(())}

def accumulate(running, metrics, decay=0.98):
    return jax.tree.map(lambda r, m: decay * r + (1 - decay) * m, running, metrics)

# الطباعة عند حدود مضاعفات 100 فقط
if step % 100 == 0:
    print(step, {k: float(v) for k, v in running.items()})

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import orbax.checkpoint as ocp

ckpt = ocp.PyTreeCheckpointer()
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
mngr = ocp.CheckpointManager("artifacts/run01", ckpt, options=options)

mngr.save(step, state)              # الحالة كلها: معاملات ومحسّن ومفتاح
print(mngr.latest_step())           # آخر خطوة محفوظة

restored = mngr.restore(mngr.latest_step(), items=state)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def ema_update(ema, params, decay=0.999):
    """نسخة بطيئة من المعاملات، تُحدَّث بعد كل خطوة."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params)

# البديل الجاهز في Optax، إن أردت الحالة وإدارة الانحياز:
# ema_tx = optax.ema(0.999)
# ema_params, ema_state = ema_tx.update(params, ema_state)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
state = mngr.restore(step, items=state)
state = state._replace(rng=jax.random.key(seed))   # لا: يهدم التسلسل
# الصحيح: لا تلمس rng إن كان محفوظاً في الحالة
state = mngr.restore(step, items=state)
loader = loader.skip(state.step * batch_size)      # نفس موضع البيانات

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-16-ch16.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import functools, jax, jax.numpy as jnp, optax
import orbax.checkpoint as ocp
from typing import Any, NamedTuple

class TrainState(NamedTuple):
    params: Any          # شجرة أوزان النموذج
    opt_state: Any       # حالة المحسّن: زخم ولحظات وعدّاد
    step: jax.Array      # () int32
    rng: jax.Array       # مفتاح dropout
    ema: Any             # نسخة بطيئة بنفس شكل params

@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({"params": params}, batch["x"], train=True,
                             rngs={"dropout": jax.random.split(state.rng)[1]})
        logp = jax.nn.log_softmax(logits)
        loss = -jnp.mean(jnp.take_along_axis(logp, batch["y"][:, None], axis=1))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = jax.tree.map(lambda e, p: 0.999 * e + 0.001 * p, state.ema, params)
    metrics = {"loss": loss, "acc": jnp.mean(jnp.argmax(logits, -1) == batch["y"])}
    new_state = state._replace(params=params, opt_state=opt_state,
                               step=state.step + 1, rng=jax.random.split(state.rng)[0],
                               ema=ema)
    return new_state, metrics

mngr = ocp.CheckpointManager("artifacts/run01", ocp.PyTreeCheckpointer(),
                             options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
for epoch in range(num_epochs):
    for batch in loader:
        state, metrics = train_step(state, batch)
        running = jax.tree.map(lambda r, m: 0.98 * r + 0.02 * m, running, metrics)
        if int(state.step) % 100 == 0:
            print(int(state.step), {k: round(float(v), 4) for k, v in running.items()})
    mngr.save(int(state.step), state)
