#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والثلاثون: دراسات حالة من الحياة العملية
JAX: From Zero to Production — Chapter 31 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-31-ch31 (lesson 31)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-31-ch31.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والثلاثون: دراسات حالة من الحياة العملية
JAX: From Zero to Production — Chapter 31 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-31-ch31 (lesson 31)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-31-ch31.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والثلاثون: دراسات حالة من الحياة العملية
JAX: From Zero to Production — Chapter 31 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-31-ch31 (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Triage(nn.Module):                 # مصنّف أشعّة صدرية ثنائي
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(2)(x)            # [سليم، خطر]

model = Triage()
key = jax.random.key(0)
x = jnp.ones((8, 128, 128, 1))           # دفعة صور
params = model.init(key, x)["params"]
# تطبيق على كامل الدفعة دفعةً واحدة بفضل vmap الداخلي لـ apply
logits = model.apply({"params": params}, x)
probs = jax.nn.softmax(logits, axis=-1)
print(float(probs[0, 1]))                # احتمال الخطورة للصورة الأولى

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def pnl_path(key, w, mu, cov, horizon=20):
    L = jnp.linalg.cholesky(cov)         # تفكيك المتباينة
    z = jax.random.normal(key, (horizon, w.shape[0]))
    rets = mu + z @ L.T                  # عوائد يومية مترابطة
    prices = jnp.cumprod(1.0 + rets, axis=0)
    return -jnp.sum(w * (prices[-1] - 1.0))

w = jnp.array([0.6, 0.4])                # أوزان المحفظة
mu = jnp.array([0.0005, 0.0003])
cov = jnp.array([[1e-4, 3e-5], [3e-5, 8e-5]])
keys = jax.random.split(jax.random.key(7), 1_000_000)
losses = jax.vmap(pnl_path, in_axes=(0, None, None, None))(
    keys, w, mu, cov)                    # مليون مسار متوازٍ
var95 = float(jnp.percentile(losses, 95))
print(var95)                             # قيمة الخطر عند 95٪

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

mesh = Mesh(jax.devices(), ("d",))
spec = NamedSharding(mesh, P("d"))

def score_one(img):                     # نموذج كشف العيب على صورة واحدة
    h = jax.nn.relu(img.reshape((64,)) @ jnp.ones((64, 16)))
    return jax.nn.sigmoid(h @ jnp.ones((16, 1)))[0]

@partial(shard_map, mesh=mesh, in_specs=(P("d"),), out_specs=P("d"))
def detect_sharded(x):
    return jax.vmap(score_one)(x)        # كل جهاز يقيم حصّته

imgs = jax.random.normal(jax.random.key(0), (256, 64))
sharded = jax.device_put(imgs, spec)     # توزيع على الأجهزة
scores = detect_sharded(sharded)
print(scores.shape)                      # (256,) بعد جمع الأشرطة

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def cosine(a, b):                       # تشابه جيب التمام
    return jnp.dot(a, b) / (
        jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)

user = jax.random.normal(jax.random.key(1), (64,))
items = jax.random.normal(jax.random.key(2), (10_000, 64))

scores = jax.vmap(lambda c: cosine(user, c))(items)
top50 = jnp.argsort(-scores)[:50]        # أعلى خمسين مرشّحاً
print(int(top50[0]), float(scores[top50[0]]))

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def forecast(params, window):           # تنبّؤ خطوة واحدة
    h = jax.nn.tanh(window @ params["in"])
    return h @ params["out"]

def rollout(params, series):
    def step(carry, x):                 # حلقة النافذة المنزلقة
        out = forecast(params, carry)
        carry = jnp.roll(carry, -1).at[-1].set(x)
        return carry, out
    _, outs = jax.lax.scan(step, series[:24], series[24:])
    return outs

series = jax.random.normal(jax.random.key(3), (200,))
params = {"in": jnp.ones((24, 16)), "out": jnp.ones((16,))}
preds = rollout(params, series)
print(preds.shape)                      # (176,) تنبّؤات

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import lax

def dynamics(s, u):                     # عربة بسيطة: [موضع، سرعة]
    return s + jnp.array([s[1], u]) * 0.1

def rollout(state, acts, target):
    def f(s, u):
        s2 = dynamics(s, u)
        return s2, jnp.sum((s2 - target) ** 2)
    _, cs = lax.scan(f, state, acts)
    return jnp.sum(cs)

state0, target = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])
keys = jax.random.split(jax.random.key(0), 4096)
acts = jax.random.normal(keys, (4096, 10))      # 4096 مرشّحاً
costs = jax.vmap(lambda a: rollout(state0, a, target))(acts)
best = acts[jnp.argmin(costs)]                  # أفضل تسلسل تحكّم
print(best.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Triage(nn.Module):                 # مصنّف أشعّة صدرية ثنائي
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(2)(x)            # [سليم، خطر]

model = Triage()
key = jax.random.key(0)
x = jnp.ones((8, 128, 128, 1))           # دفعة صور
params = model.init(key, x)["params"]
# تطبيق على كامل الدفعة دفعةً واحدة بفضل vmap الداخلي لـ apply
logits = model.apply({"params": params}, x)
probs = jax.nn.softmax(logits, axis=-1)
print(float(probs[0, 1]))                # احتمال الخطورة للصورة الأولى

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def pnl_path(key, w, mu, cov, horizon=20):
    L = jnp.linalg.cholesky(cov)         # تفكيك المتباينة
    z = jax.random.normal(key, (horizon, w.shape[0]))
    rets = mu + z @ L.T                  # عوائد يومية مترابطة
    prices = jnp.cumprod(1.0 + rets, axis=0)
    return -jnp.sum(w * (prices[-1] - 1.0))

w = jnp.array([0.6, 0.4])                # أوزان المحفظة
mu = jnp.array([0.0005, 0.0003])
cov = jnp.array([[1e-4, 3e-5], [3e-5, 8e-5]])
keys = jax.random.split(jax.random.key(7), 1_000_000)
losses = jax.vmap(pnl_path, in_axes=(0, None, None, None))(
    keys, w, mu, cov)                    # مليون مسار متوازٍ
var95 = float(jnp.percentile(losses, 95))
print(var95)                             # قيمة الخطر عند 95٪

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

mesh = Mesh(jax.devices(), ("d",))
spec = NamedSharding(mesh, P("d"))

def score_one(img):                     # نموذج كشف العيب على صورة واحدة
    h = jax.nn.relu(img.reshape((64,)) @ jnp.ones((64, 16)))
    return jax.nn.sigmoid(h @ jnp.ones((16, 1)))[0]

@partial(shard_map, mesh=mesh, in_specs=(P("d"),), out_specs=P("d"))
def detect_sharded(x):
    return jax.vmap(score_one)(x)        # كل جهاز يقيم حصّته

imgs = jax.random.normal(jax.random.key(0), (256, 64))
sharded = jax.device_put(imgs, spec)     # توزيع على الأجهزة
scores = detect_sharded(sharded)
print(scores.shape)                      # (256,) بعد جمع الأشرطة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def cosine(a, b):                       # تشابه جيب التمام
    return jnp.dot(a, b) / (
        jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)

user = jax.random.normal(jax.random.key(1), (64,))
items = jax.random.normal(jax.random.key(2), (10_000, 64))

scores = jax.vmap(lambda c: cosine(user, c))(items)
top50 = jnp.argsort(-scores)[:50]        # أعلى خمسين مرشّحاً
print(int(top50[0]), float(scores[top50[0]]))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def forecast(params, window):           # تنبّؤ خطوة واحدة
    h = jax.nn.tanh(window @ params["in"])
    return h @ params["out"]

def rollout(params, series):
    def step(carry, x):                 # حلقة النافذة المنزلقة
        out = forecast(params, carry)
        carry = jnp.roll(carry, -1).at[-1].set(x)
        return carry, out
    _, outs = jax.lax.scan(step, series[:24], series[24:])
    return outs

series = jax.random.normal(jax.random.key(3), (200,))
params = {"in": jnp.ones((24, 16)), "out": jnp.ones((16,))}
preds = rollout(params, series)
print(preds.shape)                      # (176,) تنبّؤات

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import lax

def dynamics(s, u):                     # عربة بسيطة: [موضع، سرعة]
    return s + jnp.array([s[1], u]) * 0.1

def rollout(state, acts, target):
    def f(s, u):
        s2 = dynamics(s, u)
        return s2, jnp.sum((s2 - target) ** 2)
    _, cs = lax.scan(f, state, acts)
    return jnp.sum(cs)

state0, target = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])
keys = jax.random.split(jax.random.key(0), 4096)
acts = jax.random.normal(keys, (4096, 10))      # 4096 مرشّحاً
costs = jax.vmap(lambda a: rollout(state0, a, target))(acts)
best = acts[jnp.argmin(costs)]                  # أفضل تسلسل تحكّم
print(best.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل الحادي والثلاثون: دراسات حالة من الحياة العملية
JAX: From Zero to Production — Chapter 31 — From Lab to Production

Book slug : jax-from-zero-to-production
Chapter   : ch-31-ch31 (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Triage(nn.Module):                 # مصنّف أشعّة صدرية ثنائي
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(2)(x)            # [سليم، خطر]

model = Triage()
key = jax.random.key(0)
x = jnp.ones((8, 128, 128, 1))           # دفعة صور
params = model.init(key, x)["params"]
# تطبيق على كامل الدفعة دفعةً واحدة بفضل vmap الداخلي لـ apply
logits = model.apply({"params": params}, x)
probs = jax.nn.softmax(logits, axis=-1)
print(float(probs[0, 1]))                # احتمال الخطورة للصورة الأولى

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def pnl_path(key, w, mu, cov, horizon=20):
    L = jnp.linalg.cholesky(cov)         # تفكيك المتباينة
    z = jax.random.normal(key, (horizon, w.shape[0]))
    rets = mu + z @ L.T                  # عوائد يومية مترابطة
    prices = jnp.cumprod(1.0 + rets, axis=0)
    return -jnp.sum(w * (prices[-1] - 1.0))

w = jnp.array([0.6, 0.4])                # أوزان المحفظة
mu = jnp.array([0.0005, 0.0003])
cov = jnp.array([[1e-4, 3e-5], [3e-5, 8e-5]])
keys = jax.random.split(jax.random.key(7), 1_000_000)
losses = jax.vmap(pnl_path, in_axes=(0, None, None, None))(
    keys, w, mu, cov)                    # مليون مسار متوازٍ
var95 = float(jnp.percentile(losses, 95))
print(var95)                             # قيمة الخطر عند 95٪

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

mesh = Mesh(jax.devices(), ("d",))
spec = NamedSharding(mesh, P("d"))

def score_one(img):                     # نموذج كشف العيب على صورة واحدة
    h = jax.nn.relu(img.reshape((64,)) @ jnp.ones((64, 16)))
    return jax.nn.sigmoid(h @ jnp.ones((16, 1)))[0]

@partial(shard_map, mesh=mesh, in_specs=(P("d"),), out_specs=P("d"))
def detect_sharded(x):
    return jax.vmap(score_one)(x)        # كل جهاز يقيم حصّته

imgs = jax.random.normal(jax.random.key(0), (256, 64))
sharded = jax.device_put(imgs, spec)     # توزيع على الأجهزة
scores = detect_sharded(sharded)
print(scores.shape)                      # (256,) بعد جمع الأشرطة

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def cosine(a, b):                       # تشابه جيب التمام
    return jnp.dot(a, b) / (
        jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)

user = jax.random.normal(jax.random.key(1), (64,))
items = jax.random.normal(jax.random.key(2), (10_000, 64))

scores = jax.vmap(lambda c: cosine(user, c))(items)
top50 = jnp.argsort(-scores)[:50]        # أعلى خمسين مرشّحاً
print(int(top50[0]), float(scores[top50[0]]))

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def forecast(params, window):           # تنبّؤ خطوة واحدة
    h = jax.nn.tanh(window @ params["in"])
    return h @ params["out"]

def rollout(params, series):
    def step(carry, x):                 # حلقة النافذة المنزلقة
        out = forecast(params, carry)
        carry = jnp.roll(carry, -1).at[-1].set(x)
        return carry, out
    _, outs = jax.lax.scan(step, series[:24], series[24:])
    return outs

series = jax.random.normal(jax.random.key(3), (200,))
params = {"in": jnp.ones((24, 16)), "out": jnp.ones((16,))}
preds = rollout(params, series)
print(preds.shape)                      # (176,) تنبّؤات

# ────────────────────────────────────────────────────────────────────────
# [006] book/build/md/ch31.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import lax

def dynamics(s, u):                     # عربة بسيطة: [موضع، سرعة]
    return s + jnp.array([s[1], u]) * 0.1

def rollout(state, acts, target):
    def f(s, u):
        s2 = dynamics(s, u)
        return s2, jnp.sum((s2 - target) ** 2)
    _, cs = lax.scan(f, state, acts)
    return jnp.sum(cs)

state0, target = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])
keys = jax.random.split(jax.random.key(0), 4096)
acts = jax.random.normal(keys, (4096, 10))      # 4096 مرشّحاً
costs = jax.vmap(lambda a: rollout(state0, a, target))(acts)
best = acts[jnp.argmin(costs)]                  # أفضل تسلسل تحكّم
print(best.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from flax import linen as nn

class Triage(nn.Module):                 # مصنّف أشعّة صدرية ثنائي
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3))(x))
        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(2)(x)            # [سليم، خطر]

model = Triage()
key = jax.random.key(0)
x = jnp.ones((8, 128, 128, 1))           # دفعة صور
params = model.init(key, x)["params"]
# تطبيق على كامل الدفعة دفعةً واحدة بفضل vmap الداخلي لـ apply
logits = model.apply({"params": params}, x)
probs = jax.nn.softmax(logits, axis=-1)
print(float(probs[0, 1]))                # احتمال الخطورة للصورة الأولى

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def pnl_path(key, w, mu, cov, horizon=20):
    L = jnp.linalg.cholesky(cov)         # تفكيك المتباينة
    z = jax.random.normal(key, (horizon, w.shape[0]))
    rets = mu + z @ L.T                  # عوائد يومية مترابطة
    prices = jnp.cumprod(1.0 + rets, axis=0)
    return -jnp.sum(w * (prices[-1] - 1.0))

w = jnp.array([0.6, 0.4])                # أوزان المحفظة
mu = jnp.array([0.0005, 0.0003])
cov = jnp.array([[1e-4, 3e-5], [3e-5, 8e-5]])
keys = jax.random.split(jax.random.key(7), 1_000_000)
losses = jax.vmap(pnl_path, in_axes=(0, None, None, None))(
    keys, w, mu, cov)                    # مليون مسار متوازٍ
var95 = float(jnp.percentile(losses, 95))
print(var95)                             # قيمة الخطر عند 95٪

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

mesh = Mesh(jax.devices(), ("d",))
spec = NamedSharding(mesh, P("d"))

def score_one(img):                     # نموذج كشف العيب على صورة واحدة
    h = jax.nn.relu(img.reshape((64,)) @ jnp.ones((64, 16)))
    return jax.nn.sigmoid(h @ jnp.ones((16, 1)))[0]

@partial(shard_map, mesh=mesh, in_specs=(P("d"),), out_specs=P("d"))
def detect_sharded(x):
    return jax.vmap(score_one)(x)        # كل جهاز يقيم حصّته

imgs = jax.random.normal(jax.random.key(0), (256, 64))
sharded = jax.device_put(imgs, spec)     # توزيع على الأجهزة
scores = detect_sharded(sharded)
print(scores.shape)                      # (256,) بعد جمع الأشرطة

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

def cosine(a, b):                       # تشابه جيب التمام
    return jnp.dot(a, b) / (
        jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)

user = jax.random.normal(jax.random.key(1), (64,))
items = jax.random.normal(jax.random.key(2), (10_000, 64))

scores = jax.vmap(lambda c: cosine(user, c))(items)
top50 = jnp.argsort(-scores)[:50]        # أعلى خمسين مرشّحاً
print(int(top50[0]), float(scores[top50[0]]))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp

@jax.jit
def forecast(params, window):           # تنبّؤ خطوة واحدة
    h = jax.nn.tanh(window @ params["in"])
    return h @ params["out"]

def rollout(params, series):
    def step(carry, x):                 # حلقة النافذة المنزلقة
        out = forecast(params, carry)
        carry = jnp.roll(carry, -1).at[-1].set(x)
        return carry, out
    _, outs = jax.lax.scan(step, series[:24], series[24:])
    return outs

series = jax.random.normal(jax.random.key(3), (200,))
params = {"in": jnp.ones((24, 16)), "out": jnp.ones((16,))}
preds = rollout(params, series)
print(preds.shape)                      # (176,) تنبّؤات

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/jax-from-zero-to-production/notebooks/ch-31-ch31.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import lax

def dynamics(s, u):                     # عربة بسيطة: [موضع، سرعة]
    return s + jnp.array([s[1], u]) * 0.1

def rollout(state, acts, target):
    def f(s, u):
        s2 = dynamics(s, u)
        return s2, jnp.sum((s2 - target) ** 2)
    _, cs = lax.scan(f, state, acts)
    return jnp.sum(cs)

state0, target = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])
keys = jax.random.split(jax.random.key(0), 4096)
acts = jax.random.normal(keys, (4096, 10))      # 4096 مرشّحاً
costs = jax.vmap(lambda a: rollout(state0, a, target))(acts)
best = acts[jnp.argmin(costs)]                  # أفضل تسلسل تحكّم
print(best.shape)
