#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع والعشرون: التصدير والنشر
JAX: From Zero to Production — jax.export

Book slug : jax-from-zero-to-production
Chapter   : ch-29-ch29 (lesson 29)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-29-ch29.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع والعشرون: التصدير والنشر
JAX: From Zero to Production — jax.export

Book slug : jax-from-zero-to-production
Chapter   : ch-29-ch29 (lesson 29)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/jax-from-zero-to-production/code/ch-29-ch29.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع والعشرون: التصدير والنشر
JAX: From Zero to Production — jax.export

Book slug : jax-from-zero-to-production
Chapter   : ch-29-ch29 (lesson 29)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

params = {"w": jnp.ones((4, 3), dtype=jnp.float32),
          "b": jnp.zeros((3,), dtype=jnp.float32)}

def model(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

# التصدير يلتقط الرسم لأشكال الوسائط المثال: (params, f32[2,4])
exported = export.export(model)(params, jnp.ones((2, 4), dtype=jnp.float32))
print(exported.in_avals)        # أشكال المدخلات
print(exported.out_avals)       # شكل المخرج: f32[2,3]

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

# في جهة التدريب: التسلسل إلى ملف
serialized = export.serialize(exported)
with open("model.jaxe", "wb") as f:
    f.write(serialized)

# في جهة الخدمة: التحميل والتنفيذ بلا شيفرة model.py
with open("model.jaxe", "rb") as f:
    loaded = export.deserialize(f.read())

params = load_params()                       # تُحمّل من التخزين
x = jnp.ones((2, 4), dtype=jnp.float32)
out = loaded.call(params, x)                  # تنفيذ النموذج المُصدَّر
print(out.shape)                             # (2, 3)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, jax.numpy as jnp
from jax import export

app = FastAPI()
# يُحمّل مرّة واحدة عند إقلاع العملية، لا لكل طلب
EXPORTED = export.deserialize(open("model.jaxe", "rb").read())
PARAMS = load_params()

class Req(BaseModel):
    x: list[list[float]]

@app.post("/predict")
def predict(req: Req):
    arr = np.asarray(req.x, dtype=np.float32)        # f32[N,4]
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))    # تنفيذ مُصدَّر
    return {"y": np.asarray(out).tolist()}

# بديل أسرع زمن استجابة: gRPC بنفس نمط «حمّل مرّة، نفّذ كثيراً»

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
# مسار موصى به: jax.export ثم محوّل ONNX، لا jax2tf
from jax import export

exported = export.export(model)(params, jnp.ones((1, 4), dtype=jnp.float32))
serialized = export.serialize(exported)

# هنا يأتي محوّل ONNX الطرفي (خارج JAX)، مثل:
# onnx_model = my_onnx_converter.from_jax_export(serialized)
# onnx.save(onnx_model, "model.onnx")

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
import time
from prometheus_client import Counter, Histogram

REQS = Counter("predict_total", "عدد طلبات التنبؤ")
LAT = Histogram("predict_latency_seconds", "زمن الاستجابة")

@LAT.time()
def serve(arr):
    REQS.inc()
    t0 = time.perf_counter()
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))
    out.block_until_ready()              # لا تحسب الزمن قبل اكتمال XLA
    return np.asarray(out)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

params = {"w": jnp.ones((4, 3), dtype=jnp.float32),
          "b": jnp.zeros((3,), dtype=jnp.float32)}

def model(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

# التصدير يلتقط الرسم لأشكال الوسائط المثال: (params, f32[2,4])
exported = export.export(model)(params, jnp.ones((2, 4), dtype=jnp.float32))
print(exported.in_avals)        # أشكال المدخلات
print(exported.out_avals)       # شكل المخرج: f32[2,3]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

# في جهة التدريب: التسلسل إلى ملف
serialized = export.serialize(exported)
with open("model.jaxe", "wb") as f:
    f.write(serialized)

# في جهة الخدمة: التحميل والتنفيذ بلا شيفرة model.py
with open("model.jaxe", "rb") as f:
    loaded = export.deserialize(f.read())

params = load_params()                       # تُحمّل من التخزين
x = jnp.ones((2, 4), dtype=jnp.float32)
out = loaded.call(params, x)                  # تنفيذ النموذج المُصدَّر
print(out.shape)                             # (2, 3)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, jax.numpy as jnp
from jax import export

app = FastAPI()
# يُحمّل مرّة واحدة عند إقلاع العملية، لا لكل طلب
EXPORTED = export.deserialize(open("model.jaxe", "rb").read())
PARAMS = load_params()

class Req(BaseModel):
    x: list[list[float]]

@app.post("/predict")
def predict(req: Req):
    arr = np.asarray(req.x, dtype=np.float32)        # f32[N,4]
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))    # تنفيذ مُصدَّر
    return {"y": np.asarray(out).tolist()}

# بديل أسرع زمن استجابة: gRPC بنفس نمط «حمّل مرّة، نفّذ كثيراً»

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# مسار موصى به: jax.export ثم محوّل ONNX، لا jax2tf
from jax import export

exported = export.export(model)(params, jnp.ones((1, 4), dtype=jnp.float32))
serialized = export.serialize(exported)

# هنا يأتي محوّل ONNX الطرفي (خارج JAX)، مثل:
# onnx_model = my_onnx_converter.from_jax_export(serialized)
# onnx.save(onnx_model, "model.onnx")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
from prometheus_client import Counter, Histogram

REQS = Counter("predict_total", "عدد طلبات التنبؤ")
LAT = Histogram("predict_latency_seconds", "زمن الاستجابة")

@LAT.time()
def serve(arr):
    REQS.inc()
    t0 = time.perf_counter()
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))
    out.block_until_ready()              # لا تحسب الزمن قبل اكتمال XLA
    return np.asarray(out)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX من الصفر إلى الإنتاج — الفصل التاسع والعشرون: التصدير والنشر
JAX: From Zero to Production — jax.export

Book slug : jax-from-zero-to-production
Chapter   : ch-29-ch29 (lesson 29)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

params = {"w": jnp.ones((4, 3), dtype=jnp.float32),
          "b": jnp.zeros((3,), dtype=jnp.float32)}

def model(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

# التصدير يلتقط الرسم لأشكال الوسائط المثال: (params, f32[2,4])
exported = export.export(model)(params, jnp.ones((2, 4), dtype=jnp.float32))
print(exported.in_avals)        # أشكال المدخلات
print(exported.out_avals)       # شكل المخرج: f32[2,3]

# ────────────────────────────────────────────────────────────────────────
# [002] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

# في جهة التدريب: التسلسل إلى ملف
serialized = export.serialize(exported)
with open("model.jaxe", "wb") as f:
    f.write(serialized)

# في جهة الخدمة: التحميل والتنفيذ بلا شيفرة model.py
with open("model.jaxe", "rb") as f:
    loaded = export.deserialize(f.read())

params = load_params()                       # تُحمّل من التخزين
x = jnp.ones((2, 4), dtype=jnp.float32)
out = loaded.call(params, x)                  # تنفيذ النموذج المُصدَّر
print(out.shape)                             # (2, 3)

# ────────────────────────────────────────────────────────────────────────
# [003] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, jax.numpy as jnp
from jax import export

app = FastAPI()
# يُحمّل مرّة واحدة عند إقلاع العملية، لا لكل طلب
EXPORTED = export.deserialize(open("model.jaxe", "rb").read())
PARAMS = load_params()

class Req(BaseModel):
    x: list[list[float]]

@app.post("/predict")
def predict(req: Req):
    arr = np.asarray(req.x, dtype=np.float32)        # f32[N,4]
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))    # تنفيذ مُصدَّر
    return {"y": np.asarray(out).tolist()}

# بديل أسرع زمن استجابة: gRPC بنفس نمط «حمّل مرّة، نفّذ كثيراً»

# ────────────────────────────────────────────────────────────────────────
# [004] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
# مسار موصى به: jax.export ثم محوّل ONNX، لا jax2tf
from jax import export

exported = export.export(model)(params, jnp.ones((1, 4), dtype=jnp.float32))
serialized = export.serialize(exported)

# هنا يأتي محوّل ONNX الطرفي (خارج JAX)، مثل:
# onnx_model = my_onnx_converter.from_jax_export(serialized)
# onnx.save(onnx_model, "model.onnx")

# ────────────────────────────────────────────────────────────────────────
# [005] book/build/md/ch29.md
# ────────────────────────────────────────────────────────────────────────
import time
from prometheus_client import Counter, Histogram

REQS = Counter("predict_total", "عدد طلبات التنبؤ")
LAT = Histogram("predict_latency_seconds", "زمن الاستجابة")

@LAT.time()
def serve(arr):
    REQS.inc()
    t0 = time.perf_counter()
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))
    out.block_until_ready()              # لا تحسب الزمن قبل اكتمال XLA
    return np.asarray(out)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

params = {"w": jnp.ones((4, 3), dtype=jnp.float32),
          "b": jnp.zeros((3,), dtype=jnp.float32)}

def model(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

# التصدير يلتقط الرسم لأشكال الوسائط المثال: (params, f32[2,4])
exported = export.export(model)(params, jnp.ones((2, 4), dtype=jnp.float32))
print(exported.in_avals)        # أشكال المدخلات
print(exported.out_avals)       # شكل المخرج: f32[2,3]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
from jax import export

# في جهة التدريب: التسلسل إلى ملف
serialized = export.serialize(exported)
with open("model.jaxe", "wb") as f:
    f.write(serialized)

# في جهة الخدمة: التحميل والتنفيذ بلا شيفرة model.py
with open("model.jaxe", "rb") as f:
    loaded = export.deserialize(f.read())

params = load_params()                       # تُحمّل من التخزين
x = jnp.ones((2, 4), dtype=jnp.float32)
out = loaded.call(params, x)                  # تنفيذ النموذج المُصدَّر
print(out.shape)                             # (2, 3)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, jax.numpy as jnp
from jax import export

app = FastAPI()
# يُحمّل مرّة واحدة عند إقلاع العملية، لا لكل طلب
EXPORTED = export.deserialize(open("model.jaxe", "rb").read())
PARAMS = load_params()

class Req(BaseModel):
    x: list[list[float]]

@app.post("/predict")
def predict(req: Req):
    arr = np.asarray(req.x, dtype=np.float32)        # f32[N,4]
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))    # تنفيذ مُصدَّر
    return {"y": np.asarray(out).tolist()}

# بديل أسرع زمن استجابة: gRPC بنفس نمط «حمّل مرّة، نفّذ كثيراً»

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# مسار موصى به: jax.export ثم محوّل ONNX، لا jax2tf
from jax import export

exported = export.export(model)(params, jnp.ones((1, 4), dtype=jnp.float32))
serialized = export.serialize(exported)

# هنا يأتي محوّل ONNX الطرفي (خارج JAX)، مثل:
# onnx_model = my_onnx_converter.from_jax_export(serialized)
# onnx.save(onnx_model, "model.onnx")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/jax-from-zero-to-production/notebooks/ch-29-ch29.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
from prometheus_client import Counter, Histogram

REQS = Counter("predict_total", "عدد طلبات التنبؤ")
LAT = Histogram("predict_latency_seconds", "زمن الاستجابة")

@LAT.time()
def serve(arr):
    REQS.inc()
    t0 = time.perf_counter()
    out = EXPORTED.call(PARAMS, jnp.asarray(arr))
    out.block_until_ready()              # لا تحسب الزمن قبل اكتمال XLA
    return np.asarray(out)
