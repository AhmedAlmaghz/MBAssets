#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الأداء والتدريب الموزّع
Professional TensorFlow — Performance

Book slug : tensorflow-professional-ar
Chapter   : ch-18-ch18-performance (lesson 18)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-18-ch18-performance.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الأداء والتدريب الموزّع
Professional TensorFlow — Performance

Book slug : tensorflow-professional-ar
Chapter   : ch-18-ch18-performance (lesson 18)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch18_performance.md
# ────────────────────────────────────────────────────────────────────────
import time
import tensorflow as tf

@tf.function(jit_compile=False)
def step(x):
    return tf.linalg.matmul(x, x, transpose_b=True)

x = tf.random.normal((512, 128))
for _ in range(5):
    step(x)  # إحماء وتتبع
start = time.perf_counter()
for _ in range(50):
    step(x)
tf.experimental.async_wait()
elapsed = time.perf_counter() - start
print(f"{50 / elapsed:.1f} خطوة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch18_performance.md
# ────────────────────────────────────────────────────────────────────────
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = make_model()
    model.compile(optimizer="adam", loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch18_performance.md
# ────────────────────────────────────────────────────────────────────────
start = time.perf_counter()
for _ in range(20):
    step(batch)
tf.experimental.async_wait()
print(time.perf_counter() - start)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-18-ch18-performance.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import time
import tensorflow as tf

@tf.function(jit_compile=False)
def step(x):
    return tf.linalg.matmul(x, x, transpose_b=True)

x = tf.random.normal((512, 128))
for _ in range(5):
    step(x)  # إحماء وتتبع
start = time.perf_counter()
for _ in range(50):
    step(x)
tf.experimental.async_wait()
elapsed = time.perf_counter() - start
print(f"{50 / elapsed:.1f} خطوة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-18-ch18-performance.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = make_model()
    model.compile(optimizer="adam", loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-18-ch18-performance.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
start = time.perf_counter()
for _ in range(20):
    step(batch)
tf.experimental.async_wait()
print(time.perf_counter() - start)
