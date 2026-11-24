#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الملحق D: مرجع TensorFlow/Keras السريع
Professional TensorFlow — D Cheatsheet

Book slug : tensorflow-professional-ar
Chapter   : app-d-appendix-d-cheatsheet (lesson d)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/app-d-appendix-d-cheatsheet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الملحق D: مرجع TensorFlow/Keras السريع
Professional TensorFlow — D Cheatsheet

Book slug : tensorflow-professional-ar
Chapter   : app-d-appendix-d-cheatsheet (lesson d)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/appendix/appendix_d_cheatsheet.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.constant([[1., 2.], [3., 4.]])
v = tf.Variable(2.)
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(x * v)
grad = tape.gradient(loss, v)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/appendix/appendix_d_cheatsheet.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input((20,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/app-d-appendix-d-cheatsheet.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.constant([[1., 2.], [3., 4.]])
v = tf.Variable(2.)
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(x * v)
grad = tape.gradient(loss, v)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/app-d-appendix-d-cheatsheet.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input((20,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
