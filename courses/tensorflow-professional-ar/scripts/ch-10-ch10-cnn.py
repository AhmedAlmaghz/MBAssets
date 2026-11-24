#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الرؤية الحاسوبية: CNN من الالتفاف إلى الإنتاج
Professional TensorFlow — Cnn

Book slug : tensorflow-professional-ar
Chapter   : ch-10-ch10-cnn (lesson 10)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-10-ch10-cnn.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الرؤية الحاسوبية: CNN من الالتفاف إلى الإنتاج
Professional TensorFlow — Cnn

Book slug : tensorflow-professional-ar
Chapter   : ch-10-ch10-cnn (lesson 10)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch10_cnn.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input((64, 64, 3)),
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(), layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation="softmax"),
])
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch10_cnn.md
# ────────────────────────────────────────────────────────────────────────
augment = keras.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
])
augmented = augment(images, training=True)

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch10_cnn.md
# ────────────────────────────────────────────────────────────────────────
images = tf.cast(images, tf.float32) / 255.0
print(tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-10-ch10-cnn.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input((64, 64, 3)),
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(), layers.Activation("relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation="softmax"),
])
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-10-ch10-cnn.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
augment = keras.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
])
augmented = augment(images, training=True)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-10-ch10-cnn.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
images = tf.cast(images, tf.float32) / 255.0
print(tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy())
