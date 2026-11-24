#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الشبكات العصبية من الصفر
Professional TensorFlow — Neural Nets

Book slug : tensorflow-professional-ar
Chapter   : ch-06-ch06-neural-nets (lesson 6)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-06-ch06-neural-nets.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الشبكات العصبية من الصفر
Professional TensorFlow — Neural Nets

Book slug : tensorflow-professional-ar
Chapter   : ch-06-ch06-neural-nets (lesson 6)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch06_neural_nets.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.constant([[0.0], [1.0], [2.0], [3.0]])
y = tf.constant([[1.0], [3.0], [5.0], [7.0]])
w = tf.Variable([[0.0]])
b = tf.Variable([0.0])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

for _ in range(100):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(x @ w + b - y))
    dw, db = tape.gradient(loss, [w, b])
    optimizer.apply_gradients([(dw, w), (db, b)])
print(w.numpy(), b.numpy())
# المتوقع: w تقترب من 2 وb تقترب من 1.

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch06_neural_nets.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = keras.Sequential([
    layers.Input((28, 28)), layers.Flatten(),
    layers.Dense(128, activation="relu"), layers.Dense(10),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
print(model.evaluate(x_test, y_test, verbose=0))

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch06_neural_nets.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers
classifier = keras.Sequential([layers.Input((28, 28, 1)), layers.Flatten(),
    layers.Dense(64, activation="relu"), layers.Dense(10)])
classifier.compile(optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch06_neural_nets.md
# ────────────────────────────────────────────────────────────────────────
import keras
keras.utils.set_random_seed(42)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-06-ch06-neural-nets.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.constant([[0.0], [1.0], [2.0], [3.0]])
y = tf.constant([[1.0], [3.0], [5.0], [7.0]])
w = tf.Variable([[0.0]])
b = tf.Variable([0.0])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

for _ in range(100):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(x @ w + b - y))
    dw, db = tape.gradient(loss, [w, b])
    optimizer.apply_gradients([(dw, w), (db, b)])
print(w.numpy(), b.numpy())
# المتوقع: w تقترب من 2 وb تقترب من 1.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-06-ch06-neural-nets.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = keras.Sequential([
    layers.Input((28, 28)), layers.Flatten(),
    layers.Dense(128, activation="relu"), layers.Dense(10),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
print(model.evaluate(x_test, y_test, verbose=0))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-06-ch06-neural-nets.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers
classifier = keras.Sequential([layers.Input((28, 28, 1)), layers.Flatten(),
    layers.Dense(64, activation="relu"), layers.Dense(10)])
classifier.compile(optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-06-ch06-neural-nets.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
keras.utils.set_random_seed(42)
