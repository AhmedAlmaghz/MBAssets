#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التوليد: Autoencoder وVAE وGAN
Professional TensorFlow — Generative

Book slug : tensorflow-professional-ar
Chapter   : ch-15-ch15-generative (lesson 15)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-15-ch15-generative.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التوليد: Autoencoder وVAE وGAN
Professional TensorFlow — Generative

Book slug : tensorflow-professional-ar
Chapter   : ch-15-ch15-generative (lesson 15)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch15_generative.md
# ────────────────────────────────────────────────────────────────────────
inputs = keras.Input((28, 28, 1))
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(16, activation="relu")(z)
outputs = keras.layers.Dense(784, activation="sigmoid")(z)
outputs = keras.layers.Reshape((28, 28, 1))(outputs)
autoencoder = keras.Model(inputs, outputs)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch15_generative.md
# ────────────────────────────────────────────────────────────────────────
noise = tf.random.normal(tf.shape(images), stddev=0.1)
noisy_images = tf.clip_by_value(images + noise, 0.0, 1.0)

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch15_generative.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

def sample_latent(mean, log_var):
    epsilon = tf.random.normal(tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

def kl_per_example(mean, log_var):
    term = 1.0 + log_var - tf.square(mean) - tf.exp(log_var)
    return -0.5 * tf.reduce_sum(term, axis=-1)

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch15_generative.md
# ────────────────────────────────────────────────────────────────────────
z = tf.random.normal((16, latent_dim))
samples = decoder(z, training=False)
print(samples.shape)  # 16 عينات مولدة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-15-ch15-generative.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
inputs = keras.Input((28, 28, 1))
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(16, activation="relu")(z)
outputs = keras.layers.Dense(784, activation="sigmoid")(z)
outputs = keras.layers.Reshape((28, 28, 1))(outputs)
autoencoder = keras.Model(inputs, outputs)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-15-ch15-generative.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
noise = tf.random.normal(tf.shape(images), stddev=0.1)
noisy_images = tf.clip_by_value(images + noise, 0.0, 1.0)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-15-ch15-generative.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

def sample_latent(mean, log_var):
    epsilon = tf.random.normal(tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

def kl_per_example(mean, log_var):
    term = 1.0 + log_var - tf.square(mean) - tf.exp(log_var)
    return -0.5 * tf.reduce_sum(term, axis=-1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-15-ch15-generative.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
z = tf.random.normal((16, latent_dim))
samples = decoder(z, training=False)
print(samples.shape)  # 16 عينات مولدة
