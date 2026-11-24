#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — نماذج الانتشار: التوليد الحديث
Professional TensorFlow — Diffusion

Book slug : tensorflow-professional-ar
Chapter   : ch-16-ch16-diffusion (lesson 16)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-16-ch16-diffusion.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — نماذج الانتشار: التوليد الحديث
Professional TensorFlow — Diffusion

Book slug : tensorflow-professional-ar
Chapter   : ch-16-ch16-diffusion (lesson 16)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch16_diffusion.md
# ────────────────────────────────────────────────────────────────────────
alpha_bar = 0.7
epsilon = tf.random.normal(tf.shape(x0))
xt = alpha_bar**0.5 * x0 + (1.0-alpha_bar)**0.5 * epsilon

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch16_diffusion.md
# ────────────────────────────────────────────────────────────────────────
with tf.GradientTape() as tape:
    predicted_noise = denoiser([xt, timesteps], training=True)
    loss = tf.reduce_mean(tf.square(epsilon - predicted_noise))

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch16_diffusion.md
# ────────────────────────────────────────────────────────────────────────
time_embedding = keras.layers.Embedding(input_dim=num_steps, output_dim=64)(timesteps)
conditioned = keras.layers.Concatenate()([pooled_image_features, time_embedding])

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-16-ch16-diffusion.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
alpha_bar = 0.7
epsilon = tf.random.normal(tf.shape(x0))
xt = alpha_bar**0.5 * x0 + (1.0-alpha_bar)**0.5 * epsilon

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-16-ch16-diffusion.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
with tf.GradientTape() as tape:
    predicted_noise = denoiser([xt, timesteps], training=True)
    loss = tf.reduce_mean(tf.square(epsilon - predicted_noise))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-16-ch16-diffusion.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
time_embedding = keras.layers.Embedding(input_dim=num_steps, output_dim=64)(timesteps)
conditioned = keras.layers.Concatenate()([pooled_image_features, time_embedding])
