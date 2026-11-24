#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — tf.data: هندسة خطوط البيانات
Professional TensorFlow — Data

Book slug : tensorflow-professional-ar
Chapter   : ch-09-ch09-data (lesson 9)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-09-ch09-data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — tf.data: هندسة خطوط البيانات
Professional TensorFlow — Data

Book slug : tensorflow-professional-ar
Chapter   : ch-09-ch09-data (lesson 9)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch09_data.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.range(1000, dtype=tf.float32)
y = 3.0 * x + 1.0
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.shuffle(1000, seed=42, reshuffle_each_iteration=True)
ds = ds.batch(32, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
for xb, yb in ds.take(1):
    print(xb.shape, yb.shape)
# المتوقع: دفعة حتى 32 قيمة ودفعة أهداف بالشكل نفسه.

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch09_data.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.shuffle(10_000, seed=42, reshuffle_each_iteration=True)
ds = ds.map(parse_and_augment, num_parallel_calls=AUTOTUNE)
ds = ds.batch(64, drop_remainder=False)
ds = ds.prefetch(AUTOTUNE)

for batch_x, batch_y in ds.take(1):
    print(batch_x.shape, batch_y.shape)

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch09_data.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

def read_image(path, label):
    encoded = tf.io.read_file(path)
    image = tf.io.decode_jpeg(encoded, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
train_ds = train_ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch09_data.md
# ────────────────────────────────────────────────────────────────────────
ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

# ────────────────────────────────────────────────────────────────────────
# [005] book/src/chapters/ch09_data.md
# ────────────────────────────────────────────────────────────────────────
for features, labels in ds.take(1):
    print(features.shape, labels.shape)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-09-ch09-data.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.range(1000, dtype=tf.float32)
y = 3.0 * x + 1.0
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.shuffle(1000, seed=42, reshuffle_each_iteration=True)
ds = ds.batch(32, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
for xb, yb in ds.take(1):
    print(xb.shape, yb.shape)
# المتوقع: دفعة حتى 32 قيمة ودفعة أهداف بالشكل نفسه.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-09-ch09-data.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.shuffle(10_000, seed=42, reshuffle_each_iteration=True)
ds = ds.map(parse_and_augment, num_parallel_calls=AUTOTUNE)
ds = ds.batch(64, drop_remainder=False)
ds = ds.prefetch(AUTOTUNE)

for batch_x, batch_y in ds.take(1):
    print(batch_x.shape, batch_y.shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-09-ch09-data.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

def read_image(path, label):
    encoded = tf.io.read_file(path)
    image = tf.io.decode_jpeg(encoded, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
train_ds = train_ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-09-ch09-data.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/tensorflow-professional-ar/notebooks/ch-09-ch09-data.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
for features, labels in ds.take(1):
    print(features.shape, labels.shape)
