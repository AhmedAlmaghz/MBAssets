#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التنسورات: التفكير بالأبعاد
Professional TensorFlow — Tensors

Book slug : tensorflow-professional-ar
Chapter   : ch-03-ch03-tensors (lesson 3)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-03-ch03-tensors.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — التنسورات: التفكير بالأبعاد
Professional TensorFlow — Tensors

Book slug : tensorflow-professional-ar
Chapter   : ch-03-ch03-tensors (lesson 3)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print(x.shape, x.dtype, x.ndim)
print(x[:, 1])
# المتوقع: (2, 3) وfloat32 و2، ثم [2. 5.]

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

# محاكاة دفعة صور RGB بأعداد صحيحة بين 0 و255.
raw = tf.random.uniform((2, 128, 192, 3), maxval=256, dtype=tf.int32)
raw = tf.cast(raw, tf.uint8)
images = tf.image.convert_image_dtype(raw, tf.float32)

one_image = images[0]
green_without_axis = images[..., 1]
green_with_axis = images[..., 1:2]
print(images.shape, one_image.shape)
print(green_without_axis.shape, green_with_axis.shape)
# المتوقع: (2, 128, 192, 3) و(128, 192, 3)
# ثم (2, 128, 192) و(2, 128, 192, 1).

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.reshape(tf.range(12), (3, 4))
column = tf.expand_dims(x[:, 0], axis=-1)
joined = tf.concat([column, column], axis=1)
stacked = tf.stack([column, column], axis=0)
print(x.shape, column.shape, joined.shape, stacked.shape)
# المتوقع: (3, 4)، (3, 1)، (3, 2)، (2, 3, 1)

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

images = tf.zeros((8, 32, 32, 3))
mean = tf.constant([0.5, 0.4, 0.3])
normalized = images - mean
print(normalized.shape)
# المتوقع: (8, 32, 32, 3)، إذ بُثّ متوسط القنوات على المحاور السابقة.

# ────────────────────────────────────────────────────────────────────────
# [005] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

signature = tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
@tf.function(input_signature=[signature])
def normalize_batch(values):
    tf.debugging.assert_all_finite(values, "المدخل يحتوي NaN أو Inf")
    return tf.math.l2_normalize(values, axis=-1)

print(normalize_batch(tf.ones((3, 4))).shape)
print(normalize_batch(tf.ones((12, 4))).shape)
# كلا الحجمين يعيدان رسمًا واحدًا يقبل دفعة متغيرة.

# ────────────────────────────────────────────────────────────────────────
# [006] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

scores = tf.constant([[0.2, 0.8, 0.4], [0.9, 0.1, 0.5]])
row_thresholds = tf.constant([[0.5], [0.3]])
decisions = scores >= row_thresholds
print(decisions.numpy())

# ────────────────────────────────────────────────────────────────────────
# [007] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
x = tf.zeros((4, 3, 32, 32))
y = tf.transpose(x, (0, 2, 3, 1))
print(y.shape)  # (4, 32, 32, 3)

# ────────────────────────────────────────────────────────────────────────
# [008] book/src/chapters/ch03_tensors.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
tokens = tf.ragged.constant([[4, 7, 2], [9, 1]])
print(tokens.row_lengths().numpy())  # [3 2]

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print(x.shape, x.dtype, x.ndim)
print(x[:, 1])
# المتوقع: (2, 3) وfloat32 و2، ثم [2. 5.]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

# محاكاة دفعة صور RGB بأعداد صحيحة بين 0 و255.
raw = tf.random.uniform((2, 128, 192, 3), maxval=256, dtype=tf.int32)
raw = tf.cast(raw, tf.uint8)
images = tf.image.convert_image_dtype(raw, tf.float32)

one_image = images[0]
green_without_axis = images[..., 1]
green_with_axis = images[..., 1:2]
print(images.shape, one_image.shape)
print(green_without_axis.shape, green_with_axis.shape)
# المتوقع: (2, 128, 192, 3) و(128, 192, 3)
# ثم (2, 128, 192) و(2, 128, 192, 1).

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

x = tf.reshape(tf.range(12), (3, 4))
column = tf.expand_dims(x[:, 0], axis=-1)
joined = tf.concat([column, column], axis=1)
stacked = tf.stack([column, column], axis=0)
print(x.shape, column.shape, joined.shape, stacked.shape)
# المتوقع: (3, 4)، (3, 1)، (3, 2)، (2, 3, 1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

images = tf.zeros((8, 32, 32, 3))
mean = tf.constant([0.5, 0.4, 0.3])
normalized = images - mean
print(normalized.shape)
# المتوقع: (8, 32, 32, 3)، إذ بُثّ متوسط القنوات على المحاور السابقة.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

signature = tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
@tf.function(input_signature=[signature])
def normalize_batch(values):
    tf.debugging.assert_all_finite(values, "المدخل يحتوي NaN أو Inf")
    return tf.math.l2_normalize(values, axis=-1)

print(normalize_batch(tf.ones((3, 4))).shape)
print(normalize_batch(tf.ones((12, 4))).shape)
# كلا الحجمين يعيدان رسمًا واحدًا يقبل دفعة متغيرة.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf

scores = tf.constant([[0.2, 0.8, 0.4], [0.9, 0.1, 0.5]])
row_thresholds = tf.constant([[0.5], [0.3]])
decisions = scores >= row_thresholds
print(decisions.numpy())

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
x = tf.zeros((4, 3, 32, 32))
y = tf.transpose(x, (0, 2, 3, 1))
print(y.shape)  # (4, 32, 32, 3)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/tensorflow-professional-ar/notebooks/ch-03-ch03-tensors.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
tokens = tf.ragged.constant([[4, 7, 2], [9, 1]])
print(tokens.row_lengths().numpy())  # [3 2]
