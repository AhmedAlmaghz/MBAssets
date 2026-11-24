#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — بيئة 2026: Python 3.14 وTensorFlow 2.21
Professional TensorFlow — Environment

Book slug : tensorflow-professional-ar
Chapter   : ch-02-ch02-environment (lesson 2)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-02-ch02-environment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — بيئة 2026: Python 3.14 وTensorFlow 2.21
Professional TensorFlow — Environment

Book slug : tensorflow-professional-ar
Chapter   : ch-02-ch02-environment (lesson 2)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch02_environment.md
# ────────────────────────────────────────────────────────────────────────
import sys
import tensorflow as tf

print(sys.version)
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
# المتوقع: إصدار Python، 2.21.0، ثم قائمة الأجهزة التي اكتشفها TensorFlow.

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch02_environment.md
# ────────────────────────────────────────────────────────────────────────
import sys
print(sys.executable)
print(sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch02_environment.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
with tf.device("/CPU:0"):
    result = tf.reduce_sum(tf.ones((128, 128)))
print(float(result))  # المتوقع: 16384.0

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-02-ch02-environment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import sys
import tensorflow as tf

print(sys.version)
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
# المتوقع: إصدار Python، 2.21.0، ثم قائمة الأجهزة التي اكتشفها TensorFlow.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-02-ch02-environment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import sys
print(sys.executable)
print(sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-02-ch02-environment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
with tf.device("/CPU:0"):
    result = tf.reduce_sum(tf.ones((128, 128)))
print(float(result))  # المتوقع: 16384.0
