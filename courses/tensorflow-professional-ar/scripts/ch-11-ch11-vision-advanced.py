#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الرؤية المتقدمة: النقل والكشف والتجزئة
Professional TensorFlow — Vision Advanced

Book slug : tensorflow-professional-ar
Chapter   : ch-11-ch11-vision-advanced (lesson 11)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-11-ch11-vision-advanced.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — الرؤية المتقدمة: النقل والكشف والتجزئة
Professional TensorFlow — Vision Advanced

Book slug : tensorflow-professional-ar
Chapter   : ch-11-ch11-vision-advanced (lesson 11)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch11_vision_advanced.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                         input_shape=(224, 224, 3), pooling="avg")
base.trainable = False
inputs = keras.Input((224, 224, 3))
x = keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
outputs = layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch11_vision_advanced.md
# ────────────────────────────────────────────────────────────────────────
base.trainable = False  # درب رأس التصنيف أولًا
# افتح بعض الطبقات العلويا ثم أعد التجميع بمعدل أصغر

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch11_vision_advanced.md
# ────────────────────────────────────────────────────────────────────────
intersection = 40.0
area_a, area_b = 100.0, 120.0
iou = intersection / (area_a + area_b - intersection)
print(round(iou, 3))

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-11-ch11-vision-advanced.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                         input_shape=(224, 224, 3), pooling="avg")
base.trainable = False
inputs = keras.Input((224, 224, 3))
x = keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
outputs = layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-11-ch11-vision-advanced.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
base.trainable = False  # درب رأس التصنيف أولًا
# افتح بعض الطبقات العلويا ثم أعد التجميع بمعدل أصغر

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-11-ch11-vision-advanced.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
intersection = 40.0
area_a, area_b = 100.0, 120.0
iou = intersection / (area_a + area_b - intersection)
print(round(iou, 3))
