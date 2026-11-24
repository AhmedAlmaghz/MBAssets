#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — المحسّنات والتنظيم: فنّ التدريب
Professional TensorFlow — Training

Book slug : tensorflow-professional-ar
Chapter   : ch-07-ch07-training (lesson 7)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-07-ch07-training.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — المحسّنات والتنظيم: فنّ التدريب
Professional TensorFlow — Training

Book slug : tensorflow-professional-ar
Chapter   : ch-07-ch07-training (lesson 7)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch07_training.md
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input((64,)),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.25),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
              loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
callbacks = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                           patience=3, restore_best_weights=True)]

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch07_training.md
# ────────────────────────────────────────────────────────────────────────
import keras

# اضبط السياسة قبل بناء النموذج، ثم تحقق من dtype للطبقات والمخرجات.
keras.mixed_precision.set_global_policy("mixed_float16")
model = make_model()
print(model.layers[0].compute_dtype, model.layers[0].variable_dtype)

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch07_training.md
# ────────────────────────────────────────────────────────────────────────
optimizer = keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy")

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch07_training.md
# ────────────────────────────────────────────────────────────────────────
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=1000, alpha=0.1)
optimizer = keras.optimizers.Adam(learning_rate=schedule)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-07-ch07-training.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

model = keras.Sequential([
    layers.Input((64,)),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.25),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
              loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
callbacks = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                           patience=3, restore_best_weights=True)]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-07-ch07-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras

# اضبط السياسة قبل بناء النموذج، ثم تحقق من dtype للطبقات والمخرجات.
keras.mixed_precision.set_global_policy("mixed_float16")
model = make_model()
print(model.layers[0].compute_dtype, model.layers[0].variable_dtype)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-07-ch07-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
optimizer = keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-07-ch07-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=1000, alpha=0.1)
optimizer = keras.optimizers.Adam(learning_rate=schedule)
