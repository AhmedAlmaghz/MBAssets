#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — TensorFlow: الحكاية والمنظومة والمكانة
Professional TensorFlow — Story Ecosystem

Book slug : tensorflow-professional-ar
Chapter   : ch-01-ch01-story-ecosystem (lesson 1)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-01-ch01-story-ecosystem.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — TensorFlow: الحكاية والمنظومة والمكانة
Professional TensorFlow — Story Ecosystem

Book slug : tensorflow-professional-ar
Chapter   : ch-01-ch01-story-ecosystem (lesson 1)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch01_story_ecosystem.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
x = tf.constant([1.0, 2.0, 3.0])
print(x * 2)   # tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32)

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch01_story_ecosystem.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)          # نتائج قابلة للتكرار قدر الإمكان

# 1) بيانات صغيرة: سعر بالدولار → سعر بالريال (علاقة خطية تقريبية)
x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype="float32")
y = np.array([[3.75], [7.50], [11.25], [15.00], [18.75]], dtype="float32")

# 2) النموذج: طبقة خطية واحدة (Dense) بلا دالة تنشيط
model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=1),
])

# 3) التجميع: خسارة MSE، محسّن SGD بمعدل تعلّم صغير
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.05),
    loss=keras.losses.MeanSquaredError(),
    metrics=["mae"],
)

# 4) التدريب: 200 دورة على خمس نقاط
history = model.fit(x, y, epochs=200, verbose=0)

# 5) الفحص
print("W =", model.layers[0].get_weights()[0].ravel()[0])
print("b =", model.layers[0].get_weights()[1][0])
print("التنبؤ بسعر 7$ =", model.predict(np.array([[7.0]]), verbose=0)[0][0])
# النتيجة المتوقعة: W قريب من 3.75، b قريب من 0، والتنبؤ قريب من 26.25

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch01_story_ecosystem.md
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
history = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=30, callbacks=callbacks)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-01-ch01-story-ecosystem.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
x = tf.constant([1.0, 2.0, 3.0])
print(x * 2)   # tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-01-ch01-story-ecosystem.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)          # نتائج قابلة للتكرار قدر الإمكان

# 1) بيانات صغيرة: سعر بالدولار → سعر بالريال (علاقة خطية تقريبية)
x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype="float32")
y = np.array([[3.75], [7.50], [11.25], [15.00], [18.75]], dtype="float32")

# 2) النموذج: طبقة خطية واحدة (Dense) بلا دالة تنشيط
model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=1),
])

# 3) التجميع: خسارة MSE، محسّن SGD بمعدل تعلّم صغير
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.05),
    loss=keras.losses.MeanSquaredError(),
    metrics=["mae"],
)

# 4) التدريب: 200 دورة على خمس نقاط
history = model.fit(x, y, epochs=200, verbose=0)

# 5) الفحص
print("W =", model.layers[0].get_weights()[0].ravel()[0])
print("b =", model.layers[0].get_weights()[1][0])
print("التنبؤ بسعر 7$ =", model.predict(np.array([[7.0]]), verbose=0)[0][0])
# النتيجة المتوقعة: W قريب من 3.75، b قريب من 0، والتنبؤ قريب من 26.25

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-01-ch01-story-ecosystem.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
history = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=30, callbacks=callbacks)
