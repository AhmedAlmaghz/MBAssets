# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : 8.9 مثال محلول: اختيار الخسارة لكل حالة
# المصدر  : src/chapters/ch08_loss_functions.md:245
# الترتيب : 6/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# الحالة 1: انحدار على أسعار مع outliers
reg = keras.Sequential([
    keras.layers.Input((20,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1),
])
reg.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.Huber(delta=1.0),
            metrics=[keras.metrics.MeanAbsoluteError()])

# الحالة 2: تصنيف متعدّد مع one-hot
clf = keras.Sequential([
    keras.layers.Input((100,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(5),                       # logits
])
clf.compile(optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy()])

# الحالة 3: تصنيف ثنائي
bin_clf = keras.Sequential([
    keras.layers.Input((30,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),                       # logit واحد
])
bin_clf.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()])
