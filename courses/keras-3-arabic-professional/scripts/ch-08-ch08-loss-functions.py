#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
Keras 3 in Arabic — Loss Functions

Book slug : keras-3-arabic-professional
Chapter   : ch-08-ch08-loss-functions (lesson 8)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-08-ch08-loss-functions.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
Keras 3 in Arabic — Loss Functions

Book slug : keras-3-arabic-professional
Chapter   : ch-08-ch08-loss-functions (lesson 8)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-08-ch08-loss-functions.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
Keras 3 in Arabic — Loss Functions

Book slug : keras-3-arabic-professional
Chapter   : ch-08-ch08-loss-functions (lesson 8)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch08_loss_functions/001_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : النسخة المتفرّقة
# المصدر  : src/chapters/ch08_loss_functions.md:90
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch08_loss_functions/002_label_smoothing.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : معامل تنعيم التسميات (Label Smoothing)
# المصدر  : src/chapters/ch08_loss_functions.md:104
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch08_loss_functions/003_s8_5_loss_custom.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : 8.5 خسارة مخصّصة
# المصدر  : src/chapters/ch08_loss_functions.md:139
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch08_loss_functions/004_sample_weight.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `sample_weight`: وزن لكل عيّنة
# المصدر  : src/chapters/ch08_loss_functions.md:196
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch08_loss_functions/005_class_weight.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `class_weight`: وزن لكل صنف
# المصدر  : src/chapters/ch08_loss_functions.md:207
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch08_loss_functions/006_s8_9_example_solved_loss.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : النسخة المتفرّقة
# المصدر  : src/chapters/ch08_loss_functions.md:90
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : معامل تنعيم التسميات (Label Smoothing)
# المصدر  : src/chapters/ch08_loss_functions.md:104
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : 8.5 خسارة مخصّصة
# المصدر  : src/chapters/ch08_loss_functions.md:139
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `sample_weight`: وزن لكل عيّنة
# المصدر  : src/chapters/ch08_loss_functions.md:196
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `class_weight`: وزن لكل صنف
# المصدر  : src/chapters/ch08_loss_functions.md:207
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
Keras 3 in Arabic — Loss Functions

Book slug : keras-3-arabic-professional
Chapter   : ch-08-ch08-loss-functions (lesson 8)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch08_loss_functions/001_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : النسخة المتفرّقة
# المصدر  : src/chapters/ch08_loss_functions.md:90
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch08_loss_functions/002_label_smoothing.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : معامل تنعيم التسميات (Label Smoothing)
# المصدر  : src/chapters/ch08_loss_functions.md:104
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch08_loss_functions/003_s8_5_loss_custom.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : 8.5 خسارة مخصّصة
# المصدر  : src/chapters/ch08_loss_functions.md:139
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch08_loss_functions/004_sample_weight.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `sample_weight`: وزن لكل عيّنة
# المصدر  : src/chapters/ch08_loss_functions.md:196
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch08_loss_functions/005_class_weight.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `class_weight`: وزن لكل صنف
# المصدر  : src/chapters/ch08_loss_functions.md:207
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch08_loss_functions/006_s8_9_example_solved_loss.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch08_loss_functions.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : النسخة المتفرّقة
# المصدر  : src/chapters/ch08_loss_functions.md:90
# الترتيب : 1/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : معامل تنعيم التسميات (Label Smoothing)
# المصدر  : src/chapters/ch08_loss_functions.md:104
# الترتيب : 2/6
# ──────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : 8.5 خسارة مخصّصة
# المصدر  : src/chapters/ch08_loss_functions.md:139
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `sample_weight`: وزن لكل عيّنة
# المصدر  : src/chapters/ch08_loss_functions.md:196
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `class_weight`: وزن لكل صنف
# المصدر  : src/chapters/ch08_loss_functions.md:207
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

class FocalLoss(keras.losses.Loss):
    """خسارة بتركيز على الأمثلة الصعبة (من المفيد مع البيانات غير المتوازنة)."""

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.convert_to_tensor(y_true)

        if self.from_logits:
            # احسب log-probabilities بطريقة مستقرة
            log_p = keras.ops.log_softmax(y_pred, axis=-1)
            ce = -y_true * log_p
        else:
            p = keras.ops.clip(y_pred, 1e-7, 1.0)
            ce = -y_true * keras.ops.log(p)

        # p_t: احتمال الصنف الحقيقي
        p_t = keras.ops.sum(y_true * keras.ops.exp(log_p), axis=-1) if self.from_logits \
            else keras.ops.sum(y_true * y_pred, axis=-1)

        # (1 - p_t)^gamma يقلّل وزن الأمثلة السهلة
        loss = self.alpha * ((1 - p_t) ** self.gamma) * keras.ops.sum(ce, axis=-1)
        return keras.ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                       "from_logits": self.from_logits})
        return config


# الاستخدام
model.compile(
    optimizer="adam",
    loss=FocalLoss(gamma=2.0, alpha=0.25, from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class_weight = {0: 0.56, 1: 1.11, 2: 3.33}
model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-08-ch08-loss-functions.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
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
