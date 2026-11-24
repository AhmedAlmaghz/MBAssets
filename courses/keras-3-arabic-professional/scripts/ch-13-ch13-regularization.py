#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
Keras 3 in Arabic — Regularization

Book slug : keras-3-arabic-professional
Chapter   : ch-13-ch13-regularization (lesson 13)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-13-ch13-regularization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
Keras 3 in Arabic — Regularization

Book slug : keras-3-arabic-professional
Chapter   : ch-13-ch13-regularization (lesson 13)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-13-ch13-regularization.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
Keras 3 in Arabic — Regularization

Book slug : keras-3-arabic-professional
Chapter   : ch-13-ch13-regularization (lesson 13)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch13_regularization/001_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : التنظيمات والقيود المدمجة
# المصدر  : src/chapters/ch13_regularization.md:100
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch13_regularization/002_overfitting_example_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : مثال تطبيقي كامل: اكتشاف فرط التكيّف (overfitting
# المصدر  : src/chapters/ch13_regularization.md:126
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch13_regularization/003_lambda_validation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : تقاطع التحقّق: اختر $\lambda$ بالأدلة
# المصدر  : src/chapters/ch13_regularization.md:181
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch13_regularization/004_s13_5_example_solved_one_step_diagnosis.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:240
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch13_regularization/005_s13_5_example_solved_one_step_diagnosis.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:282
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : التنظيمات والقيود المدمجة
# المصدر  : src/chapters/ch13_regularization.md:100
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : مثال تطبيقي كامل: اكتشاف فرط التكيّف (overfitting
# المصدر  : src/chapters/ch13_regularization.md:126
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : تقاطع التحقّق: اختر $\lambda$ بالأدلة
# المصدر  : src/chapters/ch13_regularization.md:181
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:240
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:282
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
Keras 3 in Arabic — Regularization

Book slug : keras-3-arabic-professional
Chapter   : ch-13-ch13-regularization (lesson 13)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch13_regularization/001_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : التنظيمات والقيود المدمجة
# المصدر  : src/chapters/ch13_regularization.md:100
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch13_regularization/002_overfitting_example_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : مثال تطبيقي كامل: اكتشاف فرط التكيّف (overfitting
# المصدر  : src/chapters/ch13_regularization.md:126
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch13_regularization/003_lambda_validation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : تقاطع التحقّق: اختر $\lambda$ بالأدلة
# المصدر  : src/chapters/ch13_regularization.md:181
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch13_regularization/004_s13_5_example_solved_one_step_diagnosis.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:240
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch13_regularization/005_s13_5_example_solved_one_step_diagnosis.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:282
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch13_regularization.md
# ────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : التنظيمات والقيود المدمجة
# المصدر  : src/chapters/ch13_regularization.md:100
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : مثال تطبيقي كامل: اكتشاف فرط التكيّف (overfitting
# المصدر  : src/chapters/ch13_regularization.md:126
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : تقاطع التحقّق: اختر $\lambda$ بالأدلة
# المصدر  : src/chapters/ch13_regularization.md:181
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:240
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:282
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# ── Dense مع تنظيم L2 على الأوزان وعلى النشاط
layer = keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_regularizer=keras.regularizers.L2(l2=0.001),
    bias_regularizer=keras.regularizers.L1(0.0001),
    activity_regularizer=keras.regularizers.L2(0.001),
    kernel_constraint=keras.constraints.MaxNorm(max_value=3.0),
)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np

# مؤشّر ألفا-بيتا (grid) على 5 طيّات
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
folds = 5
split = np.array_split(np.arange(len(x_train)), folds)

scores = {}
for alpha in alphas:
    fold_losses = []
    for i in range(folds):
        val_idx = split[i]
        tr_idx = np.concatenate([split[j] for j in range(folds) if j != i])
        m = build_model(alpha)
        m.fit(x_train[tr_idx], y_train[tr_idx], epochs=25, batch_size=64, verbose=0)
        fold_losses.append(float(m.evaluate(x_train[val_idx], y_train[val_idx], verbose=0)[0]))
    scores[alpha] = float(np.mean(fold_losses))
    print(alpha, "->", scores[alpha])

best_alpha = min(scores, key=scores.get)
print("أفضل alpha:", best_alpha)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-13-ch13-regularization.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)
