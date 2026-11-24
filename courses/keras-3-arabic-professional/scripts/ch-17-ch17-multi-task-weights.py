#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
Keras 3 in Arabic — Multi Task Weights

Book slug : keras-3-arabic-professional
Chapter   : ch-17-ch17-multi-task-weights (lesson 17)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-17-ch17-multi-task-weights.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
Keras 3 in Arabic — Multi Task Weights

Book slug : keras-3-arabic-professional
Chapter   : ch-17-ch17-multi-task-weights (lesson 17)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-17-ch17-multi-task-weights.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
Keras 3 in Arabic — Multi Task Weights

Book slug : keras-3-arabic-professional
Chapter   : ch-17-ch17-multi-task-weights (lesson 17)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch17_multi_task_weights/001_weighted_metrics_measure.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : `weighted_metrics`: قياس مرجّح
# المصدر  : src/chapters/ch17_multi_task_weights.md:57
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch17_multi_task_weights/002_first_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الأول: مدخلات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:106
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# مدخلات متعدّدة: صورة + متجه خصائص
img_input = keras.Input(shape=(32, 32, 3), name="image")
feat_input = keras.Input(shape=(10,), name="features")

x = keras.layers.Conv2D(32, 3, activation="relu")(img_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)

y = keras.layers.Dense(32, activation="relu")(feat_input)

merged = keras.layers.Concatenate()([x, y])
out = keras.layers.Dense(4, activation="softmax")(merged)

model = keras.Model(inputs=[img_input, feat_input], outputs=out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # التسميات أعداد صحيحة
    metrics=["accuracy"],
)
model.summary()

# بيانات تجريبية
imgs = np.random.rand(64, 32, 32, 3).astype("float32")
feats = np.random.rand(64, 10).astype("float32")
labels = np.random.randint(0, 4, size=64)

history = model.fit(
    {"image": imgs, "features": feats},
    labels,
    epochs=3,
    batch_size=8,
    verbose=0,
)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch17_multi_task_weights/003_second_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثاني: مخرجات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:153
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مخرجان: تصنيف العيّنة + انحدار العيب
class_out = keras.Input(shape=(1,), name="y_class")
reg_out = keras.Input(shape=(1,), name="y_reg")

inp = keras.Input(shape=(20,), name="features")
h = keras.layers.Dense(64, activation="relu")(inp)
h = keras.layers.Dense(32, activation="relu")(h)

pred_class = keras.layers.Dense(3, activation="softmax", name="class")(h)
pred_reg = keras.layers.Dense(1, name="reg")(h)

multi = keras.Model(inputs=inp, outputs=[pred_class, pred_reg])
multi.compile(
    optimizer="adam",
    loss={
        "class": "sparse_categorical_crossentropy",
        "reg": "mse",
    },
    loss_weights={
        "class": 1.0,
        "reg": 0.1,     # وازن بين المخرجين
    },
    metrics={
        "class": ["accuracy"],
        "reg": ["mae"],
    },
)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch17_multi_task_weights/004_multi_task_third.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثالث: تعلّم متعدّد المهام (Multi-task)
# المصدر  : src/chapters/ch17_multi_task_weights.md:192
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(20,))
shared = keras.layers.Dense(64, activation="relu")(inp)
shared = keras.layers.BatchNormalization()(shared)
shared = keras.layers.Dense(32, activation="relu")(shared)

out_a = keras.layers.Dense(1, activation="sigmoid", name="task_a")(shared)
out_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

mtl = keras.Model(inputs=inp, outputs=[out_a, out_b])
mtl.compile(
    optimizer="adam",
    loss={"task_a": "binary_crossentropy", "task_b": "binary_crossentropy"},
    loss_weights={"task_a": 1.0, "task_b": 0.5},
)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch17_multi_task_weights/005_focal_loss.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : Focal Loss: لماذا يعمل؟
# المصدر  : src/chapters/ch17_multi_task_weights.md:244
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras

loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=False,
    alpha=0.25,     # وزن الصنف النادر
    gamma=2.0,      # معامل التركيز
)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch17_multi_task_weights/006_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:261
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

counts = np.array([98000, 2000])           # 98% / 2%
n = counts.sum()
weights = n / (len(counts) * counts)       # [0.510, 25.0]
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print(class_weight)   # النتيجة المتوقعة: {0: 0.5102, 1: 25.0}

model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch17_multi_task_weights/007_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:278
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=True,
    alpha=0.75,     # أنزّل وزن الصنف الشائع
    gamma=2.0,
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss,
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC(),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch17_multi_task_weights/008_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:300
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: جرّب عتبات مختلفة على الاحتمالات
probs = model.predict(x_val).ravel()
y_true = y_val.ravel()
for thr in (0.3, 0.5, 0.7):
    pred = (probs >= thr).astype("float32")
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"thr={thr}: P={precision:.3f} R={recall:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch17_multi_task_weights/009_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:318
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=5,
                                   restore_best_weights=True),
]

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
> keras.layers.Dense(C)   # بلا activation
> loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
> ```

## 17.3 المهام المتعدّدة: ثلاثة أنماط

### النمط الأول: مدخلات متعدّدة

When كل عيّنة تحمل أكثر من نوع من الخصائص. مثال: تصنيف منتج من (صورة + وصف نصّي + السعر).

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
**متى يتفوّق على نموذجين منفصلين؟**
- المهام **مرتبطة** (تشترك في بنية تمثيل مفيدة).
- البيانات **محدودة** (المهمة النادرة تستفيد من بيانات المهمة الغزيرة عبر التمثيل المشترك).
- تحتاج **حجم نموذج** أصغر في الإنتاج (شبكة واحدة).

**متى يكون أسوأ؟**
- المهام **متنافسة** أو غير مرتبطة؛ عندها يتضارب التدرّجان.
- كل مهمة تحتاج **سعة** مختلفة كثيرًا.

> **نصيحة:** وزّع الخسائر بحيث تعكس تكرار المهام. المهمة النادرة تحتاج وزنًا أكبر عادةً، لا أصغر.

## 17.4 خمس أدوات لعدم التوازن

| الأداة | المبدأ | متى تختارها | الكلفة |
|---|---|---|---|
| `class_weight` | وزن في الخسارة | أسرع حل، فعّال عادةً | قد يخفض الدقّة العامة |
| `sample_weight` | وزن لكل عيّنة | جودة متفاوتة للعيّنات | تحتاج حساب الأوزان |
| `Data augmentation` | توليد تنوّع | أمثلة نادرة قليلة | تحتاج عمليات مناسبة |
| `Focal loss` | تركيز على الصعب | تشبّه extreme imbalance | تتطلّب `BinaryFocalCrossentropy` |
| `Resampling` | أكثر/أقلّ عيّنات | تحكّم كامل | خطر تسرّب أو تشويه التوزيع |

### Focal Loss: لماذا يعمل؟

الخسارة المتقاطعة العادية تعاقب التنبّؤ **السهل** بنفس القدر الذي تعاقب به التنبّؤ **الصعب**:

$$
\mathcal{L}_{CE} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

حيث $p_t$ احتمال الصنف الحقيقي، و$\gamma \ge 0$ «معامل التركيز». عند $\gamma = 2$ تُقلَّص خسارة الأمثلة السهلة (منخفضة $p_t$ بعد التصحيح) إلى الربع تقريبًا، بينما تبقى خسارة الأمثلة الصعبة شبه كما هي. النتيجة: الشبكة **تتركّز** على النادرة.

Keras 3 يوفّرها جاهزة: `keras.losses.BinaryFocalCrossentropy` و`keras.losses.CategoricalFocalCrossentropy` (مع `alpha` و`gamma`).

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
> **ملاحظة:** اختيار العتبة قرار **تشغيلي** لا رياضي: حدّد من تحلّة الفريق «كم تنبيهًا خاطئًا يمكن تحمّله أسبوعيًا؟» ثم اختر العتبة التي تحقّق ذلك. أداة `keras.metrics.RecallAtPrecision` تساعد في البحث الآلي.

**الخطوة 5 — راقب المقاييس الصحيحة.** مع 2% إيجابيات، لا يصحّ مراقبة `accuracy`. راقب `val_auc` و`val_recall` مع `mode="max"`.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : `weighted_metrics`: قياس مرجّح
# المصدر  : src/chapters/ch17_multi_task_weights.md:57
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الأول: مدخلات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:106
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# مدخلات متعدّدة: صورة + متجه خصائص
img_input = keras.Input(shape=(32, 32, 3), name="image")
feat_input = keras.Input(shape=(10,), name="features")

x = keras.layers.Conv2D(32, 3, activation="relu")(img_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)

y = keras.layers.Dense(32, activation="relu")(feat_input)

merged = keras.layers.Concatenate()([x, y])
out = keras.layers.Dense(4, activation="softmax")(merged)

model = keras.Model(inputs=[img_input, feat_input], outputs=out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # التسميات أعداد صحيحة
    metrics=["accuracy"],
)
model.summary()

# بيانات تجريبية
imgs = np.random.rand(64, 32, 32, 3).astype("float32")
feats = np.random.rand(64, 10).astype("float32")
labels = np.random.randint(0, 4, size=64)

history = model.fit(
    {"image": imgs, "features": feats},
    labels,
    epochs=3,
    batch_size=8,
    verbose=0,
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثاني: مخرجات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:153
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مخرجان: تصنيف العيّنة + انحدار العيب
class_out = keras.Input(shape=(1,), name="y_class")
reg_out = keras.Input(shape=(1,), name="y_reg")

inp = keras.Input(shape=(20,), name="features")
h = keras.layers.Dense(64, activation="relu")(inp)
h = keras.layers.Dense(32, activation="relu")(h)

pred_class = keras.layers.Dense(3, activation="softmax", name="class")(h)
pred_reg = keras.layers.Dense(1, name="reg")(h)

multi = keras.Model(inputs=inp, outputs=[pred_class, pred_reg])
multi.compile(
    optimizer="adam",
    loss={
        "class": "sparse_categorical_crossentropy",
        "reg": "mse",
    },
    loss_weights={
        "class": 1.0,
        "reg": 0.1,     # وازن بين المخرجين
    },
    metrics={
        "class": ["accuracy"],
        "reg": ["mae"],
    },
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثالث: تعلّم متعدّد المهام (Multi-task)
# المصدر  : src/chapters/ch17_multi_task_weights.md:192
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(20,))
shared = keras.layers.Dense(64, activation="relu")(inp)
shared = keras.layers.BatchNormalization()(shared)
shared = keras.layers.Dense(32, activation="relu")(shared)

out_a = keras.layers.Dense(1, activation="sigmoid", name="task_a")(shared)
out_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

mtl = keras.Model(inputs=inp, outputs=[out_a, out_b])
mtl.compile(
    optimizer="adam",
    loss={"task_a": "binary_crossentropy", "task_b": "binary_crossentropy"},
    loss_weights={"task_a": 1.0, "task_b": 0.5},
)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : Focal Loss: لماذا يعمل؟
# المصدر  : src/chapters/ch17_multi_task_weights.md:244
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras

loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=False,
    alpha=0.25,     # وزن الصنف النادر
    gamma=2.0,      # معامل التركيز
)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:261
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

counts = np.array([98000, 2000])           # 98% / 2%
n = counts.sum()
weights = n / (len(counts) * counts)       # [0.510, 25.0]
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print(class_weight)   # النتيجة المتوقعة: {0: 0.5102, 1: 25.0}

model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:278
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=True,
    alpha=0.75,     # أنزّل وزن الصنف الشائع
    gamma=2.0,
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss,
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC(),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:300
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: جرّب عتبات مختلفة على الاحتمالات
probs = model.predict(x_val).ravel()
y_true = y_val.ravel()
for thr in (0.3, 0.5, 0.7):
    pred = (probs >= thr).astype("float32")
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"thr={thr}: P={precision:.3f} R={recall:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:318
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=5,
                                   restore_best_weights=True),
]

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
> keras.layers.Dense(C)   # بلا activation
> loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
> ```

## 17.3 المهام المتعدّدة: ثلاثة أنماط

### النمط الأول: مدخلات متعدّدة

When كل عيّنة تحمل أكثر من نوع من الخصائص. مثال: تصنيف منتج من (صورة + وصف نصّي + السعر).

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
**متى يتفوّق على نموذجين منفصلين؟**
- المهام **مرتبطة** (تشترك في بنية تمثيل مفيدة).
- البيانات **محدودة** (المهمة النادرة تستفيد من بيانات المهمة الغزيرة عبر التمثيل المشترك).
- تحتاج **حجم نموذج** أصغر في الإنتاج (شبكة واحدة).

**متى يكون أسوأ؟**
- المهام **متنافسة** أو غير مرتبطة؛ عندها يتضارب التدرّجان.
- كل مهمة تحتاج **سعة** مختلفة كثيرًا.

> **نصيحة:** وزّع الخسائر بحيث تعكس تكرار المهام. المهمة النادرة تحتاج وزنًا أكبر عادةً، لا أصغر.

## 17.4 خمس أدوات لعدم التوازن

| الأداة | المبدأ | متى تختارها | الكلفة |
|---|---|---|---|
| `class_weight` | وزن في الخسارة | أسرع حل، فعّال عادةً | قد يخفض الدقّة العامة |
| `sample_weight` | وزن لكل عيّنة | جودة متفاوتة للعيّنات | تحتاج حساب الأوزان |
| `Data augmentation` | توليد تنوّع | أمثلة نادرة قليلة | تحتاج عمليات مناسبة |
| `Focal loss` | تركيز على الصعب | تشبّه extreme imbalance | تتطلّب `BinaryFocalCrossentropy` |
| `Resampling` | أكثر/أقلّ عيّنات | تحكّم كامل | خطر تسرّب أو تشويه التوزيع |

### Focal Loss: لماذا يعمل؟

الخسارة المتقاطعة العادية تعاقب التنبّؤ **السهل** بنفس القدر الذي تعاقب به التنبّؤ **الصعب**:

$$
\mathcal{L}_{CE} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

حيث $p_t$ احتمال الصنف الحقيقي، و$\gamma \ge 0$ «معامل التركيز». عند $\gamma = 2$ تُقلَّص خسارة الأمثلة السهلة (منخفضة $p_t$ بعد التصحيح) إلى الربع تقريبًا، بينما تبقى خسارة الأمثلة الصعبة شبه كما هي. النتيجة: الشبكة **تتركّز** على النادرة.

Keras 3 يوفّرها جاهزة: `keras.losses.BinaryFocalCrossentropy` و`keras.losses.CategoricalFocalCrossentropy` (مع `alpha` و`gamma`).

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
> **ملاحظة:** اختيار العتبة قرار **تشغيلي** لا رياضي: حدّد من تحلّة الفريق «كم تنبيهًا خاطئًا يمكن تحمّله أسبوعيًا؟» ثم اختر العتبة التي تحقّق ذلك. أداة `keras.metrics.RecallAtPrecision` تساعد في البحث الآلي.

**الخطوة 5 — راقب المقاييس الصحيحة.** مع 2% إيجابيات، لا يصحّ مراقبة `accuracy`. راقب `val_auc` و`val_recall` مع `mode="max"`.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
Keras 3 in Arabic — Multi Task Weights

Book slug : keras-3-arabic-professional
Chapter   : ch-17-ch17-multi-task-weights (lesson 17)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch17_multi_task_weights/001_weighted_metrics_measure.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : `weighted_metrics`: قياس مرجّح
# المصدر  : src/chapters/ch17_multi_task_weights.md:57
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch17_multi_task_weights/002_first_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الأول: مدخلات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:106
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# مدخلات متعدّدة: صورة + متجه خصائص
img_input = keras.Input(shape=(32, 32, 3), name="image")
feat_input = keras.Input(shape=(10,), name="features")

x = keras.layers.Conv2D(32, 3, activation="relu")(img_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)

y = keras.layers.Dense(32, activation="relu")(feat_input)

merged = keras.layers.Concatenate()([x, y])
out = keras.layers.Dense(4, activation="softmax")(merged)

model = keras.Model(inputs=[img_input, feat_input], outputs=out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # التسميات أعداد صحيحة
    metrics=["accuracy"],
)
model.summary()

# بيانات تجريبية
imgs = np.random.rand(64, 32, 32, 3).astype("float32")
feats = np.random.rand(64, 10).astype("float32")
labels = np.random.randint(0, 4, size=64)

history = model.fit(
    {"image": imgs, "features": feats},
    labels,
    epochs=3,
    batch_size=8,
    verbose=0,
)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch17_multi_task_weights/003_second_multi.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثاني: مخرجات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:153
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مخرجان: تصنيف العيّنة + انحدار العيب
class_out = keras.Input(shape=(1,), name="y_class")
reg_out = keras.Input(shape=(1,), name="y_reg")

inp = keras.Input(shape=(20,), name="features")
h = keras.layers.Dense(64, activation="relu")(inp)
h = keras.layers.Dense(32, activation="relu")(h)

pred_class = keras.layers.Dense(3, activation="softmax", name="class")(h)
pred_reg = keras.layers.Dense(1, name="reg")(h)

multi = keras.Model(inputs=inp, outputs=[pred_class, pred_reg])
multi.compile(
    optimizer="adam",
    loss={
        "class": "sparse_categorical_crossentropy",
        "reg": "mse",
    },
    loss_weights={
        "class": 1.0,
        "reg": 0.1,     # وازن بين المخرجين
    },
    metrics={
        "class": ["accuracy"],
        "reg": ["mae"],
    },
)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch17_multi_task_weights/004_multi_task_third.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثالث: تعلّم متعدّد المهام (Multi-task)
# المصدر  : src/chapters/ch17_multi_task_weights.md:192
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(20,))
shared = keras.layers.Dense(64, activation="relu")(inp)
shared = keras.layers.BatchNormalization()(shared)
shared = keras.layers.Dense(32, activation="relu")(shared)

out_a = keras.layers.Dense(1, activation="sigmoid", name="task_a")(shared)
out_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

mtl = keras.Model(inputs=inp, outputs=[out_a, out_b])
mtl.compile(
    optimizer="adam",
    loss={"task_a": "binary_crossentropy", "task_b": "binary_crossentropy"},
    loss_weights={"task_a": 1.0, "task_b": 0.5},
)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch17_multi_task_weights/005_focal_loss.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : Focal Loss: لماذا يعمل؟
# المصدر  : src/chapters/ch17_multi_task_weights.md:244
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras

loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=False,
    alpha=0.25,     # وزن الصنف النادر
    gamma=2.0,      # معامل التركيز
)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch17_multi_task_weights/006_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:261
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

counts = np.array([98000, 2000])           # 98% / 2%
n = counts.sum()
weights = n / (len(counts) * counts)       # [0.510, 25.0]
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print(class_weight)   # النتيجة المتوقعة: {0: 0.5102, 1: 25.0}

model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch17_multi_task_weights/007_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:278
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=True,
    alpha=0.75,     # أنزّل وزن الصنف الشائع
    gamma=2.0,
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss,
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC(),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch17_multi_task_weights/008_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:300
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: جرّب عتبات مختلفة على الاحتمالات
probs = model.predict(x_val).ravel()
y_true = y_val.ravel()
for thr in (0.3, 0.5, 0.7):
    pred = (probs >= thr).astype("float32")
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"thr={thr}: P={precision:.3f} R={recall:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch17_multi_task_weights/009_s17_5_example_solved_classification_transactions.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:318
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=5,
                                   restore_best_weights=True),
]

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
> keras.layers.Dense(C)   # بلا activation
> loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
> ```

## 17.3 المهام المتعدّدة: ثلاثة أنماط

### النمط الأول: مدخلات متعدّدة

When كل عيّنة تحمل أكثر من نوع من الخصائص. مثال: تصنيف منتج من (صورة + وصف نصّي + السعر).

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
**متى يتفوّق على نموذجين منفصلين؟**
- المهام **مرتبطة** (تشترك في بنية تمثيل مفيدة).
- البيانات **محدودة** (المهمة النادرة تستفيد من بيانات المهمة الغزيرة عبر التمثيل المشترك).
- تحتاج **حجم نموذج** أصغر في الإنتاج (شبكة واحدة).

**متى يكون أسوأ؟**
- المهام **متنافسة** أو غير مرتبطة؛ عندها يتضارب التدرّجان.
- كل مهمة تحتاج **سعة** مختلفة كثيرًا.

> **نصيحة:** وزّع الخسائر بحيث تعكس تكرار المهام. المهمة النادرة تحتاج وزنًا أكبر عادةً، لا أصغر.

## 17.4 خمس أدوات لعدم التوازن

| الأداة | المبدأ | متى تختارها | الكلفة |
|---|---|---|---|
| `class_weight` | وزن في الخسارة | أسرع حل، فعّال عادةً | قد يخفض الدقّة العامة |
| `sample_weight` | وزن لكل عيّنة | جودة متفاوتة للعيّنات | تحتاج حساب الأوزان |
| `Data augmentation` | توليد تنوّع | أمثلة نادرة قليلة | تحتاج عمليات مناسبة |
| `Focal loss` | تركيز على الصعب | تشبّه extreme imbalance | تتطلّب `BinaryFocalCrossentropy` |
| `Resampling` | أكثر/أقلّ عيّنات | تحكّم كامل | خطر تسرّب أو تشويه التوزيع |

### Focal Loss: لماذا يعمل؟

الخسارة المتقاطعة العادية تعاقب التنبّؤ **السهل** بنفس القدر الذي تعاقب به التنبّؤ **الصعب**:

$$
\mathcal{L}_{CE} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

حيث $p_t$ احتمال الصنف الحقيقي، و$\gamma \ge 0$ «معامل التركيز». عند $\gamma = 2$ تُقلَّص خسارة الأمثلة السهلة (منخفضة $p_t$ بعد التصحيح) إلى الربع تقريبًا، بينما تبقى خسارة الأمثلة الصعبة شبه كما هي. النتيجة: الشبكة **تتركّز** على النادرة.

Keras 3 يوفّرها جاهزة: `keras.losses.BinaryFocalCrossentropy` و`keras.losses.CategoricalFocalCrossentropy` (مع `alpha` و`gamma`).

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch17_multi_task_weights.md
# ────────────────────────────────────────────────────────────────────────
> **ملاحظة:** اختيار العتبة قرار **تشغيلي** لا رياضي: حدّد من تحلّة الفريق «كم تنبيهًا خاطئًا يمكن تحمّله أسبوعيًا؟» ثم اختر العتبة التي تحقّق ذلك. أداة `keras.metrics.RecallAtPrecision` تساعد في البحث الآلي.

**الخطوة 5 — راقب المقاييس الصحيحة.** مع 2% إيجابيات، لا يصحّ مراقبة `accuracy`. راقب `val_auc` و`val_recall` مع `mode="max"`.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : `weighted_metrics`: قياس مرجّح
# المصدر  : src/chapters/ch17_multi_task_weights.md:57
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الأول: مدخلات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:106
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# مدخلات متعدّدة: صورة + متجه خصائص
img_input = keras.Input(shape=(32, 32, 3), name="image")
feat_input = keras.Input(shape=(10,), name="features")

x = keras.layers.Conv2D(32, 3, activation="relu")(img_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)

y = keras.layers.Dense(32, activation="relu")(feat_input)

merged = keras.layers.Concatenate()([x, y])
out = keras.layers.Dense(4, activation="softmax")(merged)

model = keras.Model(inputs=[img_input, feat_input], outputs=out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # التسميات أعداد صحيحة
    metrics=["accuracy"],
)
model.summary()

# بيانات تجريبية
imgs = np.random.rand(64, 32, 32, 3).astype("float32")
feats = np.random.rand(64, 10).astype("float32")
labels = np.random.randint(0, 4, size=64)

history = model.fit(
    {"image": imgs, "features": feats},
    labels,
    epochs=3,
    batch_size=8,
    verbose=0,
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثاني: مخرجات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:153
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مخرجان: تصنيف العيّنة + انحدار العيب
class_out = keras.Input(shape=(1,), name="y_class")
reg_out = keras.Input(shape=(1,), name="y_reg")

inp = keras.Input(shape=(20,), name="features")
h = keras.layers.Dense(64, activation="relu")(inp)
h = keras.layers.Dense(32, activation="relu")(h)

pred_class = keras.layers.Dense(3, activation="softmax", name="class")(h)
pred_reg = keras.layers.Dense(1, name="reg")(h)

multi = keras.Model(inputs=inp, outputs=[pred_class, pred_reg])
multi.compile(
    optimizer="adam",
    loss={
        "class": "sparse_categorical_crossentropy",
        "reg": "mse",
    },
    loss_weights={
        "class": 1.0,
        "reg": 0.1,     # وازن بين المخرجين
    },
    metrics={
        "class": ["accuracy"],
        "reg": ["mae"],
    },
)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثالث: تعلّم متعدّد المهام (Multi-task)
# المصدر  : src/chapters/ch17_multi_task_weights.md:192
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(20,))
shared = keras.layers.Dense(64, activation="relu")(inp)
shared = keras.layers.BatchNormalization()(shared)
shared = keras.layers.Dense(32, activation="relu")(shared)

out_a = keras.layers.Dense(1, activation="sigmoid", name="task_a")(shared)
out_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

mtl = keras.Model(inputs=inp, outputs=[out_a, out_b])
mtl.compile(
    optimizer="adam",
    loss={"task_a": "binary_crossentropy", "task_b": "binary_crossentropy"},
    loss_weights={"task_a": 1.0, "task_b": 0.5},
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : Focal Loss: لماذا يعمل؟
# المصدر  : src/chapters/ch17_multi_task_weights.md:244
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras

loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=False,
    alpha=0.25,     # وزن الصنف النادر
    gamma=2.0,      # معامل التركيز
)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:261
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

counts = np.array([98000, 2000])           # 98% / 2%
n = counts.sum()
weights = n / (len(counts) * counts)       # [0.510, 25.0]
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print(class_weight)   # النتيجة المتوقعة: {0: 0.5102, 1: 25.0}

model.fit(x_train, y_train, class_weight=class_weight, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:278
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=True,
    alpha=0.75,     # أنزّل وزن الصنف الشائع
    gamma=2.0,
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss,
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC(),
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:300
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: جرّب عتبات مختلفة على الاحتمالات
probs = model.predict(x_val).ravel()
y_true = y_val.ravel()
for thr in (0.3, 0.5, 0.7):
    pred = (probs >= thr).astype("float32")
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"thr={thr}: P={precision:.3f} R={recall:.3f}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:318
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=5,
                                   restore_best_weights=True),
]

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
> keras.layers.Dense(C)   # بلا activation
> loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
> ```

## 17.3 المهام المتعدّدة: ثلاثة أنماط

### النمط الأول: مدخلات متعدّدة

When كل عيّنة تحمل أكثر من نوع من الخصائص. مثال: تصنيف منتج من (صورة + وصف نصّي + السعر).

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
**متى يتفوّق على نموذجين منفصلين؟**
- المهام **مرتبطة** (تشترك في بنية تمثيل مفيدة).
- البيانات **محدودة** (المهمة النادرة تستفيد من بيانات المهمة الغزيرة عبر التمثيل المشترك).
- تحتاج **حجم نموذج** أصغر في الإنتاج (شبكة واحدة).

**متى يكون أسوأ؟**
- المهام **متنافسة** أو غير مرتبطة؛ عندها يتضارب التدرّجان.
- كل مهمة تحتاج **سعة** مختلفة كثيرًا.

> **نصيحة:** وزّع الخسائر بحيث تعكس تكرار المهام. المهمة النادرة تحتاج وزنًا أكبر عادةً، لا أصغر.

## 17.4 خمس أدوات لعدم التوازن

| الأداة | المبدأ | متى تختارها | الكلفة |
|---|---|---|---|
| `class_weight` | وزن في الخسارة | أسرع حل، فعّال عادةً | قد يخفض الدقّة العامة |
| `sample_weight` | وزن لكل عيّنة | جودة متفاوتة للعيّنات | تحتاج حساب الأوزان |
| `Data augmentation` | توليد تنوّع | أمثلة نادرة قليلة | تحتاج عمليات مناسبة |
| `Focal loss` | تركيز على الصعب | تشبّه extreme imbalance | تتطلّب `BinaryFocalCrossentropy` |
| `Resampling` | أكثر/أقلّ عيّنات | تحكّم كامل | خطر تسرّب أو تشويه التوزيع |

### Focal Loss: لماذا يعمل؟

الخسارة المتقاطعة العادية تعاقب التنبّؤ **السهل** بنفس القدر الذي تعاقب به التنبّؤ **الصعب**:

$$
\mathcal{L}_{CE} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)
$$

حيث $p_t$ احتمال الصنف الحقيقي، و$\gamma \ge 0$ «معامل التركيز». عند $\gamma = 2$ تُقلَّص خسارة الأمثلة السهلة (منخفضة $p_t$ بعد التصحيح) إلى الربع تقريبًا، بينما تبقى خسارة الأمثلة الصعبة شبه كما هي. النتيجة: الشبكة **تتركّز** على النادرة.

Keras 3 يوفّرها جاهزة: `keras.losses.BinaryFocalCrossentropy` و`keras.losses.CategoricalFocalCrossentropy` (مع `alpha` و`gamma`).

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-17-ch17-multi-task-weights.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
> **ملاحظة:** اختيار العتبة قرار **تشغيلي** لا رياضي: حدّد من تحلّة الفريق «كم تنبيهًا خاطئًا يمكن تحمّله أسبوعيًا؟» ثم اختر العتبة التي تحقّق ذلك. أداة `keras.metrics.RecallAtPrecision` تساعد في البحث الآلي.

**الخطوة 5 — راقب المقاييس الصحيحة.** مع 2% إيجابيات، لا يصحّ مراقبة `accuracy`. راقب `val_auc` و`val_recall` مع `mode="max"`.
