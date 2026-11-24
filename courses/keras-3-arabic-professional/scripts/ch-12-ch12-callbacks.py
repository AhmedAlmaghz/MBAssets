#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الـ Callbacks: أتمتة دورة التدريب
Keras 3 in Arabic — Callbacks

Book slug : keras-3-arabic-professional
Chapter   : ch-12-ch12-callbacks (lesson 12)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-12-ch12-callbacks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الـ Callbacks: أتمتة دورة التدريب
Keras 3 in Arabic — Callbacks

Book slug : keras-3-arabic-professional
Chapter   : ch-12-ch12-callbacks (lesson 12)
Code files: 21 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-12-ch12-callbacks.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الـ Callbacks: أتمتة دورة التدريب
Keras 3 in Arabic — Callbacks

Book slug : keras-3-arabic-professional
Chapter   : ch-12-ch12-callbacks (lesson 12)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch12_callbacks/002_example_tracking.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : مثال أول: تتبع بسيط
# المصدر  : src/chapters/ch12_callbacks.md:65
# الترتيب : 2/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch12_callbacks/003_start_from_epoch_baseline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `start_from_epoch` و`baseline`
# المصدر  : src/chapters/ch12_callbacks.md:147
# الترتيب : 3/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch12_callbacks/004_keras_weights_h5_versus.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `.keras` مقابل `.weights.h5`: فرق جوهري
# المصدر  : src/chapters/ch12_callbacks.md:218
# الترتيب : 4/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch12_callbacks/005_learningratescheduler_scheduling_inside.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `LearningRateScheduler`: الجدولة من داخل الحلقة
# المصدر  : src/chapters/ch12_callbacks.md:301
# الترتيب : 5/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch12_callbacks/006_tensorboard.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `TensorBoard`: كيف تقرأ ما سجّله
# المصدر  : src/chapters/ch12_callbacks.md:358
# الترتيب : 6/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch12_callbacks/007_s12_5_callback.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.5 كتابة `Callback` مخصّص
# المصدر  : src/chapters/ch12_callbacks.md:393
# الترتيب : 7/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch12_callbacks/008_callback_scheduling_rate_learning.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : جدولة معدّل التعلّم من داخل الـcallback
# المصدر  : src/chapters/ch12_callbacks.md:471
# الترتيب : 8/11
# ──────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch12_callbacks/009_pipeline_production_training_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : تطبيق واقعي: خطّ إنتاج تدريب لا يُترك للمصادفة
# المصدر  : src/chapters/ch12_callbacks.md:500
# الترتيب : 9/11
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch12_callbacks/010_s12_7_exercises.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:546
# الترتيب : 10/11
# ──────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch12_callbacks/011_s12_7_exercises.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:562
# الترتيب : 11/11
# ──────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : مثال أول: تتبع بسيط
# المصدر  : src/chapters/ch12_callbacks.md:65
# الترتيب : 2/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `start_from_epoch` و`baseline`
# المصدر  : src/chapters/ch12_callbacks.md:147
# الترتيب : 3/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `.keras` مقابل `.weights.h5`: فرق جوهري
# المصدر  : src/chapters/ch12_callbacks.md:218
# الترتيب : 4/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `LearningRateScheduler`: الجدولة من داخل الحلقة
# المصدر  : src/chapters/ch12_callbacks.md:301
# الترتيب : 5/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `TensorBoard`: كيف تقرأ ما سجّله
# المصدر  : src/chapters/ch12_callbacks.md:358
# الترتيب : 6/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.5 كتابة `Callback` مخصّص
# المصدر  : src/chapters/ch12_callbacks.md:393
# الترتيب : 7/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : جدولة معدّل التعلّم من داخل الـcallback
# المصدر  : src/chapters/ch12_callbacks.md:471
# الترتيب : 8/11
# ──────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : تطبيق واقعي: خطّ إنتاج تدريب لا يُترك للمصادفة
# المصدر  : src/chapters/ch12_callbacks.md:500
# الترتيب : 9/11
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:546
# الترتيب : 10/11
# ──────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:562
# الترتيب : 11/11
# ──────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الـ Callbacks: أتمتة دورة التدريب
Keras 3 in Arabic — Callbacks

Book slug : keras-3-arabic-professional
Chapter   : ch-12-ch12-callbacks (lesson 12)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch12_callbacks/002_example_tracking.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : مثال أول: تتبع بسيط
# المصدر  : src/chapters/ch12_callbacks.md:65
# الترتيب : 2/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch12_callbacks/003_start_from_epoch_baseline.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `start_from_epoch` و`baseline`
# المصدر  : src/chapters/ch12_callbacks.md:147
# الترتيب : 3/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch12_callbacks/004_keras_weights_h5_versus.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `.keras` مقابل `.weights.h5`: فرق جوهري
# المصدر  : src/chapters/ch12_callbacks.md:218
# الترتيب : 4/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch12_callbacks/005_learningratescheduler_scheduling_inside.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `LearningRateScheduler`: الجدولة من داخل الحلقة
# المصدر  : src/chapters/ch12_callbacks.md:301
# الترتيب : 5/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch12_callbacks/006_tensorboard.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `TensorBoard`: كيف تقرأ ما سجّله
# المصدر  : src/chapters/ch12_callbacks.md:358
# الترتيب : 6/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch12_callbacks/007_s12_5_callback.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.5 كتابة `Callback` مخصّص
# المصدر  : src/chapters/ch12_callbacks.md:393
# الترتيب : 7/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch12_callbacks/008_callback_scheduling_rate_learning.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : جدولة معدّل التعلّم من داخل الـcallback
# المصدر  : src/chapters/ch12_callbacks.md:471
# الترتيب : 8/11
# ──────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch12_callbacks/009_pipeline_production_training_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : تطبيق واقعي: خطّ إنتاج تدريب لا يُترك للمصادفة
# المصدر  : src/chapters/ch12_callbacks.md:500
# الترتيب : 9/11
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch12_callbacks/010_s12_7_exercises.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:546
# الترتيب : 10/11
# ──────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch12_callbacks/011_s12_7_exercises.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:562
# الترتيب : 11/11
# ──────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch12_callbacks.md
# ────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : مثال أول: تتبع بسيط
# المصدر  : src/chapters/ch12_callbacks.md:65
# الترتيب : 2/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `start_from_epoch` و`baseline`
# المصدر  : src/chapters/ch12_callbacks.md:147
# الترتيب : 3/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `.keras` مقابل `.weights.h5`: فرق جوهري
# المصدر  : src/chapters/ch12_callbacks.md:218
# الترتيب : 4/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `LearningRateScheduler`: الجدولة من داخل الحلقة
# المصدر  : src/chapters/ch12_callbacks.md:301
# الترتيب : 5/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `TensorBoard`: كيف تقرأ ما سجّله
# المصدر  : src/chapters/ch12_callbacks.md:358
# الترتيب : 6/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.5 كتابة `Callback` مخصّص
# المصدر  : src/chapters/ch12_callbacks.md:393
# الترتيب : 7/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : جدولة معدّل التعلّم من داخل الـcallback
# المصدر  : src/chapters/ch12_callbacks.md:471
# الترتيب : 8/11
# ──────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : تطبيق واقعي: خطّ إنتاج تدريب لا يُترك للمصادفة
# المصدر  : src/chapters/ch12_callbacks.md:500
# الترتيب : 9/11
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:546
# الترتيب : 10/11
# ──────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:562
# الترتيب : 11/11
# ──────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
import tempfile
import os

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tmpdir = tempfile.mkdtemp()

# 1) حفظ النموذج كاملًا (أرشيف .keras)
full_path = os.path.join(tmpdir, "model.keras")
ckpt_full = keras.callbacks.ModelCheckpoint(
    filepath=full_path,
    monitor="val_loss",
    save_best_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_full], verbose=0)
print("حجم ملف .keras:", os.path.getsize(full_path), "بايت")
# النتيجة المتوقعة: حجم ملف .keras أكبر من حجم ملف .weights.h5

# 2) حفظ الأوزان فقط (.weights.h5)
weights_path = os.path.join(tmpdir, "model.weights.h5")
ckpt_weights = keras.callbacks.ModelCheckpoint(
    filepath=weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
model.fit(x, y, epochs=3, batch_size=16, validation_split=0.2,
          callbacks=[ckpt_weights], verbose=0)
print("حجم ملف .weights.h5:", os.path.getsize(weights_path), "بايت")
# النتيجة المتوقعة: حجم ملف .weights.h5 أصغر من ملف .keras

# 3) إعادة تحميل النموذج كاملًا بلا إعادة بناء البنية
reloaded = keras.saving.load_model(full_path)
print("أبعاد المخرجات:", reloaded.output_shape)
# النتيجة المتوقعة: أبعاد المخرجات: (None, 1)

# 4) إعادة تحميل الأوزان فقط: تتطلّب بنية مطابقة مسبقًا
fresh = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
fresh.load_weights(weights_path)
print("تم تحميل الأوزان فقط بنجاح")
# النتيجة المتوقعة: تم تحميل الأوزان فقط بنجاح

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# شطر من المثال أعلاه: قراءة معدّل التعلّم الحالي ثم كتابته
current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
self.model.optimizer.learning_rate = new_lr

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-12-ch12-callbacks.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)
