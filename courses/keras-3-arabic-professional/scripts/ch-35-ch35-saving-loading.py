#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الحفظ والتحميل
Keras 3 in Arabic — Saving Loading

Book slug : keras-3-arabic-professional
Chapter   : ch-35-ch35-saving-loading (lesson 35)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-35-ch35-saving-loading.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الحفظ والتحميل
Keras 3 in Arabic — Saving Loading

Book slug : keras-3-arabic-professional
Chapter   : ch-35-ch35-saving-loading (lesson 35)
Code files: 19 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-35-ch35-saving-loading.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الحفظ والتحميل
Keras 3 in Arabic — Saving Loading

Book slug : keras-3-arabic-professional
Chapter   : ch-35-ch35-saving-loading (lesson 35)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch35_saving_loading/001_s35_2_saving_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.2 الحفظ الكامل
# المصدر  : src/chapters/ch35_saving_loading.md:33
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch35_saving_loading/002_s35_3_saving_optimizer.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.3 الحفظ بلا حالة المحسّن
# المصدر  : src/chapters/ch35_saving_loading.md:64
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch35_saving_loading/003_s35_4_modelcheckpoint_checkpoint_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.4 نقطة الحفظ (ModelCheckpoint)
# المصدر  : src/chapters/ch35_saving_loading.md:84
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch35_saving_loading/004_s35_6_layers_custom_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.6 الطبقات المخصّصة في الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:127
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch35_saving_loading/005_s35_7_resume_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.7 استئناف التدريب
# المصدر  : src/chapters/ch35_saving_loading.md:160
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch35_saving_loading/006_s35_10_safe_mode_safety.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:215
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch35_saving_loading/007_s35_10_safe_mode_safety.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:230
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch35_saving_loading/008_s35_11_regularization_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.11 تنظيم نقاط الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:259
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch35_saving_loading/009_s35_12_example_saving_weights.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.12 مثال: حفظ الأوزان لنقلها بين بنيتين
# المصدر  : src/chapters/ch35_saving_loading.md:291
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.2 الحفظ الكامل
# المصدر  : src/chapters/ch35_saving_loading.md:33
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.3 الحفظ بلا حالة المحسّن
# المصدر  : src/chapters/ch35_saving_loading.md:64
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.4 نقطة الحفظ (ModelCheckpoint)
# المصدر  : src/chapters/ch35_saving_loading.md:84
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.6 الطبقات المخصّصة في الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:127
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.7 استئناف التدريب
# المصدر  : src/chapters/ch35_saving_loading.md:160
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:215
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:230
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.11 تنظيم نقاط الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:259
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.12 مثال: حفظ الأوزان لنقلها بين بنيتين
# المصدر  : src/chapters/ch35_saving_loading.md:291
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الحفظ والتحميل
Keras 3 in Arabic — Saving Loading

Book slug : keras-3-arabic-professional
Chapter   : ch-35-ch35-saving-loading (lesson 35)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch35_saving_loading/001_s35_2_saving_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.2 الحفظ الكامل
# المصدر  : src/chapters/ch35_saving_loading.md:33
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch35_saving_loading/002_s35_3_saving_optimizer.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.3 الحفظ بلا حالة المحسّن
# المصدر  : src/chapters/ch35_saving_loading.md:64
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch35_saving_loading/003_s35_4_modelcheckpoint_checkpoint_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.4 نقطة الحفظ (ModelCheckpoint)
# المصدر  : src/chapters/ch35_saving_loading.md:84
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch35_saving_loading/004_s35_6_layers_custom_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.6 الطبقات المخصّصة في الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:127
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch35_saving_loading/005_s35_7_resume_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.7 استئناف التدريب
# المصدر  : src/chapters/ch35_saving_loading.md:160
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch35_saving_loading/006_s35_10_safe_mode_safety.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:215
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch35_saving_loading/007_s35_10_safe_mode_safety.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:230
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch35_saving_loading/008_s35_11_regularization_saving.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.11 تنظيم نقاط الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:259
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch35_saving_loading/009_s35_12_example_saving_weights.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.12 مثال: حفظ الأوزان لنقلها بين بنيتين
# المصدر  : src/chapters/ch35_saving_loading.md:291
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch35_saving_loading.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.2 الحفظ الكامل
# المصدر  : src/chapters/ch35_saving_loading.md:33
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.3 الحفظ بلا حالة المحسّن
# المصدر  : src/chapters/ch35_saving_loading.md:64
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.4 نقطة الحفظ (ModelCheckpoint)
# المصدر  : src/chapters/ch35_saving_loading.md:84
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.6 الطبقات المخصّصة في الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:127
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.7 استئناف التدريب
# المصدر  : src/chapters/ch35_saving_loading.md:160
# الترتيب : 5/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:215
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:230
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.11 تنظيم نقاط الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:259
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.12 مثال: حفظ الأوزان لنقلها بين بنيتين
# المصدر  : src/chapters/ch35_saving_loading.md:291
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بناء نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# ─── 2. تدريب سريع
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, verbose=0)

# ─── 3. حفظ كامل
model.save("model_full.keras")

# ─── 4. تحميل
loaded = keras.saving.load_model("model_full.keras")
# النتيجة المتوقعة: نموذج مُحمَّل، جاهز للاستدلال أو الاستئناف.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ عند تحسّن خسارة التحقّق
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",        # المسار
    monitor="val_loss",                 # المقياس المُراقَب
    save_best_only=True,                # حفظ الأفضل فقط
    verbose=1,
)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=5, batch_size=32,
          validation_split=0.2, callbacks=[checkpoint], verbose=0)
# النتيجة المتوقعة: ملفّ best_model.keras فيه الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# نموذج مدَّبَب مُسبقًا (محاكاة)
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=2, verbose=0)

# حفظ
model.save("checkpoint.keras")

# استئناف
resumed = keras.saving.load_model("checkpoint.keras")
resumed.fit(x, y, epochs=2, verbose=0)   # متابعة
# النتيجة المتوقعة: خسارة تنخفض أكثر مقارنة بالنموذج المُحمَّل.

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras

# التحميل الآمن (الافتراضيّ)
m = keras.saving.load_model("model_full.keras", safe_mode=True)

# إن كان الملفّ يُنشيئ كائنات مخصّصة (مثل Lambda)، يجب تعطيل الوضع:
# m = keras.saving.load_model("m.keras", safe_mode=False)
# النتيجة المتوقعة: التحميل الآمن ينجح بلا تنفيذ أيّ كود.

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

# نقطة حفظ دوريّة (كلّ دورة، بمُعرّف رقميّ)
periodic = keras.callbacks.ModelCheckpoint(
    filepath="ckpts/epoch_{epoch:03d}.keras",
    save_best_only=False, verbose=0)
# نقطة حفظ للأفضل
best = keras.callbacks.ModelCheckpoint(
    filepath="best.keras", monitor="val_loss",
    save_best_only=True, verbose=0)

x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, validation_split=0.2,
          callbacks=[periodic, best], verbose=0)
# النتيجة المتوقعة: ملفّات مرقّمة + ملفّ الأفضل.

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-35-ch35-saving-loading.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج مصدر
src = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
src.compile(optimizer="adam", loss="mse")
src.save_weights("src.weights.h5")

# ─── 2. نموذج هدف بنفس البنية (بنية جديدة في الشيفرة)
dst = keras.Sequential([keras.Input(shape=(16,)),
                        keras.layers.Dense(32, activation="relu"),
                        keras.layers.Dense(1)])
dst.load_weights("src.weights.h5")

# ─── 3. تحقّق: الأوزان متطابقة
w_src = src.get_weights()
w_dst = dst.get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w_src, w_dst))
# النتيجة المتوقعة: الأوزان منقولة بالكامل.
