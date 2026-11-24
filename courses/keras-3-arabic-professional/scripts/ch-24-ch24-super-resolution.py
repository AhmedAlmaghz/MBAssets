#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تكبير الصور (Super-Resolution)
Keras 3 in Arabic — Super Resolution

Book slug : keras-3-arabic-professional
Chapter   : ch-24-ch24-super-resolution (lesson 24)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-24-ch24-super-resolution.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تكبير الصور (Super-Resolution)
Keras 3 in Arabic — Super Resolution

Book slug : keras-3-arabic-professional
Chapter   : ch-24-ch24-super-resolution (lesson 24)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-24-ch24-super-resolution.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تكبير الصور (Super-Resolution)
Keras 3 in Arabic — Super Resolution

Book slug : keras-3-arabic-professional
Chapter   : ch-24-ch24-super-resolution (lesson 24)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch24_super_resolution/001_s24_5_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.5 التنفيذ في Keras
# المصدر  : src/chapters/ch24_super_resolution.md:60
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch24_super_resolution/002_psnr_ssim_measure.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : قياس PSNR و SSIM عمليًّا
# المصدر  : src/chapters/ch24_super_resolution.md:144
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch24_super_resolution/003_evaluation_upscaling_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : تقييم ثلاث طرق تكبير على نفس البيانات
# المصدر  : src/chapters/ch24_super_resolution.md:175
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch24_super_resolution/004_s24_9_measure_upscaling_energy_frequency.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.9 قياس التكبير عبر طاقة التردّد العالي
# المصدر  : src/chapters/ch24_super_resolution.md:215
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.5 التنفيذ في Keras
# المصدر  : src/chapters/ch24_super_resolution.md:60
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : قياس PSNR و SSIM عمليًّا
# المصدر  : src/chapters/ch24_super_resolution.md:144
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : تقييم ثلاث طرق تكبير على نفس البيانات
# المصدر  : src/chapters/ch24_super_resolution.md:175
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.9 قياس التكبير عبر طاقة التردّد العالي
# المصدر  : src/chapters/ch24_super_resolution.md:215
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — تكبير الصور (Super-Resolution)
Keras 3 in Arabic — Super Resolution

Book slug : keras-3-arabic-professional
Chapter   : ch-24-ch24-super-resolution (lesson 24)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch24_super_resolution/001_s24_5_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.5 التنفيذ في Keras
# المصدر  : src/chapters/ch24_super_resolution.md:60
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch24_super_resolution/002_psnr_ssim_measure.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : قياس PSNR و SSIM عمليًّا
# المصدر  : src/chapters/ch24_super_resolution.md:144
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch24_super_resolution/003_evaluation_upscaling_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : تقييم ثلاث طرق تكبير على نفس البيانات
# المصدر  : src/chapters/ch24_super_resolution.md:175
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch24_super_resolution/004_s24_9_measure_upscaling_energy_frequency.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.9 قياس التكبير عبر طاقة التردّد العالي
# المصدر  : src/chapters/ch24_super_resolution.md:215
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch24_super_resolution.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.5 التنفيذ في Keras
# المصدر  : src/chapters/ch24_super_resolution.md:60
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : قياس PSNR و SSIM عمليًّا
# المصدر  : src/chapters/ch24_super_resolution.md:144
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : تقييم ثلاث طرق تكبير على نفس البيانات
# المصدر  : src/chapters/ch24_super_resolution.md:175
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 24: تكبير الصور (Super-Resolution)
# القسم   : 24.9 قياس التكبير عبر طاقة التردّد العالي
# المصدر  : src/chapters/ch24_super_resolution.md:215
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 400 زوج (منخفضة، عالية)
n = 400
x_high = np.random.rand(n, 64, 64, 3).astype("float32") * 255.0
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))  # 32×32

# ─── 2. شبكة تكبير بسيطة: استخراج → تمثيل → تكبير
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(12, 3, padding="same")(x)      # 3 قنوات × 4 (2²)
# التوسّع عبر تجميع القنوات في الفضاء
out = keras.ops.depth_to_space(x, 2)                    # 64×64×3
out = keras.layers.Activation("sigmoid")(out)           # في [0, 1]

model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_absolute_error",         # L1 بدل L2: حواف أقلّ ضبابية
    metrics=["mse"],
)
model.fit(x_low, np.array(keras.ops.image.resize(
    keras.ops.cast(x_high, "float32") / 255.0, (64, 64))),
    epochs=10, batch_size=16, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: MAE ≈ 0.15 بعد 10 دورات على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. صورة أصلية ونُسخة مُعادة البناء (محاكاة)
x_true = np.random.rand(8, 64, 64, 3).astype("float32")      # أصلية
x_pred = x_true + np.random.randn(8, 64, 64, 3) * 0.05       # مع ضجيج
x_pred = np.clip(x_pred, 0.0, 1.0).astype("float32")

# ─── 2. PSNR يدويًّا
mse = float(np.mean((x_true - x_pred) ** 2))
psnr = 10.0 * np.log10(1.0 / mse)
print(f"PSNR = {psnr:.2f} dB")

# ─── 3. نفس الحساب بـkeras.ops (ينتقل مع المحرّك)
mse_k = keras.ops.mean(keras.ops.square(x_true - x_pred))
psnr_k = float(10.0 * keras.ops.log10(1.0 / mse_k))
print(f"PSNR (keras.ops) = {psnr_k:.2f} dB")

# ─── 4. SSIM عبر Keras ops
ssim = float(np.array(keras.ops.image.ssim(x_true, x_pred, max_val=1.0))[0])
print(f"SSIM = {ssim:.4f}")
# النتيجة المتوقعة: PSNR ≈ 26–30 dB و SSIM ≈ 0.85–0.95.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات: صور أصلية ونُسخ منخفضة الدقّة
n = 60
x_high = np.random.rand(n, 64, 64, 3).astype("float32")
x_low = np.array(keras.ops.image.resize(x_high, (32, 32)))
x_up = np.array(keras.ops.image.resize(x_low, (64, 64)))   # استيفاء

# ─── 2. شبكة تكبير مصغّرة
inp = keras.Input(shape=(32, 32, 3))
y = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
y = keras.layers.Conv2D(12, 3, padding="same")(y)
out = keras.ops.depth_to_space(y, 2)           # 64×64×3
sr = keras.Model(inp, out)
sr.compile(optimizer="adam", loss="mean_absolute_error")
sr.fit(x_low, x_high, epochs=8, batch_size=16, verbose=0)

# ─── 3. تقييم الطريقتين
def psnr_avg(a, b):
    # PSNR = 10·log10(max² / MSE) — لا توجد keras.ops.image.psnr في Keras 3
    a = keras.ops.convert_to_numpy(a)
    b = keras.ops.convert_to_numpy(b)
    mse = np.mean(np.square(a.astype('float64') - b.astype('float64')))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))

print(f"interpolation PSNR = {psnr_avg(x_up, x_high):.2f} dB")
print(f"CNN        PSNR = {psnr_avg(sr.predict(x_low, verbose=0), x_high):.2f} dB")
# النتيجة المتوقعة: CNN أعلى من الاستيفاء بعد التدريب.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-24-ch24-super-resolution.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

def high_freq_energy(img):
    """طاقة التردّد العاليّ: مجموع مربّعات الفروق الأفقية والرأسيّة."""
    dx = img[:, 1:, :, :] - img[:, :-1, :, :]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return float(np.mean(dx ** 2) + np.mean(dy ** 2))

# ─── 1. ثلاث نسخ: أصليّة، استيفاء، شبكة تكبير
x_true = np.random.rand(10, 64, 64, 3).astype("float32")
x_up = np.array(keras.ops.image.resize(
    keras.ops.image.resize(x_true, (32, 32)), (64, 64)))   # استيفاء
x_sr = np.clip(x_up + np.random.randn(*x_up.shape) * 0.03, 0, 1)

# ─── 2. القياس
print(f"أصلية       : {high_freq_energy(x_true):.5f}")
print(f"استيفاء     : {high_freq_energy(x_up):.5f}")
print(f"شبكة تكبير : {high_freq_energy(x_sr):.5f}")
# النتيجة المتوقعة: الأصلية أعلى طاقة؛ الاستيفاء أدنى (تفاصيل مفقودة)؛
# والنموذج الجيّد يقترب من الأصلية.
