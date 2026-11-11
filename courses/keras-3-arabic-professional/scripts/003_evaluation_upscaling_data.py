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
