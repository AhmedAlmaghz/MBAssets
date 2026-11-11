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
