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
