# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : التحقّق اليدوي من عملية الالتفاف
# المصدر  : src/chapters/ch18_convolution_math.md:229
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0
