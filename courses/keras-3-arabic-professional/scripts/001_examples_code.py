# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:129
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994
