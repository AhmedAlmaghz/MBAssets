# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:161
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras

# تفعيل مستقل: نفس الناتج لكن أوضح في القراءة
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64),            # طبقة خطية صافية
    keras.layers.Activation("relu"),   # ثم التنشيط كطبقة مستقلة
    keras.layers.Dense(32),
    keras.layers.Activation("gelu"),
    keras.layers.Dense(1),
])
print(model.output_shape)  # النتيجة المتوقعة: (None, 1)
