# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : فكّ التجميد المتدرّج
# المصدر  : src/chapters/ch30_kerashub.md:172
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.
