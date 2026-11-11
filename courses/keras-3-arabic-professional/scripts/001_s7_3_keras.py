# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:142
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# الشبكة نفسها بتغيير واحد فقط: وجود أو غياب nonlinearity
model = keras.Sequential([
    keras.Input(shape=(8,)),
    keras.layers.Dense(64, activation="relu"),   # تنشيط مدمج في الطبقة
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),                        # طبقة إخراج خام (logits)
])

model.summary()
