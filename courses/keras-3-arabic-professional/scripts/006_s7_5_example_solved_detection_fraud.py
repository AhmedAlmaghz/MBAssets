# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.5 مثال محلول: بوابة كشف الاحتيال
# المصدر  : src/chapters/ch07_activations.md:300
# الترتيب : 6/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import ops

# بيانات اصطناعية: 3 فئات،Feature 4
rng = np.random.default_rng(0)
n = 1500
features = rng.normal(size=(n, 4)).astype("float32")
labels = rng.integers(0, 3, size=n)          # 0=عادي، 1=مشبوه، 2=احتيال
labels = labels.astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(32, activation="gelu"),
    keras.layers.Dense(3),                     # logits خام
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(features, labels, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

logits = model.predict(features[:5], verbose=0)     # شكل (5, 3) من logits
probs = ops.softmax(logits, axis=-1)                # تحويل صريح إلى احتمالات
print(probs)
print("النتيجة المتوقعة: كل صف يجمع إلى 1.0، والصف 0 يعطي argmax=0")
