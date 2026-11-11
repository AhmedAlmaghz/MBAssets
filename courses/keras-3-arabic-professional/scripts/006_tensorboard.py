# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `TensorBoard`: كيف تقرأ ما سجّله
# المصدر  : src/chapters/ch12_callbacks.md:358
# الترتيب : 6/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

tb = keras.callbacks.TensorBoard(
    log_dir="./logs",           # مجلد السجلّات
    write_graph=True,          # ارسم رسم بياني للبنية
    histogram_freq=1,          # سجّل توزيع أوزان كل حقبة
    embeddings_freq=0,         # لا حاجة لتضمينات في هذا المثال
    update_freq="epoch",       # حدّث السجلّات في نهاية كل حقبة
)

model.fit(x, y, epochs=5, batch_size=16, validation_split=0.2,
          callbacks=[tb], verbose=0)
print("اكتمل الكتابة في ./logs")
