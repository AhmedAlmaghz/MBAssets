# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : مثال أول: تتبع بسيط
# المصدر  : src/chapters/ch12_callbacks.md:65
# الترتيب : 2/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# نجمع سجلًا لكل دفعة، لنرى متى تُستدعى الخطّافات فعليًّا
class BatchLogger(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, batch, logs=None):
        # logs هنا يحمل المقاييس المجمَّعة حتى الدفعة الحالية
        self.records.append((batch, float(logs["loss"])))
        if batch == 0:
            print("دفعة 0: الخسارة =", float(logs["loss"]))


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

logger = BatchLogger()
history = model.fit(x, y, epochs=2, batch_size=16, callbacks=[logger], verbose=0)

print("عدد الدفعات المسجّلة =", len(logger.records))
print("مفاتيح السجل =", sorted(history.history.keys()))
# النتيجة المتوقعة: عدد الدفعات المسجّلة = 8
# النتيجة المتوقعة: مفاتيح السجل = ['loss']
