# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `LearningRateScheduler`: الجدولة من داخل الحلقة
# المصدر  : src/chapters/ch12_callbacks.md:301
# الترتيب : 5/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

x = np.random.default_rng(0).standard_normal((64, 8)).astype("float32")
y = (x[:, 0] * 2.0).astype("float32")

# معدّل التعلّم الأساسي: نحتفظ به في ثابت لأن الـcallback يعيد الحساب كل حقبة
BASE_LR = 1e-3
WARMUP_EPOCHS = 5


def warmup_schedule(epoch, lr):
    """إحماء خطّي تصاعدي في أول 5 حقبات ثم تخفيض أُسّي بمعامل 0.7."""
    if epoch < WARMUP_EPOCHS:
        # تصاعد خطّي من base_lr/WARMUP_EPOCHS حتى base_lr
        return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    # بعد الإحماء: نضرب في كل حقبة فتتراكم بنسبة 0.7^epoch
    return lr * 0.7


scheduler = keras.callbacks.LearningRateScheduler(warmup_schedule, verbose=1)
history = model.fit(x, y, epochs=10, batch_size=16,
                    callbacks=[scheduler], verbose=0)
print("سجل معدّل التعلّم =", [round(v, 7) for v in history.history["learning_rate"]])
# النتيجة المتوقعة:
# [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0007, 0.00049, 0.000343, 0.0002401, 0.0001681]
