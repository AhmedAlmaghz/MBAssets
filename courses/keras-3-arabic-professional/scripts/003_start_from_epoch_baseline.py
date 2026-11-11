# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : `start_from_epoch` و`baseline`
# المصدر  : src/chapters/ch12_callbacks.md:147
# الترتيب : 3/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة تجعل النتيجة قابلة للتكرار حرفيًّا
keras.utils.set_random_seed(0)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse",
    metrics=["mae"],
)

# بيانات لها حدّأدنى للخطأ (ضوضاء غير قابلة للاختزال): النموذج سيبلغها ثم يتذبذب
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",      # ما الذي نراقبه
    patience=2,              # كم حقبة بلا تحسّن نتحمّلها
    min_delta=1e-4,          # أقل تحسّن يُعدّ تحسّنًا
    mode="min",              # تصريحٌ بالاتجاه: لا تعتمد على الاستنتاج
    restore_best_weights=True,  # أعد أفضل الأوزان لا آخرها
    start_from_epoch=2,      # تجاهل الحقبتين الأولى
    verbose=1,
)

history = model.fit(
    x, y,
    epochs=12,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0,
)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11
# النتيجة المتوقعة (رسالة Keras): Epoch 11: early stopping
# النتيجة المتوقعة (رسالة Keras): Restoring model weights from the end of the best epoch: 9.
