# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : مثال تطبيقي كامل: اكتشاف فرط التكيّف (overfitting
# المصدر  : src/chapters/ch13_regularization.md:126
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── بيانات اصطناعية: تصنيف ثنائي بحدّ خطّي + ضجيج
rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 20)).astype("float32")
y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype("float32")

x_train, x_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]


def build_model(reg_strength):
    """نفس البنية، مع تدرّج منتظم متزايد على الطبقات المخفية."""
    reg = keras.regularizers.L2(reg_strength)
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── النموذجان: بلا تنظيم وبـL2
plain = build_model(0.0)
regularized = build_model(0.01)

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

for name, m in (("plain", plain), ("regularized", regularized)):
    hist = m.fit(x_train, y_train, validation_data=(x_val, y_val),
                 epochs=60, batch_size=64, verbose=0, callbacks=cb)
    tr, va = hist.history["loss"][-1], hist.history["val_loss"][-1]
    print(f"{name:12s} train={tr:.4f}  val={va:.4f}  gap={va - tr:+.4f}")
# النتيجة المتوقعة: فجوة النموذج المنتظم أصغر بوضوح من غيره
