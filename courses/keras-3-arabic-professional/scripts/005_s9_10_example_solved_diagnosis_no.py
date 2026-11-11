# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : 9.10 مثال محلول: تشخيص تدرّب لا يتقارب
# المصدر  : src/chapters/ch09_optimizers.md:210
# الترتيب : 5/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.random.rand(512, 20).astype("float32")
y = (np.sin(x[:, 0] * 6) > 0).astype("float32")

def make_model():
    return keras.Sequential([
        keras.layers.Input((20,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

configs = [
    ("Adam lr=1e-2", keras.optimizers.Adam(learning_rate=1e-2)),
    ("Adam lr=1e-3", keras.optimizers.Adam(learning_rate=1e-3)),
    ("SGD  lr=0.1",  keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("SGD  lr=0.01", keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
]

for name, opt in configs:
    keras.utils.set_random_seed(0)
    m = make_model()
    m.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    h = m.fit(x, y, epochs=30, batch_size=32, verbose=0)
    print(f"{name:14} | final loss={h.history['loss'][-1]:.4f} "
          f"| acc={h.history['accuracy'][-1]:.4f}")
