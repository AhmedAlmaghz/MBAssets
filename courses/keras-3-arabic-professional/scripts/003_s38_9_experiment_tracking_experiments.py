# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.9 تتبّع التجارب (Experiment Tracking)
# المصدر  : src/chapters/ch38_reproducibility.md:175
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import json
import time

keras.utils.set_random_seed(42)

# ─── 1. بناء وتدريب نموذج
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(32, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
h = model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. سجلّ التجربة
record = {
    "run_id": f"run_{int(time.time())}",
    "seed": 42,
    "epochs": 5,
    "batch_size": 32,
    "final_loss": float(h.history["loss"][-1]),
}
with open("runs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
# النتيجة المتوقعة: سطر واحد في runs.jsonl يصف التجربة كاملة.
