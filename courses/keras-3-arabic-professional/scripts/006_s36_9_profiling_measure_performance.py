# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.9 قياس الأداء (Profiling)
# المصدر  : src/chapters/ch36_acceleration.md:200
# الترتيب : 6/6
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.Input(shape=(16,)),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(1000, 16).astype("float32")
y = np.random.rand(1000, 1).astype("float32")

# ─── 1. قياس زمن الدورة
t0 = time.perf_counter()
model.fit(x, y, epochs=1, batch_size=32, verbose=0)
epoch_time = time.perf_counter() - t0
print(f"epoch time = {epoch_time:.3f}s")

# ─── 2. قياس زمن الاستدلال
t0 = time.perf_counter()
_ = model.predict(x[:100], verbose=0)
infer_time = time.perf_counter() - t0
print(f"inference 100 = {infer_time * 1000:.1f}ms")
# النتيجة المتوقعة: أرقام فعلية تُقارن قبل وبعد كلّ تحسين.
