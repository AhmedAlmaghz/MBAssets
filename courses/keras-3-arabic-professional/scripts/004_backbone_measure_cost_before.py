# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : قياس كلفة backbone قبل الالتزام
# المصدر  : src/chapters/ch30_kerashub.md:132
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.
