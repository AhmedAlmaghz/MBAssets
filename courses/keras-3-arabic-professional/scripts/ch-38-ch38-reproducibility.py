#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التكرار والاختبار
Keras 3 in Arabic — Reproducibility

Book slug : keras-3-arabic-professional
Chapter   : ch-38-ch38-reproducibility (lesson 38)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-38-ch38-reproducibility.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التكرار والاختبار
Keras 3 in Arabic — Reproducibility

Book slug : keras-3-arabic-professional
Chapter   : ch-38-ch38-reproducibility (lesson 38)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-38-ch38-reproducibility.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التكرار والاختبار
Keras 3 in Arabic — Reproducibility

Book slug : keras-3-arabic-professional
Chapter   : ch-38-ch38-reproducibility (lesson 38)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch38_reproducibility/001_s38_2_keras_seed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.2 البذرة في Keras 3
# المصدر  : src/chapters/ch38_reproducibility.md:30
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch38_reproducibility/002_s38_4_tests_unit.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.4 اختبارات وحدة للنموذج
# المصدر  : src/chapters/ch38_reproducibility.md:78
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch38_reproducibility/003_s38_9_experiment_tracking_experiments.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch38_reproducibility/004_s38_10_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.10 مقارنة النماذج بعدّة بذور
# المصدر  : src/chapters/ch38_reproducibility.md:222
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.2 البذرة في Keras 3
# المصدر  : src/chapters/ch38_reproducibility.md:30
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.4 اختبارات وحدة للنموذج
# المصدر  : src/chapters/ch38_reproducibility.md:78
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.10 مقارنة النماذج بعدّة بذور
# المصدر  : src/chapters/ch38_reproducibility.md:222
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التكرار والاختبار
Keras 3 in Arabic — Reproducibility

Book slug : keras-3-arabic-professional
Chapter   : ch-38-ch38-reproducibility (lesson 38)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch38_reproducibility/001_s38_2_keras_seed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.2 البذرة في Keras 3
# المصدر  : src/chapters/ch38_reproducibility.md:30
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch38_reproducibility/002_s38_4_tests_unit.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.4 اختبارات وحدة للنموذج
# المصدر  : src/chapters/ch38_reproducibility.md:78
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch38_reproducibility/003_s38_9_experiment_tracking_experiments.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch38_reproducibility/004_s38_10_part.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.10 مقارنة النماذج بعدّة بذور
# المصدر  : src/chapters/ch38_reproducibility.md:222
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch38_reproducibility.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.2 البذرة في Keras 3
# المصدر  : src/chapters/ch38_reproducibility.md:30
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.4 اختبارات وحدة للنموذج
# المصدر  : src/chapters/ch38_reproducibility.md:78
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 38: التكرار والاختبار
# القسم   : 38.10 مقارنة النماذج بعدّة بذور
# المصدر  : src/chapters/ch38_reproducibility.md:222
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بناء نموذج
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# تدريب
x = np.random.rand(200, 16).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.fit(x, y, epochs=3, batch_size=32, verbose=0)

# إعادة البناء بنفس البذرة
keras.utils.set_random_seed(42)
model2 = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model2.compile(optimizer="adam", loss="mse")
model2.set_weights(model.get_weights())
# النتيجة المتوقعة: نفس الأوزان — التكرار ناجح.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_model(input_dim=16):
    """نموذج بسيط للاختبار."""
    return keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

def test_model_builds():
    """النموذج يُبنى ويُعطي شكلًا صحيحًا."""
    m = build_model()
    out = m(np.ones((4, 16), dtype="float32"))
    assert out.shape == (4, 1)

def test_model_trains():
    """النموذج يُدرَّب ويُنخفض الخسارة."""
    keras.utils.set_random_seed(42)
    m = build_model()
    m.compile(optimizer="adam", loss="mse")
    x = np.random.rand(100, 16).astype("float32")
    y = np.random.rand(100, 1).astype("float32")
    h = m.fit(x, y, epochs=2, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]

def test_model_predicts_range():
    """المُخرَج في [0, 1]."""
    m = keras.Sequential([
        keras.Input(shape=(16,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    out = m(np.random.rand(4, 16).astype("float32"))
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(out_np >= 0.0) and np.all(out_np <= 1.0)

# تشغيل الاختبارات
test_model_builds()
test_model_trains()
test_model_predicts_range()
# النتيجة المتوقعة: كلّ الاختبارات تنجح.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-38-ch38-reproducibility.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def train_once(seed):
    """تدريب واحد ببذرة محدّدة، ويُعطي دقّة الاختبار."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([keras.Input(shape=(16,)),
                              keras.layers.Dense(32, activation="relu"),
                              keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(200, 16).astype("float32")
    y = np.random.rand(200, 1).astype("float32")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    # evaluate يعيد عددًا واحدًا حين تكون الخسارة فقط
    return float(np.ravel(model.evaluate(x, y, verbose=0))[0])

# ─── 1. خمس بذور
losses = [train_once(s) for s in (0, 1, 2, 3, 4)]
print(f"mean = {np.mean(losses):.4f}, std = {np.std(losses):.4f}")
# النتيجة المتوقعة: متوسّط وانحراف — الفرق يُحكَم بـ std لا برقم واحد.
