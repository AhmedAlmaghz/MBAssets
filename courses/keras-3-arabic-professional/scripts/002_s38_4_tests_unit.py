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
