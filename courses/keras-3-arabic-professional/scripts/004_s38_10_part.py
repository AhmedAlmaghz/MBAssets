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
