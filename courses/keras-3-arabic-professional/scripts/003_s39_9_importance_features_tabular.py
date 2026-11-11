# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.9 أهميّة الميزات في النماذج الجدوليّة
# المصدر  : src/chapters/ch39_interpretability.md:162
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.
