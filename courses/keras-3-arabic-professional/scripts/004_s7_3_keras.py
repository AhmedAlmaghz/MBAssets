# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.3 التنفيذ في Keras 3
# المصدر  : src/chapters/ch07_activations.md:207
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

def make_moons(n=400, seed=0):
    """قمران متداخلان: كل صنف دائرة."""
    rng = np.random.default_rng(seed)
    x, y = [], []
    for _ in range(n // 2):
        t = rng.uniform(0, np.pi)
        x.append([np.cos(t), np.sin(t)]);            y.append(0)
        x.append([1 - np.cos(t), 0.5 - np.sin(t)]);   y.append(1)
    return np.array(x, "float32"), np.array(y, "float32")

x, y = make_moons()
x_val, y_val = make_moons(200, seed=7)

def build(activation):
    """نفس البنية تمامًا؛ يتغيّر فقط وجود nonlinearity."""
    m = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

linear_model = build(None)          # شبكة خطية بحتة
relu_model = build("relu")          # شبكة غير خطية
linear_model.fit(x, y, epochs=60, batch_size=32, verbose=0)
relu_model.fit(x, y, epochs=60, batch_size=32, verbose=0)

print("دقة الخطية:", linear_model.evaluate(x_val, y_val, verbose=0)[1])
print("دقة ReLU :", relu_model.evaluate(x_val, y_val, verbose=0)[1])
# النتيجة المتوقعة: الخطية ~0.5 (فشل تام)، ReLU أعلى من 0.9
