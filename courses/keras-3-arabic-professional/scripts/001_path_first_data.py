# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الأول: بيانات في الذاكرة
# المصدر  : src/chapters/ch14_data_pipeline.md:64
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)
