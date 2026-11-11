# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.4 الحفظ والتحميل
# المصدر  : src/chapters/ch34_custom_layers.md:133
# الترتيب : 3/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نموذج بسيط يحوي الطبقة المخصّصة MyDense
model = keras.Sequential([
    keras.Input(shape=(16,)),
    MyDense(32, activation="relu"),
    MyDense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.rand(64, 16).astype("float32"),
          np.random.rand(64, 1).astype("float32"),
          epochs=1, batch_size=16, verbose=0)

# الحفظ
model.save("my_model.keras")

# التحميل — نُمرّر الطبقة المخصّصة في custom_objects
loaded = keras.saving.load_model("my_model.keras",
                                 custom_objects={"MyDense": MyDense})
print(type(loaded.layers[1]).__name__)   # النتيجة المتوقعة: MyDense
