# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.3 الواجهة الثانية: Functional — الأصل الحقيقي
# المصدر  : src/chapters/ch04_model_architecture.md:58
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(784,), name="input_layer")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="output_layer")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
