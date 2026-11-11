# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : فرع واحد بمخرجين
# المصدر  : src/chapters/ch04_model_architecture.md:99
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inputs = keras.Input(shape=(100,))
shared = keras.layers.Dense(64, activation="relu")(inputs)

head_a = keras.layers.Dense(3, activation="softmax", name="task_a")(shared)
head_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

model = keras.Model(inputs=inputs, outputs=[head_a, head_b])
