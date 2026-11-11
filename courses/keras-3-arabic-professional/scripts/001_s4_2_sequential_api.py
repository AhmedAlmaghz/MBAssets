# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.2 الواجهة الأولى: Sequential
# المصدر  : src/chapters/ch04_model_architecture.md:37
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])

model.summary()
