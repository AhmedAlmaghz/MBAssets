# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثاني: مخرجات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:153
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# مخرجان: تصنيف العيّنة + انحدار العيب
class_out = keras.Input(shape=(1,), name="y_class")
reg_out = keras.Input(shape=(1,), name="y_reg")

inp = keras.Input(shape=(20,), name="features")
h = keras.layers.Dense(64, activation="relu")(inp)
h = keras.layers.Dense(32, activation="relu")(h)

pred_class = keras.layers.Dense(3, activation="softmax", name="class")(h)
pred_reg = keras.layers.Dense(1, name="reg")(h)

multi = keras.Model(inputs=inp, outputs=[pred_class, pred_reg])
multi.compile(
    optimizer="adam",
    loss={
        "class": "sparse_categorical_crossentropy",
        "reg": "mse",
    },
    loss_weights={
        "class": 1.0,
        "reg": 0.1,     # وازن بين المخرجين
    },
    metrics={
        "class": ["accuracy"],
        "reg": ["mae"],
    },
)
