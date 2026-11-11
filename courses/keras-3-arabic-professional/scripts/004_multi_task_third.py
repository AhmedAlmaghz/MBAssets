# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الثالث: تعلّم متعدّد المهام (Multi-task)
# المصدر  : src/chapters/ch17_multi_task_weights.md:192
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(20,))
shared = keras.layers.Dense(64, activation="relu")(inp)
shared = keras.layers.BatchNormalization()(shared)
shared = keras.layers.Dense(32, activation="relu")(shared)

out_a = keras.layers.Dense(1, activation="sigmoid", name="task_a")(shared)
out_b = keras.layers.Dense(1, activation="sigmoid", name="task_b")(shared)

mtl = keras.Model(inputs=inp, outputs=[out_a, out_b])
mtl.compile(
    optimizer="adam",
    loss={"task_a": "binary_crossentropy", "task_b": "binary_crossentropy"},
    loss_weights={"task_a": 1.0, "task_b": 0.5},
)
