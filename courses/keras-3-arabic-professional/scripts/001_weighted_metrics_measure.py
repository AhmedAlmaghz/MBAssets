# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : `weighted_metrics`: قياس مرجّح
# المصدر  : src/chapters/ch17_multi_task_weights.md:57
# الترتيب : 1/9
# ──────────────────────────────────────────────────────────────────────────
import keras

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
        keras.metrics.F1Score(average="macro"),
    ],
    weighted_metrics=[keras.metrics.TruePositives(), keras.metrics.FalsePositives()],
)
