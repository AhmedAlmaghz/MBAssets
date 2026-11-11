# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:336
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)
