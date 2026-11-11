# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:198
# الترتيب : 10/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
