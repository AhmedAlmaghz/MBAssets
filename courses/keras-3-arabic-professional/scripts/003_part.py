# ──────────────────────────────────────────────────────────────────────────
# الفصل 9: المُحسّنات: كيف يتحرّك النموذج نحو الحل؟
# القسم   : كيف نمرّرها؟
# المصدر  : src/chapters/ch09_optimizers.md:163
# الترتيب : 3/6
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine),
    loss="sparse_categorical_crossentropy",
)
