# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:246
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "category": keras.losses.CategoricalCrossentropy(),
        "price": keras.losses.MeanSquaredError(),
    },
    metrics={
        "category": [keras.metrics.CategoricalAccuracy()],
        "price": [keras.metrics.MeanAbsoluteError()],
    },
)

history = model.fit(
    x_train, {"category": y_category, "price": y_price},
    epochs=10, batch_size=64, validation_split=0.2,
)
