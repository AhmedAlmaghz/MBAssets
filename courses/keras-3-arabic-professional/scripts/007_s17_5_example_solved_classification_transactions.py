# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:278
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
loss = keras.losses.BinaryFocalCrossentropy(
    from_logits=True,
    alpha=0.75,     # أنزّل وزن الصنف الشائع
    gamma=2.0,
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss,
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC(),
    ],
)
