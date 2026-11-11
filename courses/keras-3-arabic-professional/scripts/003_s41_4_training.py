# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.4 التدريب
# المصدر  : src/chapters/ch41_project_vision.md:90
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. الاستدعاءات
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras",
                                    monitor="val_acc",
                                    save_best_only=True, verbose=1),
]

# ─── 2. التدريب
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20, batch_size=32,
    class_weight=weights,
    callbacks=callbacks,
    verbose=0,
)
# النتيجة المتوقعة: التدريب يتوقف عند عدم تحسّن، ويُستأنف من الأفضل.
