# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.5 التدريب: `fit`
# المصدر  : src/chapters/ch05_first_model.md:150
# الترتيب : 8/14
# ──────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)
