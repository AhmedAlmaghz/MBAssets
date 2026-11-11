# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : تطبيق واقعي: خطّ إنتاج تدريب لا يُترك للمصادفة
# المصدر  : src/chapters/ch12_callbacks.md:500
# الترتيب : 9/11
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    keras.callbacks.CSVLogger("training_log.csv", append=True),
    keras.callbacks.ModelCheckpoint(
        filepath="best.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
]
