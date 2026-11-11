# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:282
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
# حساب الأوزان العكسية للتكرار
import numpy as np
counts = np.bincount(y_train.astype("int64"))
n = counts.sum()
weights = n / (len(counts) * counts)     # وزن أعلى للأصناف النادرة
model.fit(
    x_train, y_train,
    class_weight={i: float(w) for i, w in enumerate(weights)},
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                         save_best_only=True),
    ],
)
