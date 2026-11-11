# ──────────────────────────────────────────────────────────────────────────
# الفصل 8: دوال الخسارة: كيف نترجم الخطأ إلى رقم؟
# القسم   : `sample_weight`: وزن لكل عيّنة
# المصدر  : src/chapters/ch08_loss_functions.md:196
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
weights = np.where(y_train == 1, 0.3, 1.0).astype("float32")
model.fit(x_train, y_train, sample_weight=weights, epochs=5, verbose=0)
