# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : 17.5 مثال محلول: تصنيف معاملات مشبوهة
# المصدر  : src/chapters/ch17_multi_task_weights.md:300
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: جرّب عتبات مختلفة على الاحتمالات
probs = model.predict(x_val).ravel()
y_true = y_val.ravel()
for thr in (0.3, 0.5, 0.7):
    pred = (probs >= thr).astype("float32")
    tp = float(np.sum((pred == 1) & (y_true == 1)))
    fp = float(np.sum((pred == 1) & (y_true == 0)))
    fn = float(np.sum((pred == 0) & (y_true == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"thr={thr}: P={precision:.3f} R={recall:.3f}")
