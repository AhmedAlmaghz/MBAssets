# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.5 التدريب والتقييم
# المصدر  : src/chapters/ch43_project_nlp.md:109
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.
