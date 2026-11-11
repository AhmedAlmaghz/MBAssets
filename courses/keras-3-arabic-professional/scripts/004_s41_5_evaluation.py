# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.5 التقييم الشامل
# المصدر  : src/chapters/ch41_project_vision.md:114
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التقييم على الاختبار
results = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
# النتيجة المتوقعة: loss وacc (الاستدعاء يُحسب يدويًّا أسفل).

# ─── 2. مقاييس لكلّ صنف
probs = keras.ops.softmax(model.predict(x_test, verbose=0), axis=-1)
preds = np.array(keras.ops.argmax(probs, axis=-1))
tp = int(((preds == 1) & (y_test == 1)).sum())
fp = int(((preds == 1) & (y_test == 0)).sum())
fn = int(((preds == 0) & (y_test == 1)).sum())
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-8)
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: precision وrecall متوازنان بعد الموازنة.
