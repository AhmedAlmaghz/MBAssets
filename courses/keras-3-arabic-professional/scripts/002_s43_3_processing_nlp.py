# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.3 المعالجة اللغوية
# المصدر  : src/chapters/ch43_project_nlp.md:63
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.
