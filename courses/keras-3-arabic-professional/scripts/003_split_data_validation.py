# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : تقسيم بيانات التحقق
# المصدر  : src/chapters/ch05_first_model.md:48
# الترتيب : 3/14
# ──────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]
