# ──────────────────────────────────────────────────────────────────────────
# الفصل 42: مشروع التنبّؤ بالسلاسل الزمنيّة
# القسم   : 42.4 خطّ الأساس
# المصدر  : src/chapters/ch42_project_timeseries.md:92
# الترتيب : 3/9
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خطّ الأساس: التكرار السابق (naive forecast)
naive_pred = x_test[:, -1, 0]
naive_mae = np.mean(np.abs(naive_pred - y_test))
print(f"naive MAE = {naive_mae:.4f}")

# ─── 2. خطّ الأساس الثاني: متوسّط النافذة
avg_pred = x_test.mean(axis=1)[:, 0]
avg_mae = np.mean(np.abs(avg_pred - y_test))
print(f"moving-avg MAE = {avg_mae:.4f}")
# النتيجة المتوقعة: خطّا أساس بقيم MAE قابلة للمقارنة.
