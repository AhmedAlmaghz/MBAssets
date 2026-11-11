# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.2 جلب البيانات
# المصدر  : src/chapters/ch05_first_model.md:28
# الترتيب : 2/14
# ──────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]
