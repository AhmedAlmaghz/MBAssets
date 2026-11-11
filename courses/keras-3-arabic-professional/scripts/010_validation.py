# ──────────────────────────────────────────────────────────────────────────
# الفصل 2: البيئة: من الصفر إلى أول استيراد ناجح
# القسم   : التحقّق
# المصدر  : src/chapters/ch02_setup_environment.md:136
# الترتيب : 10/13
# ──────────────────────────────────────────────────────────────────────────
import keras

print(keras.__version__)        # 3.15.1
print(keras.backend.backend())  # jax
print(keras.backend.standardize_dtype("float32"))
