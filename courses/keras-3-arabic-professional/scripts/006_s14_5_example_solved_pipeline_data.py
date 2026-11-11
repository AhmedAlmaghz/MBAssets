# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:244
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")
