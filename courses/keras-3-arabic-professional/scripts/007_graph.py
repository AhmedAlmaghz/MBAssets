# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : كيف تتحكّم في الرسم؟
# المصدر  : src/chapters/ch03_tensors.md:171
# الترتيب : 7/12
# ──────────────────────────────────────────────────────────────────────────
import keras

x = keras.ops.convert_to_tensor([[1.0, 2.0]])
w = keras.ops.convert_to_tensor([[1.0], [1.0]])

with keras.name_scope("forward"):
    z = keras.ops.matmul(x, w)      # الجداء
    a = keras.ops.relu(z)           # التنشيط
    loss = keras.ops.sum(a)         # «الخسارة» البسيطة

print("Loss:", float(loss))
print("Forward value (z):", z)
