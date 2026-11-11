# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : النوع والجهاز
# المصدر  : src/chapters/ch03_tensors.md:48
# الترتيب : 2/12
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

x = np.random.rand(4, 28, 28, 3).astype("float32")
t = keras.ops.convert_to_tensor(x)

print(t.shape)   # (4, 28, 28, 3)
print(t.dtype)   # float32
