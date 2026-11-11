# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : القناع السببيّ عمليًّا
# المصدر  : src/chapters/ch29_transformers.md:173
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.
