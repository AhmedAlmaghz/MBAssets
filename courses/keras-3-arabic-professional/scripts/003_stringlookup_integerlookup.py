# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `StringLookup` و`IntegerLookup`: القواميس صراحةً
# المصدر  : src/chapters/ch16_text_layers.md:183
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])
