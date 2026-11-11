# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `TextVectorization`: الطبقة المركزية
# المصدر  : src/chapters/ch16_text_layers.md:94
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)
