# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : معالجة العربية عمليًا
# المصدر  : src/chapters/ch16_text_layers.md:236
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}
