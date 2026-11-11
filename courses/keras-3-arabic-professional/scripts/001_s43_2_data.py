# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.2 البيانات
# المصدر  : src/chapters/ch43_project_nlp.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).
