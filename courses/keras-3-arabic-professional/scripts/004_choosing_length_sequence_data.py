# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : تحديد طول التسلسل من البيانات
# المصدر  : src/chapters/ch27_nlp_practical.md:192
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).
