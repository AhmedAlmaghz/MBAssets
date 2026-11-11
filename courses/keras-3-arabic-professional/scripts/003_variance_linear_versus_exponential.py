# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : جدول التباين: خطّيّ مقابل أسّيّ
# المصدر  : src/chapters/ch33_diffusion.md:178
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.
