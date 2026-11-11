# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : زمن التوليد: DDPM مقابل DDIM
# المصدر  : src/chapters/ch33_diffusion.md:210
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")
