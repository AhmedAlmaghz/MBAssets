# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.9 تشخيص استقرار التدريب
# المصدر  : src/chapters/ch32_gan.md:179
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.
