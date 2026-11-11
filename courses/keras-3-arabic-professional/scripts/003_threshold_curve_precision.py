# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : اختيار العتبة على منحنى الدقّة والاستدعاء
# المصدر  : src/chapters/ch31_vae.md:248
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.
