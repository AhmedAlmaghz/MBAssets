# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.10 تشخيص الانحراف بعد النشر
# المصدر  : src/chapters/ch39_interpretability.md:213
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.
