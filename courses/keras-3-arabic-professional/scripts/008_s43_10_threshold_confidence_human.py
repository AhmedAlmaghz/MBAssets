# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.10 عتبة الثقة والتوجيه البشريّ
# المصدر  : src/chapters/ch43_project_nlp.md:226
# الترتيب : 8/8
# ──────────────────────────────────────────────────────────────────────────
def route_review(probs, threshold=0.60):
    """توجيه المراجعة: تلقائيّ إن كانت الثقة عالية، وإلّا بشريّ."""
    label = int(np.argmax(probs))
    conf = float(np.max(probs))
    if conf >= threshold:
        return {"route": "auto", "label": label, "confidence": conf}
    return {"route": "human", "label": label, "confidence": conf}

# محاكاة على 100 مراجعة
auto, human = 0, 0
for _ in range(100):
    probs = np.random.dirichlet([2, 2, 2]).astype("float32")
    r = route_review(probs)
    auto += r["route"] == "auto"
    human += r["route"] == "human"
print(f"auto = {auto}%, human = {human}%")
# النتيجة المتوقعة: نسبة التوجيه البشريّ تُقاس وتُضبط بالعتبة.
