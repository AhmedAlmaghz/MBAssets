# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.7 الخدمة
# المصدر  : src/chapters/ch43_project_nlp.md:144
# الترتيب : 6/8
# ──────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Review(BaseModel):
    text: str

# التسميات بعد التحويل
LABEL_NAMES = {0: "سلبيّة", 1: "محايدة", 2: "إيجابيّة"}

@app.post("/sentiment")
def sentiment(review: Review):
    """تلقّي نصّ وإعطاء تقييم وثقة."""
    x = np.array([review.text], dtype=object)
    probs = model.predict(x, verbose=0)[0]
    label = int(np.argmax(probs))
    conf = float(probs[label])
    return {
        "label": LABEL_NAMES[label],
        "confidence": conf,
        "threshold": 0.60,
    }
# النتيجة المتوقعة: JSON يحتوي التسميّة والثقة.
