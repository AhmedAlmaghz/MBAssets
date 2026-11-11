# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.7 الخدمة
# المصدر  : src/chapters/ch41_project_vision.md:145
# الترتيب : 6/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. خدمة FastAPI (في الإنتاج)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Image(BaseModel):
    pixels: list[float]   # 64×64×3 مُسطَّحة

@app.post("/predict")
def predict(img: Image):
    """تلقّي صورة وإعطاء تسميّة وثقة."""
    x = np.array(img.pixels, dtype="float32").reshape(1, 64, 64, 3)
    logits = model.predict(x, verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    conf = float(np.array(keras.ops.max(probs, axis=-1))[0])
    label = int(np.array(keras.ops.argmax(probs, axis=-1))[0])
    return {"label": label, "confidence": conf,
            "threshold": 0.95}
# النتيجة المتوقعة: JSON يحتوي label وconfidence.
