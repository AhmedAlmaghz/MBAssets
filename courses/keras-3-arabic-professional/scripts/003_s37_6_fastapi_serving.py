# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.6 خدمة FastAPI
# المصدر  : src/chapters/ch37_deployment.md:100
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. ملفّ app.py
import keras
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ─── 2. تحميل النموذج مرة واحدة عند بدء الخدمة
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="mse")

class Features(BaseModel):
    values: list[float]

@app.post("/predict")
def predict(f: Features):
    """تلقّي 16 ميزة وإعطاء احتمال."""
    x = np.array(f.values, dtype="float32").reshape(1, -1)
    prob = float(model.predict(x, verbose=0)[0][0])
    return {"probability": prob, "threshold": 0.5,
            "label": int(prob >= 0.5)}
# النتيجة المتوقعة: JSON يحتوي probability وlabel.
