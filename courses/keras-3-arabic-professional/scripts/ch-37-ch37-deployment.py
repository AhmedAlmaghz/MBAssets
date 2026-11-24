#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التصدير والنشر
Keras 3 in Arabic — Deployment

Book slug : keras-3-arabic-professional
Chapter   : ch-37-ch37-deployment (lesson 37)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-37-ch37-deployment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التصدير والنشر
Keras 3 in Arabic — Deployment

Book slug : keras-3-arabic-professional
Chapter   : ch-37-ch37-deployment (lesson 37)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-37-ch37-deployment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التصدير والنشر
Keras 3 in Arabic — Deployment

Book slug : keras-3-arabic-professional
Chapter   : ch-37-ch37-deployment (lesson 37)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch37_deployment/001_s37_3_litert_tflite_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.3 التصدير إلى LiteRT/TFLite
# المصدر  : src/chapters/ch37_deployment.md:40
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch37_deployment/002_s37_5_onnx_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.5 التصدير إلى ONNX
# المصدر  : src/chapters/ch37_deployment.md:77
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch37_deployment/003_s37_6_fastapi_serving.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch37_deployment.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch37_deployment.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch37_deployment.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.3 التصدير إلى LiteRT/TFLite
# المصدر  : src/chapters/ch37_deployment.md:40
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.5 التصدير إلى ONNX
# المصدر  : src/chapters/ch37_deployment.md:77
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التصدير والنشر
Keras 3 in Arabic — Deployment

Book slug : keras-3-arabic-professional
Chapter   : ch-37-ch37-deployment (lesson 37)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch37_deployment/001_s37_3_litert_tflite_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.3 التصدير إلى LiteRT/TFLite
# المصدر  : src/chapters/ch37_deployment.md:40
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch37_deployment/002_s37_5_onnx_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.5 التصدير إلى ONNX
# المصدر  : src/chapters/ch37_deployment.md:77
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch37_deployment/003_s37_6_fastapi_serving.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch37_deployment.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch37_deployment.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch37_deployment.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.3 التصدير إلى LiteRT/TFLite
# المصدر  : src/chapters/ch37_deployment.md:40
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 37: التصدير والنشر
# القسم   : 37.5 التصدير إلى ONNX
# المصدر  : src/chapters/ch37_deployment.md:77
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# ─── 2. تصدير إلى LiteRT (تتطلّب حزمة LiteRT في الإنتاج)
# model.export("model.tflite")
# النتيجة المتوقعة: ملفّ .tflite بحجم أقلّ بكثير من .keras.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. نموذج بسيط
model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

# ─── 2. تصدير إلى ONNX (تتطلّب حزمة keras-onnx في الإنتاج)
# import tf2onnx
# spec = (keras.InputSpec(shape=(None, 16), dtype="float32"),)
# tf2onnx.convert.from_keras(model, input_signature=spec,
#                             output_path="model.onnx")
# النتيجة المتوقعة: ملفّ .onnx قابل للتشغيل على ONNX Runtime.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-37-ch37-deployment.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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
