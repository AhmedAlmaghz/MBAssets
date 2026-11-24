#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — النشر والتكامل مع الإنتاج
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 27 — TorchScript

Book slug : pytorch-keras-deep-learning
Chapter   : ch-27-ch27 (lesson 27)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-27-ch27.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — النشر والتكامل مع الإنتاج
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 27 — TorchScript

Book slug : pytorch-keras-deep-learning
Chapter   : ch-27-ch27 (lesson 27)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
import torch

model = model.eval()
dummy = torch.randn(1, 3, 224, 224)

# 1) TorchScript: يحوّل الرسم البياني إلى برنامج قابل للتنفيذ
scripted = torch.jit.trace(model, dummy)        # for models with control flow use script
scripted.save("model_ts.pt")
print("torchscript size:", os.path.getsize("model_ts.pt") / 1e6, "MB")

# 2) ONNX: المعيار الأوسع دعماً (ONNX Runtime، TensorRT، OpenVINO…)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},   # دفعات متغيّرة
    opset_version=17,
)
print("onnx size:", os.path.getsize("model.onnx") / 1e6, "MB")

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy")

# 1) ملف.keras — كل شيء في ملف واحد (weights + architecture + optimizer state)
model.save("classifier.keras")

# 2) تصدير ONNX (Keras 3.9+)
model.export("classifier.onnx")

# 3) تحميل لاحق
m2 = keras.saving.load_model("classifier.keras")
print(m2.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0).shape)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
# app.py — خدمة تصنيف صور
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import keras

app = FastAPI(title="Image Classifier API")
model = keras.saving.load_model("artifacts/classifier.keras")
CLASSES = ["cat", "dog", "car", "bird", "plane"]


def preprocess(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr[None,...]                 # (1, 224, 224, 3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "نوع ملف غير مدعوم")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "الملف كبير جداً")

    probs = model.predict(preprocess(raw), verbose=0)[0]
    idx = int(probs.argmax())
    return {
        "class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "model_version": "2026-10-20",
    }

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
# مراقبة بسيطة: تتبّع توزيع المخرجات ومقارنته بخطّ الأساس
import numpy as np

class OutputMonitor:
    def __init__(self, reference_probs: np.ndarray, z_threshold: float = 3.0):
        self.ref = reference_probs
        self.z = z_threshold

    def check(self, probs: np.ndarray) -> dict:
        mean_diff = float(np.mean(probs.mean(axis=0) - self.ref.mean(axis=0)))
        std = float(self.ref.std(axis=0).mean()) + 1e-9
        alert = abs(mean_diff) / std > self.z
        return {
            "mean_shift": mean_diff,
            "alert": alert,
            "message": "انحراف في التوزيع!" if alert else "ضمن النطاق",
        }

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
# اختبار التوافق (ضعه في tests/test_export.py)
import numpy as np
import torch
import onnxruntime as ort

def test_onnx_matches_torch(model, tmp_path):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    path = str(tmp_path / "m.onnx")
    torch.onnx.export(model, dummy, path,
                       input_names=["input"], output_names=["output"],
                       dynamic_axes={"input": {0: "b"}, "output": {0: "b"}},
                       opset_version=17)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for _ in range(5):
        x = np.random.rand(1, 3, 224, 224).astype("float32")
        y_onnx = sess.run(None, {"input": x})[0]
        y_torch = model(torch.tensor(x)).detach().numpy()
        assert np.abs(y_onnx - y_torch).max() < 1e-4

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()

def alert_if_drifted(train_probs, prod_probs, threshold=0.15):
    p = train_probs.mean(axis=0)
    q = prod_probs.mean(axis=0)
    tv = total_variation(p, q)
    return tv > threshold, tv

# اختبار بحالة مصطنعة
train = np.array([[0.7, 0.2, 0.1]] * 1000)
prod_drifted = np.array([[0.2, 0.2, 0.6]] * 500 + [[0.7, 0.2, 0.1]] * 500)
alert, tv = alert_if_drifted(train, prod_drifted)
print(f"TV={tv:.3f} alert={alert}")   # TV=0.300 alert=True

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path


def load_golden(path="golden.json"):
    """يحمّل الحزمة الذهبية: مدخلات + مخرجات مرجعية."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"]


def check_golden(model, cases, tolerance=1e-3):
    """يقارن مخرجات النموذج بالمرجع. أيّ اختلاف = إيقاف النشر."""
    failures = []
    for case in cases:
        pred = model.predict([case["input"]], verbose=0)[0]
        ref = case["expected"]
        diff = max(abs(float(a) - float(b)) for a, b in zip(pred, ref))
        if diff > tolerance:
            failures.append({
                "id": case["id"],
                "max_diff": diff,
                "expected": ref,
                "actual": [round(float(v), 6) for v in pred],
            })
    return failures


if __name__ == "__main__":
    bad = check_golden(model, load_golden())
    if bad:
        print("GOLDEN TEST FAILED")
        for f in bad[:5]:
            print(f)
        raise SystemExit(1)
    print("golden test passed")

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch27.html
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

app = FastAPI(title="نموذج فحص العيوب")

# يُحمَّل النموذج مرة واحدة عند بدء العملية، لا عند كلّ طلب
import torch

_CHECKPOINT = "defects.pt"
_model = None
_mean = None
_std = None


def load_model(path: str = _CHECKPOINT):
    """يحمّل النموذج والإحصاءات المرافقة في الذاكرة."""
    global _model, _mean, _std
    blob = torch.load(path, map_location="cpu", weights_only=True)
    _model = blob["model"]
    _model.eval()
    _mean = blob["mean"]
    _std = blob["std"]
    return _model


class PredictRequest(BaseModel):
    image: list[float]          # مصفوفة مسطّحة (H, W, C)
    height: int
    width: int


@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        load_model()
    arr = np.asarray(req.image, dtype=np.float32).reshape(req.height, req.width, 3)
    arr = (arr - _mean) / _std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    best = int(np.argmax(probs))
    return {"class_id": best, "confidence": round(probs[best], 4), "probs": probs}

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch

model = model.eval()
dummy = torch.randn(1, 3, 224, 224)

# 1) TorchScript: يحوّل الرسم البياني إلى برنامج قابل للتنفيذ
scripted = torch.jit.trace(model, dummy)        # for models with control flow use script
scripted.save("model_ts.pt")
print("torchscript size:", os.path.getsize("model_ts.pt") / 1e6, "MB")

# 2) ONNX: المعيار الأوسع دعماً (ONNX Runtime، TensorRT، OpenVINO…)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},   # دفعات متغيّرة
    opset_version=17,
)
print("onnx size:", os.path.getsize("model.onnx") / 1e6, "MB")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy")

# 1) ملف.keras — كل شيء في ملف واحد (weights + architecture + optimizer state)
model.save("classifier.keras")

# 2) تصدير ONNX (Keras 3.9+)
model.export("classifier.onnx")

# 3) تحميل لاحق
m2 = keras.saving.load_model("classifier.keras")
print(m2.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0).shape)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# app.py — خدمة تصنيف صور
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import keras

app = FastAPI(title="Image Classifier API")
model = keras.saving.load_model("artifacts/classifier.keras")
CLASSES = ["cat", "dog", "car", "bird", "plane"]


def preprocess(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr[None,...]                 # (1, 224, 224, 3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "نوع ملف غير مدعوم")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "الملف كبير جداً")

    probs = model.predict(preprocess(raw), verbose=0)[0]
    idx = int(probs.argmax())
    return {
        "class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "model_version": "2026-10-20",
    }

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# مراقبة بسيطة: تتبّع توزيع المخرجات ومقارنته بخطّ الأساس
import numpy as np

class OutputMonitor:
    def __init__(self, reference_probs: np.ndarray, z_threshold: float = 3.0):
        self.ref = reference_probs
        self.z = z_threshold

    def check(self, probs: np.ndarray) -> dict:
        mean_diff = float(np.mean(probs.mean(axis=0) - self.ref.mean(axis=0)))
        std = float(self.ref.std(axis=0).mean()) + 1e-9
        alert = abs(mean_diff) / std > self.z
        return {
            "mean_shift": mean_diff,
            "alert": alert,
            "message": "انحراف في التوزيع!" if alert else "ضمن النطاق",
        }

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# اختبار التوافق (ضعه في tests/test_export.py)
import numpy as np
import torch
import onnxruntime as ort

def test_onnx_matches_torch(model, tmp_path):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    path = str(tmp_path / "m.onnx")
    torch.onnx.export(model, dummy, path,
                       input_names=["input"], output_names=["output"],
                       dynamic_axes={"input": {0: "b"}, "output": {0: "b"}},
                       opset_version=17)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for _ in range(5):
        x = np.random.rand(1, 3, 224, 224).astype("float32")
        y_onnx = sess.run(None, {"input": x})[0]
        y_torch = model(torch.tensor(x)).detach().numpy()
        assert np.abs(y_onnx - y_torch).max() < 1e-4

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()

def alert_if_drifted(train_probs, prod_probs, threshold=0.15):
    p = train_probs.mean(axis=0)
    q = prod_probs.mean(axis=0)
    tv = total_variation(p, q)
    return tv > threshold, tv

# اختبار بحالة مصطنعة
train = np.array([[0.7, 0.2, 0.1]] * 1000)
prod_drifted = np.array([[0.2, 0.2, 0.6]] * 500 + [[0.7, 0.2, 0.1]] * 500)
alert, tv = alert_if_drifted(train, prod_drifted)
print(f"TV={tv:.3f} alert={alert}")   # TV=0.300 alert=True

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path


def load_golden(path="golden.json"):
    """يحمّل الحزمة الذهبية: مدخلات + مخرجات مرجعية."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"]


def check_golden(model, cases, tolerance=1e-3):
    """يقارن مخرجات النموذج بالمرجع. أيّ اختلاف = إيقاف النشر."""
    failures = []
    for case in cases:
        pred = model.predict([case["input"]], verbose=0)[0]
        ref = case["expected"]
        diff = max(abs(float(a) - float(b)) for a, b in zip(pred, ref))
        if diff > tolerance:
            failures.append({
                "id": case["id"],
                "max_diff": diff,
                "expected": ref,
                "actual": [round(float(v), 6) for v in pred],
            })
    return failures


if __name__ == "__main__":
    bad = check_golden(model, load_golden())
    if bad:
        print("GOLDEN TEST FAILED")
        for f in bad[:5]:
            print(f)
        raise SystemExit(1)
    print("golden test passed")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-27-ch27.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

app = FastAPI(title="نموذج فحص العيوب")

# يُحمَّل النموذج مرة واحدة عند بدء العملية، لا عند كلّ طلب
import torch

_CHECKPOINT = "defects.pt"
_model = None
_mean = None
_std = None


def load_model(path: str = _CHECKPOINT):
    """يحمّل النموذج والإحصاءات المرافقة في الذاكرة."""
    global _model, _mean, _std
    blob = torch.load(path, map_location="cpu", weights_only=True)
    _model = blob["model"]
    _model.eval()
    _mean = blob["mean"]
    _std = blob["std"]
    return _model


class PredictRequest(BaseModel):
    image: list[float]          # مصفوفة مسطّحة (H, W, C)
    height: int
    width: int


@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        load_model()
    arr = np.asarray(req.image, dtype=np.float32).reshape(req.height, req.width, 3)
    arr = (arr - _mean) / _std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    best = int(np.argmax(probs))
    return {"class_id": best, "confidence": round(probs[best], 4), "probs": probs}
