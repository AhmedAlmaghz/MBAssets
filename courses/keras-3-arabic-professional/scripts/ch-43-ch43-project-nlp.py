#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تحليل المشاعر ونشره كخدمة
Keras 3 in Arabic — Project Nlp

Book slug : keras-3-arabic-professional
Chapter   : ch-43-ch43-project-nlp (lesson 43)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-43-ch43-project-nlp.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تحليل المشاعر ونشره كخدمة
Keras 3 in Arabic — Project Nlp

Book slug : keras-3-arabic-professional
Chapter   : ch-43-ch43-project-nlp (lesson 43)
Code files: 17 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-43-ch43-project-nlp.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تحليل المشاعر ونشره كخدمة
Keras 3 in Arabic — Project Nlp

Book slug : keras-3-arabic-professional
Chapter   : ch-43-ch43-project-nlp (lesson 43)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch43_project_nlp/001_s43_2_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.2 البيانات
# المصدر  : src/chapters/ch43_project_nlp.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch43_project_nlp/002_s43_3_processing_nlp.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.3 المعالجة اللغوية
# المصدر  : src/chapters/ch43_project_nlp.md:63
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch43_project_nlp/003_s43_4_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.4 النموذج
# المصدر  : src/chapters/ch43_project_nlp.md:84
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch43_project_nlp/004_s43_5_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.5 التدريب والتقييم
# المصدر  : src/chapters/ch43_project_nlp.md:109
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch43_project_nlp/005_s43_6_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.6 التصدير
# المصدر  : src/chapters/ch43_project_nlp.md:133
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch43_project_nlp/006_s43_7_serving.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch43_project_nlp/007_s43_9_normalization_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.9 تطبيع النصّ العربيّ
# المصدر  : src/chapters/ch43_project_nlp.md:188
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch43_project_nlp/008_s43_10_threshold_confidence_human.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.2 البيانات
# المصدر  : src/chapters/ch43_project_nlp.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.3 المعالجة اللغوية
# المصدر  : src/chapters/ch43_project_nlp.md:63
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.4 النموذج
# المصدر  : src/chapters/ch43_project_nlp.md:84
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.5 التدريب والتقييم
# المصدر  : src/chapters/ch43_project_nlp.md:109
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.6 التصدير
# المصدر  : src/chapters/ch43_project_nlp.md:133
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.9 تطبيع النصّ العربيّ
# المصدر  : src/chapters/ch43_project_nlp.md:188
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — مشروع تحليل المشاعر ونشره كخدمة
Keras 3 in Arabic — Project Nlp

Book slug : keras-3-arabic-professional
Chapter   : ch-43-ch43-project-nlp (lesson 43)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch43_project_nlp/001_s43_2_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.2 البيانات
# المصدر  : src/chapters/ch43_project_nlp.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch43_project_nlp/002_s43_3_processing_nlp.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.3 المعالجة اللغوية
# المصدر  : src/chapters/ch43_project_nlp.md:63
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch43_project_nlp/003_s43_4_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.4 النموذج
# المصدر  : src/chapters/ch43_project_nlp.md:84
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch43_project_nlp/004_s43_5_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.5 التدريب والتقييم
# المصدر  : src/chapters/ch43_project_nlp.md:109
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch43_project_nlp/005_s43_6_export.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.6 التصدير
# المصدر  : src/chapters/ch43_project_nlp.md:133
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch43_project_nlp/006_s43_7_serving.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch43_project_nlp/007_s43_9_normalization_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.9 تطبيع النصّ العربيّ
# المصدر  : src/chapters/ch43_project_nlp.md:188
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch43_project_nlp/008_s43_10_threshold_confidence_human.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch43_project_nlp.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.2 البيانات
# المصدر  : src/chapters/ch43_project_nlp.md:31
# الترتيب : 1/8
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.3 المعالجة اللغوية
# المصدر  : src/chapters/ch43_project_nlp.md:63
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.4 النموذج
# المصدر  : src/chapters/ch43_project_nlp.md:84
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.5 التدريب والتقييم
# المصدر  : src/chapters/ch43_project_nlp.md:109
# الترتيب : 4/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.6 التصدير
# المصدر  : src/chapters/ch43_project_nlp.md:133
# الترتيب : 5/8
# ──────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.9 تطبيع النصّ العربيّ
# المصدر  : src/chapters/ch43_project_nlp.md:188
# الترتيب : 7/8
# ──────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة (في الإنتاج: مراجعات حقيقيّة)
positive = ["ممتاز", "جيد", "رائع", "منصح", "سريع", "عالية"]
negative = ["سيئ", "أسوأ", "بطيء", "متأخر", "مرتفع", "متواضع"]
neutral = ["عادي", "متوسط", "مقبول", "معقول"]

def make_review(label):
    """مراجعة مُصنّعة بحسب التسمية."""
    words = {"1": positive, "0": neutral, "-1": negative}[str(label)]
    n_words = np.random.randint(4, 10)
    return " ".join(list(np.random.choice(words, size=n_words)))

n = 3000
labels = np.random.choice([1, 0, -1], size=n, p=[0.5, 0.2, 0.3])
texts = np.array([make_review(l) for l in labels], dtype=object)

# ─── 2. تقسيم
n_train = int(0.7 * n)
n_val = int(0.1 * n)
x_train, y_train = texts[:n_train], labels[:n_train]
x_val, y_val = texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]
x_test, y_test = texts[n_train + n_val:], labels[n_train + n_val:]
# النتيجة المتوقعة: ثلاثة مجموعات (تدريب، تحقّق، اختبار).

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ─── 1. طبقة تحويل النصّ
vectorizer = keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
)
vectorizer.adapt(x_train)       # يُبنى على التدريب فقط

# ─── 2. تحويل التسميات إلى أرقام (من -1 إلى 0..2)
label_map = {-1: 0, 0: 1, 1: 2}
y_train_idx = np.array([label_map[l] for l in y_train])
y_val_idx = np.array([label_map[l] for l in y_val])
y_test_idx = np.array([label_map[l] for l in y_test])
# النتيجة المتوقعة: تسميات في {0, 1, 2}.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ─── 1. التدريب
model.fit(x_train, y_train_idx, validation_data=(x_val, y_val_idx),
          epochs=15, batch_size=32, verbose=0)

# ─── 2. التقييم على الاختبار
results = model.evaluate(x_test, y_test_idx, return_dict=True, verbose=0)

# ─── 3. مقاييس لكلّ صنف (F1 لكلّ صنف)
preds = np.array(keras.ops.argmax(
    model.predict(x_test, verbose=0), axis=-1))
for cls in range(3):
    tp = int(((preds == cls) & (y_test_idx == cls)).sum())
    fp = int(((preds == cls) & (y_test_idx != cls)).sum())
    fn = int(((preds != cls) & (y_test_idx == cls)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f"class {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
# النتيجة المتوقعة: F1 لكلّ صنف — لا صنف مُهمل.

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ─── 1. حفظ النموذج
model.save("sentiment_model.keras")

# ─── 2. تصدير (في الإنتاج)
# model.export("sentiment.tflite")
# النتيجة المتوقعة: ملفّ جاهز للنشر.

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import re

def normalize_arabic(text):
    """تطبيع النصّ العربيّ: توحيد الهمزات، إزالة التشكيل، توحيد الألف."""
    text = str(text)
    # إزالة التشكيل
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # توحيد الهمزات
    text = (text.replace("أ", "ا").replace("إ", "ا")
                .replace("آ", "ا").replace("ؤ", "و").replace("ئ", "ي"))
    # توحيد التاء المربوطة والهاء
    text = text.replace("ة", "ه")
    # إزالة الأرقام وعلامات الترقيم
    text = re.sub(r"[0-9\u0660-\u0669]", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # إزالة المسافات الزائدة
    return re.sub(r"\s+", " ", text).strip()

samples = ["مُمتازٌ جِدًّا!", "الخدمة سيّئة جدا", "أحبّ هذا المنتَج"]
cleaned = [normalize_arabic(s) for s in samples]
print(cleaned)
# النتيجة المتوقعة: ['ممتاز جدا', 'الخدمه سيئه جدا', 'احب هذا المنتج'].

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-43-ch43-project-nlp.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
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
