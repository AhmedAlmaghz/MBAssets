#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — معالجة اللغة الطبيعية عمليًا
Keras 3 in Arabic — Nlp Practical

Book slug : keras-3-arabic-professional
Chapter   : ch-27-ch27-nlp-practical (lesson 27)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-27-ch27-nlp-practical.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — معالجة اللغة الطبيعية عمليًا
Keras 3 in Arabic — Nlp Practical

Book slug : keras-3-arabic-professional
Chapter   : ch-27-ch27-nlp-practical (lesson 27)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-27-ch27-nlp-practical.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — معالجة اللغة الطبيعية عمليًا
Keras 3 in Arabic — Nlp Practical

Book slug : keras-3-arabic-professional
Chapter   : ch-27-ch27-nlp-practical (lesson 27)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch27_nlp_practical/001_s27_2_textvectorization_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.2 TextVectorization: من نصّ إلى أرقام
# المصدر  : src/chapters/ch27_nlp_practical.md:31
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch27_nlp_practical/002_s27_3_embedding.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.3 Embedding: من رقم إلى متّجه
# المصدر  : src/chapters/ch27_nlp_practical.md:80
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch27_nlp_practical/003_s27_4_model_classification_text_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.4 نموذج تصنيف نصّ كامل
# المصدر  : src/chapters/ch27_nlp_practical.md:93
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch27_nlp_practical/004_choosing_length_sequence_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : تحديد طول التسلسل من البيانات
# المصدر  : src/chapters/ch27_nlp_practical.md:192
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.2 TextVectorization: من نصّ إلى أرقام
# المصدر  : src/chapters/ch27_nlp_practical.md:31
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.3 Embedding: من رقم إلى متّجه
# المصدر  : src/chapters/ch27_nlp_practical.md:80
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.4 نموذج تصنيف نصّ كامل
# المصدر  : src/chapters/ch27_nlp_practical.md:93
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : تحديد طول التسلسل من البيانات
# المصدر  : src/chapters/ch27_nlp_practical.md:192
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — معالجة اللغة الطبيعية عمليًا
Keras 3 in Arabic — Nlp Practical

Book slug : keras-3-arabic-professional
Chapter   : ch-27-ch27-nlp-practical (lesson 27)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch27_nlp_practical/001_s27_2_textvectorization_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.2 TextVectorization: من نصّ إلى أرقام
# المصدر  : src/chapters/ch27_nlp_practical.md:31
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch27_nlp_practical/002_s27_3_embedding.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.3 Embedding: من رقم إلى متّجه
# المصدر  : src/chapters/ch27_nlp_practical.md:80
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch27_nlp_practical/003_s27_4_model_classification_text_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.4 نموذج تصنيف نصّ كامل
# المصدر  : src/chapters/ch27_nlp_practical.md:93
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch27_nlp_practical/004_choosing_length_sequence_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : تحديد طول التسلسل من البيانات
# المصدر  : src/chapters/ch27_nlp_practical.md:192
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch27_nlp_practical.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.2 TextVectorization: من نصّ إلى أرقام
# المصدر  : src/chapters/ch27_nlp_practical.md:31
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.3 Embedding: من رقم إلى متّجه
# المصدر  : src/chapters/ch27_nlp_practical.md:80
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : 27.4 نموذج تصنيف نصّ كامل
# المصدر  : src/chapters/ch27_nlp_practical.md:93
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 27: معالجة اللغة الطبيعية عمليًا
# القسم   : تحديد طول التسلسل من البيانات
# المصدر  : src/chapters/ch27_nlp_practical.md:192
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية صغيرة
texts = np.array([
    "المنتج ممتاز جدًا",
    "الخدمة سيئة والموظف غير متعاون",
    "السعر مرتفع لكن الجودة جيدة",
    "توصيل سريع وجودة عالية",
    "أسوأ تجربة شراء",
    "منتج عادي ولا أُنصح به",
], dtype=object)
labels = np.array([1, 0, 1, 1, 0, 0])

# ─── 2. طبقة التحويل
vectorizer = keras.layers.TextVectorization(
    max_tokens=1000,                  # أقصى عدد كلمات في المفردات
    output_mode="int",                # كلّ كلمة → رقم
    output_sequence_length=10,        # كلّ نصّ → تسلسل من 10 أرقام
)
# الخطوة الأولى: بناء المفردات على التدريب فقط
vectorizer.adapt(texts)
# الخطوة الثانية: تحويل النصوص
seq = vectorizer(texts)
# النتيجة المتوقعة: أوتور (6, 10) بأرقام صحيحة

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# تضمين بسيط
embed = keras.layers.Embedding(input_dim=1000, output_dim=32)
out = embed(keras.ops.array([[1, 2, 3]]))
# النتيجة المتوقعة: أوتور (1, 3, 32) — متّجه 32 عنصرًا لكلّ كلمة.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات نصّية (مصنّعة) — 400 مراجعة
n = 400
vocab = ["ممتاز", "سيئ", "جيد", "متوسط", "مُنصح", "غير", "توصيل", "سريع",
         "بطيء", "سعر", "مرتفع", "منخفض", "جودة", "عالية", "خدمة", "موظف"]
texts = []
labels = []
for _ in range(n):
    words = list(np.random.choice(vocab, size=8, replace=True))
    texts.append(" ".join(words))
    # قاعدة مبسّطة: كلمات موجبة → 1
    positive = {"ممتاز", "جيد", "منصح", "سريع", "منخفض", "عالية"}
    labels.append(int(any(w in positive for w in words)))
texts = np.array(texts, dtype=object)
labels = np.array(labels, dtype="int32")

# ─── 2. التحويل إلى أرقام
vectorizer = keras.layers.TextVectorization(max_tokens=200,
                                            output_sequence_length=12)
vectorizer.adapt(texts)

# ─── 3. النموذج: تضمين → GlobalAveragePooling → تصنيف
inp = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inp)                                       # (B, 12) أرقام
x = keras.layers.Embedding(input_dim=200, output_dim=32)(x)   # (B, 12, 32)
x = keras.layers.GlobalAveragePooling1D()(x)              # (B, 32)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(texts, labels, epochs=10, batch_size=16,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.75–0.85 على البيانات المصنّعة.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-27-ch27-nlp-practical.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def choose_sequence_length(texts, coverage=0.95):
    """يختار طولًا يغطي coverage من النصوص."""
    lengths = np.array([len(str(t).split()) for t in texts])
    chosen = int(np.quantile(lengths, coverage))
    return chosen, lengths

# ─── 1. بيانات بأطوال متفاوتة (محاكاة)
rng = np.random.default_rng(42)
texts = [" ".join(["كلمة"] * int(rng.integers(5, 80))) for _ in range(1000)]

# ─── 2. اختيار الطول
chosen, lengths = choose_sequence_length(texts, coverage=0.95)
print(f"متوسّط الطول = {lengths.mean():.1f}، أقصى = {lengths.max()}")
print(f"طول يغطي 95% = {chosen}")
print(f"نسبة الحشو عند {chosen} = "
      f"{1 - lengths.mean() / chosen:.2%}")
# النتيجة المتوقعة: الطول المختار أقلّ من الأقصى بمرّات،
# ونسبة الحشو في النطاق المقبول (< 40%).
