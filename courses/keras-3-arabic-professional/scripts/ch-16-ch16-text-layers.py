#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات النصية: من الجملة إلى المتجه
Keras 3 in Arabic — Text Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-16-ch16-text-layers (lesson 16)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-16-ch16-text-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات النصية: من الجملة إلى المتجه
Keras 3 in Arabic — Text Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-16-ch16-text-layers (lesson 16)
Code files: 21 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-16-ch16-text-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات النصية: من الجملة إلى المتجه
Keras 3 in Arabic — Text Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-16-ch16-text-layers (lesson 16)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch16_text_layers/001_textvectorization_layer.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `TextVectorization`: الطبقة المركزية
# المصدر  : src/chapters/ch16_text_layers.md:94
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch16_text_layers/002_int_embedding_path_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : من `int` إلى `Embedding`: المسار الكامل
# المصدر  : src/chapters/ch16_text_layers.md:133
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch16_text_layers/003_stringlookup_integerlookup.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `StringLookup` و`IntegerLookup`: القواميس صراحةً
# المصدر  : src/chapters/ch16_text_layers.md:183
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch16_text_layers/004_hashing.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `Hashing`: قاموس بلا ذاكرة
# المصدر  : src/chapters/ch16_text_layers.md:201
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch16_text_layers/005_categoryencoding_discretization.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `CategoryEncoding` و`Discretization`
# المصدر  : src/chapters/ch16_text_layers.md:219
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch16_text_layers/006_processing_arabic.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : معالجة العربية عمليًا
# المصدر  : src/chapters/ch16_text_layers.md:236
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch16_text_layers/007_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:292
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch16_text_layers/008_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:327
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch16_text_layers/009_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:336
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch16_text_layers/010_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:352
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `TextVectorization`: الطبقة المركزية
# المصدر  : src/chapters/ch16_text_layers.md:94
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : من `int` إلى `Embedding`: المسار الكامل
# المصدر  : src/chapters/ch16_text_layers.md:133
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `StringLookup` و`IntegerLookup`: القواميس صراحةً
# المصدر  : src/chapters/ch16_text_layers.md:183
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `Hashing`: قاموس بلا ذاكرة
# المصدر  : src/chapters/ch16_text_layers.md:201
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `CategoryEncoding` و`Discretization`
# المصدر  : src/chapters/ch16_text_layers.md:219
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : معالجة العربية عمليًا
# المصدر  : src/chapters/ch16_text_layers.md:236
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:292
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:327
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:336
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:352
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الطبقات النصية: من الجملة إلى المتجه
Keras 3 in Arabic — Text Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-16-ch16-text-layers (lesson 16)
Code files: 20 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch16_text_layers/001_textvectorization_layer.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `TextVectorization`: الطبقة المركزية
# المصدر  : src/chapters/ch16_text_layers.md:94
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch16_text_layers/002_int_embedding_path_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : من `int` إلى `Embedding`: المسار الكامل
# المصدر  : src/chapters/ch16_text_layers.md:133
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch16_text_layers/003_stringlookup_integerlookup.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `StringLookup` و`IntegerLookup`: القواميس صراحةً
# المصدر  : src/chapters/ch16_text_layers.md:183
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch16_text_layers/004_hashing.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `Hashing`: قاموس بلا ذاكرة
# المصدر  : src/chapters/ch16_text_layers.md:201
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch16_text_layers/005_categoryencoding_discretization.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `CategoryEncoding` و`Discretization`
# المصدر  : src/chapters/ch16_text_layers.md:219
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch16_text_layers/006_processing_arabic.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : معالجة العربية عمليًا
# المصدر  : src/chapters/ch16_text_layers.md:236
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch16_text_layers/007_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:292
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch16_text_layers/008_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:327
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch16_text_layers/009_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:336
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch16_text_layers/010_s16_5_example_solved_analysis_sentiment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:352
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch16_text_layers.md
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `TextVectorization`: الطبقة المركزية
# المصدر  : src/chapters/ch16_text_layers.md:94
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : من `int` إلى `Embedding`: المسار الكامل
# المصدر  : src/chapters/ch16_text_layers.md:133
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `StringLookup` و`IntegerLookup`: القواميس صراحةً
# المصدر  : src/chapters/ch16_text_layers.md:183
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `Hashing`: قاموس بلا ذاكرة
# المصدر  : src/chapters/ch16_text_layers.md:201
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : `CategoryEncoding` و`Discretization`
# المصدر  : src/chapters/ch16_text_layers.md:219
# الترتيب : 5/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : معالجة العربية عمليًا
# المصدر  : src/chapters/ch16_text_layers.md:236
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:292
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:327
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:336
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:352
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# نصوص عربية مصغّرة
texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
])

vec = keras.layers.TextVectorization(
    max_tokens=5000,                       # سقف حجم القاموس
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="multi_hot",               # متجه presence/absence
)
vec.adapt(texts)                          # تعلّم القاموس من البيانات
print(vec.get_vocabulary())               # النتيجة المتوقعة: {'[UNK]': 0, 'منتج': 1, ...}

out = vec(texts)
print(out.shape)                          # النتيجة المتوقعة: (4, size_vocab)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# قاموس فئات: نوع المنتج
lookup = keras.layers.StringLookup(
    num_oov_indices=1,          # فهرس مخصّص لكل قيمة خارج القاموس
    oov_token="[UNK]",
    vocabulary=["electronics", "books", "clothing"],
)
print(lookup(["electronics", "unknown-item"]))
# النتيجة المتوقعة: [3, 0]   (0 هو [UNK])

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
hash_layer = keras.layers.Hashing(
    num_bins=1024,               # حجم فضاء الهاش
    salt=1,                       # يغيّر توزيع الهاش
    output_mode="multi_hot",
)
out = hash_layer(["منتج", "جودة"])
print(out.shape)                 # النتيجة المتوقعة: (2, 1024)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# ترميز الفئات
cat = keras.layers.CategoryEncoding(num_tokens=5, output_mode="one_hot")
print(cat([1, 3]))               # النتيجة المتوقعة: [[0,1,0,0,0],[0,0,0,1,0]]

# تقسيم الأعداد إلى شرائح
disc = keras.layers.Discretization(bin_boundaries=[0, 10, 20, 30], output_mode="one_hot")
print(disc([5, 15, 25]))         # النتيجة المتوقعة: one-hot لكل شريحة

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import keras
import re
import numpy as np


def normalize_arabic(text):
    """توحيد النص العربي: إزالة التشكيل، توحيد الألف والياء والتاء المربوطة."""
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)   # حذف التشكيل
    text = re.sub(r"[أإآ]", "ا", text)                     # توحيد الألف
    text = re.sub(r"ى", "ي", text)                        # توحيد الياء
    text = re.sub(r"ؤ", "و", text)                        # واو همزة
    text = re.sub(r"ئ", "ي", text)                        # ياء همزة
    text = re.sub(r"ة", "ه", text)                        # تاء مربوطة
    text = re.sub(r"\s+", " ", text).strip()
    return text


samples = np.array([normalize_arabic(t) for t in [
    "الطَّلَبُ ممتازٌ",
    "الطلب ممتاز",
]])
vec = keras.layers.TextVectorization(output_mode="multi_hot")
vec.adapt(samples)
print(vec.get_vocabulary())
# النتيجة المتوقعة: {'[UNK]': 0, 'الطلب': 1, 'ممتاز': 2}

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
# بعد التدريب: اعرف الحجم الحقيقي
vocab = model.get_layer(name="vec").get_vocabulary()
real_size = len(vocab)
print("حجم القاموس:", real_size)     # النتيجة المتوقعة: عدد أصغر من 20000

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.F1Score(average="macro", name="f1_macro"),
    keras.metrics.Precision(class_id=0, name="p_neg"),
    keras.metrics.Recall(class_id=0, name="r_neg"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-16-ch16-text-layers.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_f1_macro",     # المقياس الذي يهمّ العمل
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]
