#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — معالجة اللغة الطبيعية والعربية
Professional TensorFlow — Nlp

Book slug : tensorflow-professional-ar
Chapter   : ch-14-ch14-nlp (lesson 14)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-14-ch14-nlp.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — معالجة اللغة الطبيعية والعربية
Professional TensorFlow — Nlp

Book slug : tensorflow-professional-ar
Chapter   : ch-14-ch14-nlp (lesson 14)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch14_nlp.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

# تطبيع عربي بسيط: نزيل التشكيل والتطويل ونوحّد الألف والياء والتاء المربوطة
def normalize_arabic(text):
    text = tf.strings.regex_replace(text, "[\u064B-\u065F\u0670]", "")
    text = tf.strings.regex_replace(text, "[أإآٱ]", "ا")
    text = tf.strings.regex_replace(text, "ى", "ي")
    text = tf.strings.regex_replace(text, "ة", "ه")
    text = tf.strings.regex_replace(text, "\u0640", "")
    text = tf.strings.regex_replace(text, r"[.,!?؟،؛:\(\)\[\]\"'«»…\-]", " ")
    return tf.strings.regex_replace(text, r"\s+", " ")

corpus = [
    "الخِدمةُ مُمتازة والموظفون لُطفاء",
    "التوصيل تأخّر لكن المنتج جيّد",
    "الأسعار مرتفعة والجودة ضعيفة",
    "التطبيق سهل الاستخدام وسريع",
]

# نحوّل الجمل إلى أعداد صحيحة بطول ثابت؛ الحشو في النهاية
vectorizer = keras.layers.TextVectorization(
    max_tokens=50,
    standardize=normalize_arabic,
    output_mode="int",
    output_sequence_length=8,
)
vectorizer.adapt(corpus)

vocab = vectorizer.get_vocabulary()
print("أول عشرة رموز:", vocab[:10])
# المتوقع: ['', '[UNK]', 'الخدمه', 'ممتازه', ...] بحسب تكرار الكلمات

print("موقع [UNK]:", vocab.index("[UNK]"))   # المتوقع: 1

encoded = vectorizer(tf.constant([
    "الخِدمةُ ممتازة",   # كلمتان موجودتان في المعجم
    "الكتابُ رائع",      # كلمتان جديدتان
]))
print(encoded.numpy())
# المتوقع: مصفوفة 2×8؛ الصف الأول يبدأ بمعرّفات معروفة،
# والصف الثاني يبدأ بالرمز 1 مرتين، وبقية الأعمدة أصفار حشو

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch14_nlp.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)

# جملتان: الأولى ثلاثة رموز حقيقية، والثانية خمسة
sequences = np.array([
    [5, 9, 2, 0, 0],
    [7, 4, 11, 3, 8],
], dtype="int32")

model = keras.Sequential([
    keras.Input(shape=(5,), dtype="int32"),
    keras.layers.Embedding(input_dim=20, output_dim=4, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
])

pooled = model(sequences)
print("شكل الخرج:", pooled.shape)   # المتوقع: (2, 4)

# نقارن متوسط الجملة الأولى بمتوسط الرموز الثلاثة فقط، ثم بالخمسة مع الحشو
embeddings = model.layers[0](sequences).numpy()
mean_real = embeddings[0, :3].mean(axis=0)
mean_padded = embeddings[0].mean(axis=0)
print("تجاهل الحشو:", np.allclose(pooled.numpy()[0], mean_real, atol=1e-6))  # True
print("احتساب الحشو:", np.allclose(pooled.numpy()[0], mean_padded, atol=1e-6))  # False

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch14_nlp.md
# ────────────────────────────────────────────────────────────────────────
import keras

# pip install -U keras-hub
RUN_KERAS_HUB = False   # اجعلها True بعد التثبيت وتوفّر اتصال بالإنترنت

train_texts = [
    "الخدمة ممتازة وسريعة", "التطبيق سهل ومفيد",
    "الموظفون متعاونون ولطفاء", "المنتج وصل بحالة جيدة",
    "الأسعار مناسبة والجودة عالية", "التجربة رائعة وسأكررها",
    "الخدمة سيئة ومملة", "التطبيق بطيء ويغلق كثيرا",
    "الموظفون غير مهتمين", "المنتج تالف وغير جيد",
    "الأسعار مرتفعة والجودة ضعيفة جدا", "التجربة مخيبة ولن أكررها",
]
train_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
test_texts = ["الطلب وصل متأخرا والمنتج تالف", "الخدمة سريعة والموظفون محترمون"]
test_labels = [0, 1]

if RUN_KERAS_HUB:
    import keras_hub

    # preset متعدد اللغات (104 لغات) بنحو 178 مليون معامل؛ التنزيل مئات الميغابايتات
    classifier = keras_hub.models.BertTextClassifier.from_preset(
        "bert_base_multi",
        num_classes=2,
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        metrics=["accuracy"],
    )
    classifier.fit(
        train_texts, train_labels,
        batch_size=4, epochs=2,
        validation_data=(test_texts, test_labels),
    )
    print(classifier.predict(test_texts, batch_size=2))
else:
    print("مسار KerasHub جاهز لكنه متوقف: فعّله بعد تثبيت keras-hub وتنزيل الأوزان.")
    # المتوقع عند التعطيل: رسالة توضيحية فقط، بلا أي تنزيل

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch14_nlp.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

def normalize_arabic(text):
    text = tf.strings.regex_replace(text, "[\u064B-\u065F\u0670]", "")
    text = tf.strings.regex_replace(text, "[أإآٱ]", "ا")
    text = tf.strings.regex_replace(text, "ى", "ي")
    text = tf.strings.regex_replace(text, "ة", "ه")
    text = tf.strings.regex_replace(text, "\u0640", "")
    text = tf.strings.regex_replace(text, r"[.,!?؟،؛:\(\)\[\]\"'«»…\-]", " ")
    return tf.strings.regex_replace(text, r"\s+", " ")

# 24 زوجًا: جملة موجبة تليها جملة سالبة، ليبقى التقسيم متوازنًا
pairs = [
    ("الخدمة كانت ممتازة والموظفون لطفاء جدا", 1), ("الخدمة سيئة والموظفون غير مهتمين", 0),
    ("أعجبني المنتج وجودته عالية", 1), ("المنتج تالف وجودته منخفضة", 0),
    ("تجربة رائعة وسأكررها بكل تأكيد", 1), ("تجربة مخيبة ولن أكررها", 0),
    ("التوصيل كان سريعا والطلب وصل بحالة جيدة", 1), ("التوصيل تأخر كثيرا والطلب وصل متأخرا", 0),
    ("الفيلم ممتع والقصة مشوقة", 1), ("الفيلم ممل والقصة ضعيفة", 0),
    ("الطعام لذيذ والمطعم نظيف", 1), ("الطعام بارد والمطعم غير نظيف", 0),
    ("التطبيق سهل الاستخدام وسريع", 1), ("التطبيق بطيء ويغلق كثيرا", 0),
    ("الرحلة كانت مريحة والطاقم محترم", 1), ("الرحلة كانت متعبة والطاقم غير محترم", 0),
    ("أنصح الجميع بهذا الكتاب المفيد", 1), ("لا أنصح بهذا الكتاب الممل", 0),
    ("الغرفة واسعة والإطلالة جميلة", 1), ("الغرفة ضيقة والإطلالة سيئة", 0),
    ("المعلم يشرح بوضوح ويجيب عن الأسئلة", 1), ("المعلم لا يشرح بوضوح ولا يجيب عن الأسئلة", 0),
    ("شعرت بالسعادة بعد قراءة الرسالة", 1), ("شعرت بالحزن بعد قراءة الرسالة", 0),
]

train_pairs, test_pairs = pairs[:20], pairs[20:]   # 40 تدريبًا و8 اختبارًا
train_texts = [p[0] for p in train_pairs]
test_texts = [p[0] for p in test_pairs]
train_labels = np.array([p[1] for p in train_pairs], dtype="float32")
test_labels = np.array([p[1] for p in test_pairs], dtype="float32")

# المعجم يُبنى من التدريب فقط، لا من الاختبار
vectorizer = keras.layers.TextVectorization(
    max_tokens=300,
    standardize=normalize_arabic,
    output_mode="int",
    output_sequence_length=10,
)
vectorizer.adapt(train_texts)
print("حجم المعجم:", len(vectorizer.get_vocabulary()))   # المتوقع: نحو 100 رمز

model = keras.Sequential([
    keras.Input(shape=(1,), dtype="string"),
    vectorizer,
    keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=16,
        mask_zero=True,
    ),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
)

history = model.fit(
    train_texts, train_labels,
    validation_data=(test_texts, test_labels),
    epochs=40, batch_size=8, verbose=0,
)

test_prob = model.predict(tf.constant(test_texts), verbose=0).ravel()
test_pred = (test_prob >= 0.5).astype("int32")
tp = int(np.sum((test_pred == 1) & (test_labels == 1)))
fp = int(np.sum((test_pred == 1) & (test_labels == 0)))
fn = int(np.sum((test_pred == 0) & (test_labels == 1)))
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
print(f"دقة الاختبار: {(test_pred == test_labels).mean():.2f}")   # غالبًا بين 0.75 و1.00
print(f"الدقة الموجبة: {precision:.2f} | الاستدعاء: {recall:.2f}")

samples = ["الخدمة رائعة والموظفون متعاونون", "الطلب تأخر والمنتج تالف"]
probs = model.predict(tf.constant(samples), verbose=0).ravel()
for text, p in zip(samples, probs):
    print(f"{text} → احتمال الإيجاب: {p:.2f}")
# المتوقع: الجملة الأولى فوق 0.5 والثانية تحته؛ القيم تتغيّر مع البذرة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-14-ch14-nlp.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

# تطبيع عربي بسيط: نزيل التشكيل والتطويل ونوحّد الألف والياء والتاء المربوطة
def normalize_arabic(text):
    text = tf.strings.regex_replace(text, "[\u064B-\u065F\u0670]", "")
    text = tf.strings.regex_replace(text, "[أإآٱ]", "ا")
    text = tf.strings.regex_replace(text, "ى", "ي")
    text = tf.strings.regex_replace(text, "ة", "ه")
    text = tf.strings.regex_replace(text, "\u0640", "")
    text = tf.strings.regex_replace(text, r"[.,!?؟،؛:\(\)\[\]\"'«»…\-]", " ")
    return tf.strings.regex_replace(text, r"\s+", " ")

corpus = [
    "الخِدمةُ مُمتازة والموظفون لُطفاء",
    "التوصيل تأخّر لكن المنتج جيّد",
    "الأسعار مرتفعة والجودة ضعيفة",
    "التطبيق سهل الاستخدام وسريع",
]

# نحوّل الجمل إلى أعداد صحيحة بطول ثابت؛ الحشو في النهاية
vectorizer = keras.layers.TextVectorization(
    max_tokens=50,
    standardize=normalize_arabic,
    output_mode="int",
    output_sequence_length=8,
)
vectorizer.adapt(corpus)

vocab = vectorizer.get_vocabulary()
print("أول عشرة رموز:", vocab[:10])
# المتوقع: ['', '[UNK]', 'الخدمه', 'ممتازه', ...] بحسب تكرار الكلمات

print("موقع [UNK]:", vocab.index("[UNK]"))   # المتوقع: 1

encoded = vectorizer(tf.constant([
    "الخِدمةُ ممتازة",   # كلمتان موجودتان في المعجم
    "الكتابُ رائع",      # كلمتان جديدتان
]))
print(encoded.numpy())
# المتوقع: مصفوفة 2×8؛ الصف الأول يبدأ بمعرّفات معروفة،
# والصف الثاني يبدأ بالرمز 1 مرتين، وبقية الأعمدة أصفار حشو

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-14-ch14-nlp.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)

# جملتان: الأولى ثلاثة رموز حقيقية، والثانية خمسة
sequences = np.array([
    [5, 9, 2, 0, 0],
    [7, 4, 11, 3, 8],
], dtype="int32")

model = keras.Sequential([
    keras.Input(shape=(5,), dtype="int32"),
    keras.layers.Embedding(input_dim=20, output_dim=4, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
])

pooled = model(sequences)
print("شكل الخرج:", pooled.shape)   # المتوقع: (2, 4)

# نقارن متوسط الجملة الأولى بمتوسط الرموز الثلاثة فقط، ثم بالخمسة مع الحشو
embeddings = model.layers[0](sequences).numpy()
mean_real = embeddings[0, :3].mean(axis=0)
mean_padded = embeddings[0].mean(axis=0)
print("تجاهل الحشو:", np.allclose(pooled.numpy()[0], mean_real, atol=1e-6))  # True
print("احتساب الحشو:", np.allclose(pooled.numpy()[0], mean_padded, atol=1e-6))  # False

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-14-ch14-nlp.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras

# pip install -U keras-hub
RUN_KERAS_HUB = False   # اجعلها True بعد التثبيت وتوفّر اتصال بالإنترنت

train_texts = [
    "الخدمة ممتازة وسريعة", "التطبيق سهل ومفيد",
    "الموظفون متعاونون ولطفاء", "المنتج وصل بحالة جيدة",
    "الأسعار مناسبة والجودة عالية", "التجربة رائعة وسأكررها",
    "الخدمة سيئة ومملة", "التطبيق بطيء ويغلق كثيرا",
    "الموظفون غير مهتمين", "المنتج تالف وغير جيد",
    "الأسعار مرتفعة والجودة ضعيفة جدا", "التجربة مخيبة ولن أكررها",
]
train_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
test_texts = ["الطلب وصل متأخرا والمنتج تالف", "الخدمة سريعة والموظفون محترمون"]
test_labels = [0, 1]

if RUN_KERAS_HUB:
    import keras_hub

    # preset متعدد اللغات (104 لغات) بنحو 178 مليون معامل؛ التنزيل مئات الميغابايتات
    classifier = keras_hub.models.BertTextClassifier.from_preset(
        "bert_base_multi",
        num_classes=2,
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        metrics=["accuracy"],
    )
    classifier.fit(
        train_texts, train_labels,
        batch_size=4, epochs=2,
        validation_data=(test_texts, test_labels),
    )
    print(classifier.predict(test_texts, batch_size=2))
else:
    print("مسار KerasHub جاهز لكنه متوقف: فعّله بعد تثبيت keras-hub وتنزيل الأوزان.")
    # المتوقع عند التعطيل: رسالة توضيحية فقط، بلا أي تنزيل

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-14-ch14-nlp.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)

def normalize_arabic(text):
    text = tf.strings.regex_replace(text, "[\u064B-\u065F\u0670]", "")
    text = tf.strings.regex_replace(text, "[أإآٱ]", "ا")
    text = tf.strings.regex_replace(text, "ى", "ي")
    text = tf.strings.regex_replace(text, "ة", "ه")
    text = tf.strings.regex_replace(text, "\u0640", "")
    text = tf.strings.regex_replace(text, r"[.,!?؟،؛:\(\)\[\]\"'«»…\-]", " ")
    return tf.strings.regex_replace(text, r"\s+", " ")

# 24 زوجًا: جملة موجبة تليها جملة سالبة، ليبقى التقسيم متوازنًا
pairs = [
    ("الخدمة كانت ممتازة والموظفون لطفاء جدا", 1), ("الخدمة سيئة والموظفون غير مهتمين", 0),
    ("أعجبني المنتج وجودته عالية", 1), ("المنتج تالف وجودته منخفضة", 0),
    ("تجربة رائعة وسأكررها بكل تأكيد", 1), ("تجربة مخيبة ولن أكررها", 0),
    ("التوصيل كان سريعا والطلب وصل بحالة جيدة", 1), ("التوصيل تأخر كثيرا والطلب وصل متأخرا", 0),
    ("الفيلم ممتع والقصة مشوقة", 1), ("الفيلم ممل والقصة ضعيفة", 0),
    ("الطعام لذيذ والمطعم نظيف", 1), ("الطعام بارد والمطعم غير نظيف", 0),
    ("التطبيق سهل الاستخدام وسريع", 1), ("التطبيق بطيء ويغلق كثيرا", 0),
    ("الرحلة كانت مريحة والطاقم محترم", 1), ("الرحلة كانت متعبة والطاقم غير محترم", 0),
    ("أنصح الجميع بهذا الكتاب المفيد", 1), ("لا أنصح بهذا الكتاب الممل", 0),
    ("الغرفة واسعة والإطلالة جميلة", 1), ("الغرفة ضيقة والإطلالة سيئة", 0),
    ("المعلم يشرح بوضوح ويجيب عن الأسئلة", 1), ("المعلم لا يشرح بوضوح ولا يجيب عن الأسئلة", 0),
    ("شعرت بالسعادة بعد قراءة الرسالة", 1), ("شعرت بالحزن بعد قراءة الرسالة", 0),
]

train_pairs, test_pairs = pairs[:20], pairs[20:]   # 40 تدريبًا و8 اختبارًا
train_texts = [p[0] for p in train_pairs]
test_texts = [p[0] for p in test_pairs]
train_labels = np.array([p[1] for p in train_pairs], dtype="float32")
test_labels = np.array([p[1] for p in test_pairs], dtype="float32")

# المعجم يُبنى من التدريب فقط، لا من الاختبار
vectorizer = keras.layers.TextVectorization(
    max_tokens=300,
    standardize=normalize_arabic,
    output_mode="int",
    output_sequence_length=10,
)
vectorizer.adapt(train_texts)
print("حجم المعجم:", len(vectorizer.get_vocabulary()))   # المتوقع: نحو 100 رمز

model = keras.Sequential([
    keras.Input(shape=(1,), dtype="string"),
    vectorizer,
    keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=16,
        mask_zero=True,
    ),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
)

history = model.fit(
    train_texts, train_labels,
    validation_data=(test_texts, test_labels),
    epochs=40, batch_size=8, verbose=0,
)

test_prob = model.predict(tf.constant(test_texts), verbose=0).ravel()
test_pred = (test_prob >= 0.5).astype("int32")
tp = int(np.sum((test_pred == 1) & (test_labels == 1)))
fp = int(np.sum((test_pred == 1) & (test_labels == 0)))
fn = int(np.sum((test_pred == 0) & (test_labels == 1)))
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
print(f"دقة الاختبار: {(test_pred == test_labels).mean():.2f}")   # غالبًا بين 0.75 و1.00
print(f"الدقة الموجبة: {precision:.2f} | الاستدعاء: {recall:.2f}")

samples = ["الخدمة رائعة والموظفون متعاونون", "الطلب تأخر والمنتج تالف"]
probs = model.predict(tf.constant(samples), verbose=0).ravel()
for text, p in zip(samples, probs):
    print(f"{text} → احتمال الإيجاب: {p:.2f}")
# المتوقع: الجملة الأولى فوق 0.5 والثانية تحته؛ القيم تتغيّر مع البذرة
