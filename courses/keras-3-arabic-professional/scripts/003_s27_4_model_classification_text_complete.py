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
