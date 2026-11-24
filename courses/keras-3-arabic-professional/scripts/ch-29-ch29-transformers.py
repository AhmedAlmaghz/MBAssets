#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المحوّلات وآلية الانتباه
Keras 3 in Arabic — Transformers

Book slug : keras-3-arabic-professional
Chapter   : ch-29-ch29-transformers (lesson 29)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-29-ch29-transformers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المحوّلات وآلية الانتباه
Keras 3 in Arabic — Transformers

Book slug : keras-3-arabic-professional
Chapter   : ch-29-ch29-transformers (lesson 29)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-29-ch29-transformers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المحوّلات وآلية الانتباه
Keras 3 in Arabic — Transformers

Book slug : keras-3-arabic-professional
Chapter   : ch-29-ch29-transformers (lesson 29)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch29_transformers/001_s29_4_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.4 التنفيذ في Keras
# المصدر  : src/chapters/ch29_transformers.md:65
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch29_transformers/002_s29_5_causal_masking_mask.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.5 القناع السببيّ (Causal Masking)
# المصدر  : src/chapters/ch29_transformers.md:128
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch29_transformers/003_mask_causal.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : القناع السببيّ عمليًّا
# المصدر  : src/chapters/ch29_transformers.md:173
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch29_transformers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch29_transformers.md
# ────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch29_transformers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.4 التنفيذ في Keras
# المصدر  : src/chapters/ch29_transformers.md:65
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.5 القناع السببيّ (Causal Masking)
# المصدر  : src/chapters/ch29_transformers.md:128
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : القناع السببيّ عمليًّا
# المصدر  : src/chapters/ch29_transformers.md:173
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — المحوّلات وآلية الانتباه
Keras 3 in Arabic — Transformers

Book slug : keras-3-arabic-professional
Chapter   : ch-29-ch29-transformers (lesson 29)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch29_transformers/001_s29_4_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.4 التنفيذ في Keras
# المصدر  : src/chapters/ch29_transformers.md:65
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch29_transformers/002_s29_5_causal_masking_mask.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.5 القناع السببيّ (Causal Masking)
# المصدر  : src/chapters/ch29_transformers.md:128
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch29_transformers/003_mask_causal.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : القناع السببيّ عمليًّا
# المصدر  : src/chapters/ch29_transformers.md:173
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch29_transformers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch29_transformers.md
# ────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch29_transformers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.4 التنفيذ في Keras
# المصدر  : src/chapters/ch29_transformers.md:65
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.5 القناع السببيّ (Causal Masking)
# المصدر  : src/chapters/ch29_transformers.md:128
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : القناع السببيّ عمليًّا
# المصدر  : src/chapters/ch29_transformers.md:173
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# في Keras 3، القناع يُمرَّر عبر use_causal_mask=True
x = keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=16)(query, value, use_causal_mask=True)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-29-ch29-transformers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. انتباه بلا قناع: كلّ رمز يرى كلّ الرموز
L, D = 6, 32
x = np.random.randn(1, L, D).astype("float32")
mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)
out_open = mha(x, x, use_causal_mask=False)

# ─── 2. انتباه بقناع سببيّ: الرمز i يرى فقط الرموز ≤ i
out_masked = mha(x, x, use_causal_mask=True)

# ─── 3. التحقّق: أول رمز لا يتغيّر (لا يرى المستقبل)
print("أول رمز بلا قناع :", np.round(keras.ops.convert_to_numpy(out_open)[0, 0, :3], 3))
print("أول رمز بقناع    :", np.round(keras.ops.convert_to_numpy(out_masked)[0, 0, :3], 3))
# النتيجة المتوقعة: في القناع السببيّ، مُخرَج الرمز الأول يعتمد على نفسه فقط
# — وهذا شرط صحيح في فكّ الترميز لتوليد النصّ.
