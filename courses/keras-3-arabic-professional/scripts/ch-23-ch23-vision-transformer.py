#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — محوّل الرؤية (Vision Transformer)
Keras 3 in Arabic — Vision Transformer

Book slug : keras-3-arabic-professional
Chapter   : ch-23-ch23-vision-transformer (lesson 23)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-23-ch23-vision-transformer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — محوّل الرؤية (Vision Transformer)
Keras 3 in Arabic — Vision Transformer

Book slug : keras-3-arabic-professional
Chapter   : ch-23-ch23-vision-transformer (lesson 23)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-23-ch23-vision-transformer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — محوّل الرؤية (Vision Transformer)
Keras 3 in Arabic — Vision Transformer

Book slug : keras-3-arabic-professional
Chapter   : ch-23-ch23-vision-transformer (lesson 23)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch23_vision_transformer/001_s23_5_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : 23.5 التنفيذ في Keras
# المصدر  : src/chapters/ch23_vision_transformer.md:61
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch23_vision_transformer/002_compute_patches_attention.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : حساب عدد الرقع وكلفة الانتباه
# المصدر  : src/chapters/ch23_vision_transformer.md:153
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch23_vision_transformer.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch23_vision_transformer.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : 23.5 التنفيذ في Keras
# المصدر  : src/chapters/ch23_vision_transformer.md:61
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : حساب عدد الرقع وكلفة الانتباه
# المصدر  : src/chapters/ch23_vision_transformer.md:153
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — محوّل الرؤية (Vision Transformer)
Keras 3 in Arabic — Vision Transformer

Book slug : keras-3-arabic-professional
Chapter   : ch-23-ch23-vision-transformer (lesson 23)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch23_vision_transformer/001_s23_5_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : 23.5 التنفيذ في Keras
# المصدر  : src/chapters/ch23_vision_transformer.md:61
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch23_vision_transformer/002_compute_patches_attention.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : حساب عدد الرقع وكلفة الانتباه
# المصدر  : src/chapters/ch23_vision_transformer.md:153
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch23_vision_transformer.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch23_vision_transformer.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : 23.5 التنفيذ في Keras
# المصدر  : src/chapters/ch23_vision_transformer.md:61
# الترتيب : 1/2
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 23: محوّل الرؤية (Vision Transformer)
# القسم   : حساب عدد الرقع وكلفة الانتباه
# المصدر  : src/chapters/ch23_vision_transformer.md:153
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة لتشغيل سريع
PATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64 رقعة
PROJECTION_DIM = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
NUM_CLASSES = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, projection_dim, dropout=0.1):
    """كتلة محوّل: انتباه متعدّد الرؤوس + MLP مع اتصال تخطّي وتطبيع."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(projection_dim * 2, activation="gelu")(y)
    y = keras.layers.Dense(projection_dim)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج الكامل (تقطيع الرقع عبر Conv2D بخطوة = حجم الرقعة)
inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.layers.Rescaling(1.0 / 255.0)(inp)
# تقطيع الرقع: التفاف بحجم الرقعة وخطوة الرقعة
x = keras.layers.Conv2D(PROJECTION_DIM, PATCH_SIZE, strides=PATCH_SIZE,
                        padding="valid")(x)
# (B, 8, 8, projection_dim) → (B, 64, projection_dim)
x = keras.layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(x)
# ترميز موضعي قابل للتعلّم
pos = keras.layers.Embedding(NUM_PATCHES, PROJECTION_DIM)(
    keras.ops.arange(NUM_PATCHES))
x = x + pos
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, PROJECTION_DIM)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 32, 32, 3).astype("float32") * 255.0
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
# النتيجة المتوقعة: الدقّة تقترب من 0.55–0.65 على بيانات عشوائية،
# وهذا طبيعي — الرقع لا تحمل نمطًا حقيقيًا.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-23-ch23-vision-transformer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def vit_cost(image_size, patch_size, projection_dim, num_heads, num_blocks):
    """يحسب عدد الرقع وكلفة الانتباه وعدد المعاملات التقريبيّ."""
    num_patches = (image_size // patch_size) ** 2
    attn_ops = num_patches ** 2                        # كلفة الانتباه O(N²)
    # معاملات كتلة واحدة: انتباه (4 مصفوفات) + MLP (توسيع ×4)
    attn_params = 4 * projection_dim ** 2
    mlp_params = 2 * projection_dim * 4 * projection_dim
    total_params = num_blocks * (attn_params + mlp_params)
    return {
        "num_patches": num_patches,
        "attn_ops_per_head_pair": attn_ops,
        "approx_params": total_params,
        "key_dim": projection_dim // num_heads,
    }

# ─── 1. إعداد صغير (مثال هذا الفصل)
print("صغير:", vit_cost(32, 4, 64, 4, 2))
# ─── 2. إعداد ViT-Base
print("Base:", vit_cost(224, 16, 768, 12, 12))
# النتيجة المتوقعة: الصغير 64 رقعة وكلفة 4096؛
# الـ Base 196 رقعة وكلفة 38,416 — أي 9.4 ضعفًا.
