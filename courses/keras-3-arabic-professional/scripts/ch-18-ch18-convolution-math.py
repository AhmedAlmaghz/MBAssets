#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الرياضيات خلف الالتفاف
Keras 3 in Arabic — Convolution Math

Book slug : keras-3-arabic-professional
Chapter   : ch-18-ch18-convolution-math (lesson 18)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-18-ch18-convolution-math.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الرياضيات خلف الالتفاف
Keras 3 in Arabic — Convolution Math

Book slug : keras-3-arabic-professional
Chapter   : ch-18-ch18-convolution-math (lesson 18)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-18-ch18-convolution-math.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الرياضيات خلف الالتفاف
Keras 3 in Arabic — Convolution Math

Book slug : keras-3-arabic-professional
Chapter   : ch-18-ch18-convolution-math (lesson 18)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch18_convolution_math/001_validation_convolution.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : التحقّق اليدوي من عملية الالتفاف
# المصدر  : src/chapters/ch18_convolution_math.md:229
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch18_convolution_math/002_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب أشكال المخرجات يدويًا والتحقّق منها
# المصدر  : src/chapters/ch18_convolution_math.md:277
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch18_convolution_math/003_compute_parameters_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب البارامترات يدويًا والتحقّق
# المصدر  : src/chapters/ch18_convolution_math.md:337
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch18_convolution_math.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch18_convolution_math.md
# ────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch18_convolution_math.md
# ────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : التحقّق اليدوي من عملية الالتفاف
# المصدر  : src/chapters/ch18_convolution_math.md:229
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب أشكال المخرجات يدويًا والتحقّق منها
# المصدر  : src/chapters/ch18_convolution_math.md:277
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب البارامترات يدويًا والتحقّق
# المصدر  : src/chapters/ch18_convolution_math.md:337
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الرياضيات خلف الالتفاف
Keras 3 in Arabic — Convolution Math

Book slug : keras-3-arabic-professional
Chapter   : ch-18-ch18-convolution-math (lesson 18)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch18_convolution_math/001_validation_convolution.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : التحقّق اليدوي من عملية الالتفاف
# المصدر  : src/chapters/ch18_convolution_math.md:229
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch18_convolution_math/002_compute_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب أشكال المخرجات يدويًا والتحقّق منها
# المصدر  : src/chapters/ch18_convolution_math.md:277
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch18_convolution_math/003_compute_parameters_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب البارامترات يدويًا والتحقّق
# المصدر  : src/chapters/ch18_convolution_math.md:337
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch18_convolution_math.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch18_convolution_math.md
# ────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch18_convolution_math.md
# ────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : التحقّق اليدوي من عملية الالتفاف
# المصدر  : src/chapters/ch18_convolution_math.md:229
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب أشكال المخرجات يدويًا والتحقّق منها
# المصدر  : src/chapters/ch18_convolution_math.md:277
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 18: الرياضيات خلف الالتفاف
# القسم   : حساب البارامترات يدويًا والتحقّق
# المصدر  : src/chapters/ch18_convolution_math.md:337
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# صورة صغيرة 5×5 بقناة واحدة، بقيم معروفة (0..24)
x = np.arange(25, dtype="float32").reshape(1, 5, 5, 1)

# مرشّح بأوزان الكل واحد وحيّز = 0 (أي مرشّح كشف الفروق)
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=3,
    padding="valid",
    use_bias=True,
    kernel_initializer="ones",
    bias_initializer="zeros",
)
y_official = keras.ops.convert_to_numpy(layer(x))  # الناتج الرسمي من Keras

# --- التنفيذ اليدوي عبر unfold ---
# 1) Keras 3 يعمل بترتيب channels_last، لكن unfold يتوقع channels_first: (N, C, H, W)
xt = keras.ops.transpose(keras.ops.convert_to_tensor(x), (0, 3, 1, 2))

# 2) تحوّل إلى مصفوفة أعمدة: كل عمود = نافذة 3×3 مسطّحة (9 عناصر)
cols = keras.ops.nn.unfold(xt, kernel_size=3, padding=0, stride=1)  # (1, 9, 9)

# 3) جدول الأوزان: 9 مواقع × قناة خرج واحدة
k = keras.ops.reshape(layer.kernel, (9, 1))  # (9, 1)

# 4) كل عمود من cols يُضرب في k ثم يُجمع -> (1, 9, 1)
manual = keras.ops.einsum("nkl,kf->nfl", cols, k)

# 5) نضيف التحييز ونعيد الترتيب إلى (N, H_out, W_out, C_out)
manual = manual + layer.bias
manual = keras.ops.reshape(keras.ops.transpose(manual, (0, 2, 1)), (1, 3, 3, 1))

print("الشكل الرسمي:", y_official.shape)
print("الشكل اليدوي:", manual.shape)
manual_np = keras.ops.convert_to_numpy(manual)
print("أقصى فرق:", float(np.max(np.abs(y_official - manual_np))))

# النتيجة المتوقعة:
# الشكل الرسمي: (1, 3, 3, 1)
# الشكل اليدوي: (1, 3, 3, 1)
# أقصى فرق: 0.0

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

def conv_out(n_in, k, p, s, d=1):
    """صيغة الخرج أحادية البعد مع التمدّد."""
    k_eff = k + (k - 1) * (d - 1)          # الحجم الفعّال للنواة
    return (n_in + 2 * p - k_eff) // s + 1  # التقريب لأسفل

# حزمة طبقات: كل عنصر (نوع, معاملات)
stack = [
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("Conv2D", dict(filters=32, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=2)),
    ("Conv2D", dict(filters=64, kernel_size=3, padding="same", strides=1)),
    ("MaxPooling2D", dict(pool_size=2, strides=2)),
    ("Conv2D", dict(filters=128, kernel_size=3, padding="same", strides=2)),
]

# (1) بناء النموذج فعليًا في Keras
inp = keras.layers.Input((32, 32, 3))
x = inp
for kind, kw in stack:
    x = getattr(keras.layers, kind)(**kw)(x)
model = keras.Model(inp, x)

# (2) المرور اليدوي على الأشكال
h = w = c = 32
print(f"{'الطبقة':<14}{'الشكل اليدوي':<22}{'الشكل الفعلي'}")
print("-" * 62)
for kind, kw in stack:
    if kind == "Conv2D":
        h = conv_out(h, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        w = conv_out(w, kw["kernel_size"], 1 if kw["padding"] == "same" else 0, kw["strides"])
        c = kw["filters"]
    else:
        h = conv_out(h, kw["pool_size"], 0, kw["strides"])
        w = conv_out(w, kw["pool_size"], 0, kw["strides"])
    actual = model.get_layer(
        f"{kind.lower()}_{next(i for i, l in enumerate(model.layers) if l.name.startswith(kind.lower()) and False)}"
    ) if False else None  # نعرض الشكل من الكود أدناه
    print(f"{kind:<14}{str((h, w, c)):<22}")

print("\nأشكال الطبقات كما يقرأها Keras:")
for layer in model.layers:
    if hasattr(layer, "output_shape"):
        print(f"  {layer.name:<28}{layer.output_shape}")

# النتيجة المتوقعة (أشكال Keras):
# conv2d (None, 32, 32, 32)
# conv2d_1 (None, 32, 32, 32)
# max_pooling2d (None, 16, 16, 32)
# conv2d_2 (None, 8, 8, 64)
# conv2d_3 (None, 8, 8, 64)
# max_pooling2d_1 (None, 4, 4, 64)
# conv2d_4 (None, 2, 2, 128)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-18-ch18-convolution-math.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# صورة 224×224×3
inputs = keras.layers.Input((224, 224, 3))
x = inputs
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.BatchNormalization(name="bn")(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
model = keras.Model(inputs, x)

# (1) الحساب اليدوي
manual = 3 * 3 * 3 * 64 + 64      # الطبقة الأولى: 1792
manual += 3 * 3 * 64 * 128 + 128   # الطبقة الثانية: 73856

# (2) ما تقوله Keras
total = sum(l.count_params() for l in model.layers)
bn = model.get_layer("bn").count_params()

print("بارامترات الطبقات التلافيفية (يدوي):", manual)
print("إجمالي النموذج (مع BN):", total)
print("بارامترات BN:", bn)
print("المحصّلة:", manual + bn, "==", total, "->", manual + bn == total)

# النتيجة المتوقعة:
# بارامترات الطبقات التلافيفية (يدوي): 75648
# إجمالي النموذج (مع BN): 75904
# بارامترات BN: 256
# المحصّلة: 75904 == 75904 -> True
