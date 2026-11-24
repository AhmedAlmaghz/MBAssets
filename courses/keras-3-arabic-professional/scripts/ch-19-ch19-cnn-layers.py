#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — طبقات الشبكة التلافيفية في Keras
Keras 3 in Arabic — Cnn Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-19-ch19-cnn-layers (lesson 19)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-19-ch19-cnn-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — طبقات الشبكة التلافيفية في Keras
Keras 3 in Arabic — Cnn Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-19-ch19-cnn-layers (lesson 19)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-19-ch19-cnn-layers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — طبقات الشبكة التلافيفية في Keras
Keras 3 in Arabic — Cnn Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-19-ch19-cnn-layers (lesson 19)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch19_cnn_layers/001_s19_5_keras_cnn_block_standard.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.5 التنفيذ في Keras: كتلة CNN معيارية
# المصدر  : src/chapters/ch19_cnn_layers.md:127
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch19_cnn_layers/003_s19_7_example_solved_classification_parts.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.7 مثال محلول: تصنيف قطع المعدّات
# المصدر  : src/chapters/ch19_cnn_layers.md:236
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch19_cnn_layers/004_compute_parameters_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : حساب البارامترات يدويًّا والتحقّق
# المصدر  : src/chapters/ch19_cnn_layers.md:286
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch19_cnn_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch19_cnn_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch19_cnn_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.5 التنفيذ في Keras: كتلة CNN معيارية
# المصدر  : src/chapters/ch19_cnn_layers.md:127
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.7 مثال محلول: تصنيف قطع المعدّات
# المصدر  : src/chapters/ch19_cnn_layers.md:236
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : حساب البارامترات يدويًّا والتحقّق
# المصدر  : src/chapters/ch19_cnn_layers.md:286
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — طبقات الشبكة التلافيفية في Keras
Keras 3 in Arabic — Cnn Layers

Book slug : keras-3-arabic-professional
Chapter   : ch-19-ch19-cnn-layers (lesson 19)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch19_cnn_layers/001_s19_5_keras_cnn_block_standard.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.5 التنفيذ في Keras: كتلة CNN معيارية
# المصدر  : src/chapters/ch19_cnn_layers.md:127
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch19_cnn_layers/003_s19_7_example_solved_classification_parts.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.7 مثال محلول: تصنيف قطع المعدّات
# المصدر  : src/chapters/ch19_cnn_layers.md:236
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch19_cnn_layers/004_compute_parameters_manual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : حساب البارامترات يدويًّا والتحقّق
# المصدر  : src/chapters/ch19_cnn_layers.md:286
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch19_cnn_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch19_cnn_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch19_cnn_layers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.5 التنفيذ في Keras: كتلة CNN معيارية
# المصدر  : src/chapters/ch19_cnn_layers.md:127
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.7 مثال محلول: تصنيف قطع المعدّات
# المصدر  : src/chapters/ch19_cnn_layers.md:236
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : حساب البارامترات يدويًّا والتحقّق
# المصدر  : src/chapters/ch19_cnn_layers.md:286
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-19-ch19-cnn-layers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. طبقة Conv2D واحدة: k²·C_in·C_out + C_out
inp = keras.Input((32, 32, 3))
conv = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)
m1 = keras.Model(inp, conv(inp))
manual = 3 * 3 * 3 * 64
actual = m1.layers[-1].count_params()
print(f"manual = {manual}, actual = {actual}, equal = {manual == actual}")
# النتيجة المتوقعة: 1728 == 1728 -> True

# ─── 2. SeparableConv2D: depthwise + pointwise
sep = keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)
m2 = keras.Model(inp, sep(inp))
depthwise = 3 * 3 * 3            # مرشّح لكل قناة مدخل
pointwise = 3 * 64               # 1×1 يخلط القنوات
manual_sep = depthwise + pointwise
actual_sep = m2.layers[-1].count_params()
print(f"manual_sep = {manual_sep}, actual = {actual_sep}")
# النتيجة المتوقعة: 219 == 219 (أقلّ من 1728 بـ 7.9×)

# ─── 3. DepthwiseConv2D
dw = keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
m3 = keras.Model(inp, dw(inp))
manual_dw = 3 * 3 * 3
print(f"manual_dw = {manual_dw}, actual = {m3.layers[-1].count_params()}")
# النتيجة المتوقعة: 27 == 27 — رخيص جدًّا لكنه لا يخلط القنوات.
