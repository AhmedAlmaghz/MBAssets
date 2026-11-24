#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأخطاء الشائعة وكيف تُصلحها
Keras 3 in Arabic — Common Errors

Book slug : keras-3-arabic-professional
Chapter   : ch-40-ch40-common-errors (lesson 40)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-40-ch40-common-errors.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأخطاء الشائعة وكيف تُصلحها
Keras 3 in Arabic — Common Errors

Book slug : keras-3-arabic-professional
Chapter   : ch-40-ch40-common-errors (lesson 40)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-40-ch40-common-errors.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأخطاء الشائعة وكيف تُصلحها
Keras 3 in Arabic — Common Errors

Book slug : keras-3-arabic-professional
Chapter   : ch-40-ch40-common-errors (lesson 40)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch40_common_errors/002_error_shape_input_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 1 — شكل مُدخل لا يطابق
# المصدر  : src/chapters/ch40_common_errors.md:39
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch40_common_errors/004_flatten_globalaveragepooling_error_before.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 2 — Flatten قبل GlobalAveragePooling
# المصدر  : src/chapters/ch40_common_errors.md:61
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch40_common_errors/006_error_rate_large.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 3 — معدّل تعلّم كبير
# المصدر  : src/chapters/ch40_common_errors.md:86
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch40_common_errors/007_nan_error_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 4 — بيانات فيها NaN
# المصدر  : src/chapters/ch40_common_errors.md:101
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch40_common_errors/008_error_different.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 5 — ميزات بمقياس مختلف
# المصدر  : src/chapters/ch40_common_errors.md:119
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch40_common_errors/010_error_layer_custom_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 7 — طبقة مخصّصة لا تُحمَّل
# المصدر  : src/chapters/ch40_common_errors.md:152
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 1 — شكل مُدخل لا يطابق
# المصدر  : src/chapters/ch40_common_errors.md:39
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 2 — Flatten قبل GlobalAveragePooling
# المصدر  : src/chapters/ch40_common_errors.md:61
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 3 — معدّل تعلّم كبير
# المصدر  : src/chapters/ch40_common_errors.md:86
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 4 — بيانات فيها NaN
# المصدر  : src/chapters/ch40_common_errors.md:101
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 5 — ميزات بمقياس مختلف
# المصدر  : src/chapters/ch40_common_errors.md:119
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 7 — طبقة مخصّصة لا تُحمَّل
# المصدر  : src/chapters/ch40_common_errors.md:152
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الأخطاء الشائعة وكيف تُصلحها
Keras 3 in Arabic — Common Errors

Book slug : keras-3-arabic-professional
Chapter   : ch-40-ch40-common-errors (lesson 40)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch40_common_errors/002_error_shape_input_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 1 — شكل مُدخل لا يطابق
# المصدر  : src/chapters/ch40_common_errors.md:39
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch40_common_errors/004_flatten_globalaveragepooling_error_before.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 2 — Flatten قبل GlobalAveragePooling
# المصدر  : src/chapters/ch40_common_errors.md:61
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch40_common_errors/006_error_rate_large.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 3 — معدّل تعلّم كبير
# المصدر  : src/chapters/ch40_common_errors.md:86
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch40_common_errors/007_nan_error_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 4 — بيانات فيها NaN
# المصدر  : src/chapters/ch40_common_errors.md:101
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch40_common_errors/008_error_different.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 5 — ميزات بمقياس مختلف
# المصدر  : src/chapters/ch40_common_errors.md:119
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch40_common_errors/010_error_layer_custom_no.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 7 — طبقة مخصّصة لا تُحمَّل
# المصدر  : src/chapters/ch40_common_errors.md:152
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch40_common_errors.md
# ────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 1 — شكل مُدخل لا يطابق
# المصدر  : src/chapters/ch40_common_errors.md:39
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 2 — Flatten قبل GlobalAveragePooling
# المصدر  : src/chapters/ch40_common_errors.md:61
# الترتيب : 4/10
# ──────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 3 — معدّل تعلّم كبير
# المصدر  : src/chapters/ch40_common_errors.md:86
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 4 — بيانات فيها NaN
# المصدر  : src/chapters/ch40_common_errors.md:101
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 5 — ميزات بمقياس مختلف
# المصدر  : src/chapters/ch40_common_errors.md:119
# الترتيب : 8/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 40: الأخطاء الشائعة وكيف تُصلحها
# القسم   : الخطأ 7 — طبقة مخصّصة لا تُحمَّل
# المصدر  : src/chapters/ch40_common_errors.md:152
# الترتيب : 10/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

# الحلّ: تحديد الشكل بحسب البيانات
inp = keras.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(16, 3, padding="same")(inp)
out = keras.layers.Dense(10, activation="softmax")(
    keras.layers.GlobalAveragePooling2D()(x))
model = keras.Model(inp, out)
# النتيجة المتوقعة: النموذج يقبل بيانات 28×28.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras

inp = keras.Input(shape=(8, 8, 16))
# ✗ خاطئ: Flatten يُنتج 1024
# x = keras.layers.Flatten()(inp)
# ✓ صحيح: GlobalAveragePooling2D يُنتج 16
x = keras.layers.GlobalAveragePooling2D()(inp)
out = keras.layers.Dense(10)(x)
# النتيجة المتوقعة: (batch, 10).

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss="mse",
)
# النتيجة المتوقعة: الخسارة لا تصبح NaN.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import numpy as np

x = np.random.rand(100, 16).astype("float32")   # بياناتك الفعلية
# تحقّق من NaN وInf
assert not np.isnan(x).any(), "x يحتوي NaN"
assert not np.isinf(x).any(), "x يحتوي Inf"
# النتيجة المتوقعة: البيانات نظيفة.

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# بيانات بمقياس مختلف
x = np.concatenate([
    np.random.rand(200, 8) * 255.0,       # ميزة كبيرة
    np.random.rand(200, 8),               # ميزة صغيرة
], axis=1).astype("float32")

norm = keras.layers.Normalization()
norm.adapt(x)       # يُحسب mean وstd على التدريب فقط
x_norm = norm(x)
# النتيجة المتوقعة: كلّ ميزة بمتوسّط ≈ 0 وانحراف ≈ 1.

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-40-ch40-common-errors.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import keras

# حمّل مع custom_objects
# m = keras.saving.load_model("m.keras",
#                             custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة تُحمَّل بنجاح.
