#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التفسير والتشخيص
Keras 3 in Arabic — Interpretability

Book slug : keras-3-arabic-professional
Chapter   : ch-39-ch39-interpretability (lesson 39)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-39-ch39-interpretability.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التفسير والتشخيص
Keras 3 in Arabic — Interpretability

Book slug : keras-3-arabic-professional
Chapter   : ch-39-ch39-interpretability (lesson 39)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-39-ch39-interpretability.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التفسير والتشخيص
Keras 3 in Arabic — Interpretability

Book slug : keras-3-arabic-professional
Chapter   : ch-39-ch39-interpretability (lesson 39)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch39_interpretability/001_s39_2_grad_cam_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.2 تنفيذ Grad-CAM في Keras
# المصدر  : src/chapters/ch39_interpretability.md:32
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch39_interpretability/002_s39_3_saliency_analysis_sensitivity.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.3 تحليل الحساسية (Saliency)
# المصدر  : src/chapters/ch39_interpretability.md:84
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch39_interpretability/003_s39_9_importance_features_tabular.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.9 أهميّة الميزات في النماذج الجدوليّة
# المصدر  : src/chapters/ch39_interpretability.md:162
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch39_interpretability/004_s39_10_diagnosis_drift_deployment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.10 تشخيص الانحراف بعد النشر
# المصدر  : src/chapters/ch39_interpretability.md:213
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch39_interpretability/005_test_leakage.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : اختبار التسريب عمليًّا
# المصدر  : src/chapters/ch39_interpretability.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.2 تنفيذ Grad-CAM في Keras
# المصدر  : src/chapters/ch39_interpretability.md:32
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.3 تحليل الحساسية (Saliency)
# المصدر  : src/chapters/ch39_interpretability.md:84
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.9 أهميّة الميزات في النماذج الجدوليّة
# المصدر  : src/chapters/ch39_interpretability.md:162
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.10 تشخيص الانحراف بعد النشر
# المصدر  : src/chapters/ch39_interpretability.md:213
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : اختبار التسريب عمليًّا
# المصدر  : src/chapters/ch39_interpretability.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التفسير والتشخيص
Keras 3 in Arabic — Interpretability

Book slug : keras-3-arabic-professional
Chapter   : ch-39-ch39-interpretability (lesson 39)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch39_interpretability/001_s39_2_grad_cam_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.2 تنفيذ Grad-CAM في Keras
# المصدر  : src/chapters/ch39_interpretability.md:32
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch39_interpretability/002_s39_3_saliency_analysis_sensitivity.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.3 تحليل الحساسية (Saliency)
# المصدر  : src/chapters/ch39_interpretability.md:84
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch39_interpretability/003_s39_9_importance_features_tabular.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.9 أهميّة الميزات في النماذج الجدوليّة
# المصدر  : src/chapters/ch39_interpretability.md:162
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch39_interpretability/004_s39_10_diagnosis_drift_deployment.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.10 تشخيص الانحراف بعد النشر
# المصدر  : src/chapters/ch39_interpretability.md:213
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch39_interpretability/005_test_leakage.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : اختبار التسريب عمليًّا
# المصدر  : src/chapters/ch39_interpretability.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch39_interpretability.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.2 تنفيذ Grad-CAM في Keras
# المصدر  : src/chapters/ch39_interpretability.md:32
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.3 تحليل الحساسية (Saliency)
# المصدر  : src/chapters/ch39_interpretability.md:84
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.9 أهميّة الميزات في النماذج الجدوليّة
# المصدر  : src/chapters/ch39_interpretability.md:162
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : 39.10 تشخيص الانحراف بعد النشر
# المصدر  : src/chapters/ch39_interpretability.md:213
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 39: التفسير والتشخيص
# القسم   : اختبار التسريب عمليًّا
# المصدر  : src/chapters/ch39_interpretability.md:230
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج CNN بسيط
inp = keras.Input(shape=(32, 32, 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 2. نموذج فرعيّ: صورة → آخر تلافيفيّ → تنبّؤ
last_conv_layer = model.get_layer("last_conv")
grad_model = keras.Model(model.input,
                         [last_conv_layer.output, model.output])

# ─── 3. حساب Grad-CAM
def gradcam(image, class_index=0):
    """خريطة Grad-CAM لصورة واحدة وصنف محدّد."""
    import tensorflow as tf          # Keras 3 لا يوحّد شريط التدرّج
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, conv_out)
    # المتوسّط العامّ للتدرّجات
    pooled = keras.ops.mean(grads, axis=(0, 1, 2))
    # الخريطة الموزونة
    heatmap = keras.ops.sum(conv_out[0] * pooled, axis=-1)
    heatmap = keras.ops.relu(heatmap)
    # تطبيع إلى [0, 1]
    heatmap = heatmap / (keras.ops.max(heatmap) + 1e-8)
    return np.array(heatmap)

# ─── 4. تطبيق على صورة
image = np.random.rand(32, 32, 3).astype("float32")
hm = gradcam(image, class_index=0)
# النتيجة المتوقعة: خريطة بحجم (16, 16) بقيم في [0, 1].

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def saliency(model, image, class_index=0):
    """خريطة الحساسية: التدرّج بالنسبة للمُدخل."""
    import tensorflow as tf          # شريط التدرّج من المحرّك النشط
    image = tf.Variable(tf.convert_to_tensor(image))   # يجب أن يكون موترًا يُراقَب
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image[None, ...])
        loss = pred[:, class_index]
    grads = tape.gradient(loss, image)
    return np.abs(np.array(grads.numpy()))

s = saliency(model, image, class_index=0)
# النتيجة المتوقعة: أوتور (32, 32, 3) بقيم غير سالبة.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج جدوليّ بسيط
model = keras.Sequential([keras.Input(shape=(8,)),
                          keras.layers.Dense(16, activation="relu"),
                          keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
x = np.random.rand(300, 8).astype("float32")
y = x[:, 0] * 2.0 + np.random.randn(300) * 0.1     # الميزة 0 هي الأهمّ
model.fit(x, y, epochs=5, batch_size=32, verbose=0)

# ─── 2. قياس الإسهام بتعطيل الميزة (permutation importance)
def permutation_importance(model, x, y, feature_idx, n=20):
    """زيادة خطأ عند خلط قيم الميزة."""
    base = float(np.mean((model.predict(x, verbose=0)[:, 0] - y) ** 2))
    increases = []
    for _ in range(n):
        x_shuffled = x.copy()
        x_shuffled[:, feature_idx] = np.random.permutation(x[:, feature_idx])
        err = float(np.mean((model.predict(x_shuffled, verbose=0)[:, 0] - y) ** 2))
        increases.append(err - base)
    return float(np.mean(increases))

for i in range(8):
    imp = permutation_importance(model, x, y, i)
    print(f"feature {i}: importance = {imp:.4f}")
# النتيجة المتوقعة: الميزة 0 تُظهر أهميّة أعلى بكثير من البقية.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def drift_score(reference_maps, current_maps):
    """متوسّط الفرق المطلق بين خرائط المرجع والإنتاج."""
    ref = np.mean(reference_maps, axis=0)
    cur = np.mean(current_maps, axis=0)
    return float(np.mean(np.abs(ref - cur)))

# في الإنتاج: يُحسب أسبوعيًّا ويُقارن بعتبة
# if drift_score(ref_maps, prod_maps) > 0.15:
#     alert("انحراف في سلوك النموذج — راجع البيانات")
# النتيجة المتوقعة: درجة انحراف تُراقب أسبوعيًّا.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-39-ch39-interpretability.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def leakage_check(x, y, threshold=0.95):
    """يرصد الميزات ذات الارتباط الشديد مع المُخرَج."""
    suspects = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if col.std() < 1e-8:
            continue
        corr = float(np.corrcoef(col, y)[0, 1])
        if abs(corr) >= threshold:
            suspects.append((i, corr))
    return suspects

# ─── 1. بيانات فيها تسريب: ميزة 3 = المُخرَج + ضجيج صغير
n = 300
x = np.random.randn(n, 8).astype("float32")
y = (2.0 * x[:, 0] + np.random.randn(n) * 0.1).astype("float32")
x[:, 3] = y + np.random.randn(n) * 0.01      # تسريب

# ─── 2. الارتباطات
for i, c in leakage_check(x, y, threshold=0.90):
    print(f"ميزة {i}: correlation = {c:.4f} ← تسريب محتمل")
# النتيجة المتوقعة: الميزة 3 تُرصد (ارتباط ≈ 0.9999).
