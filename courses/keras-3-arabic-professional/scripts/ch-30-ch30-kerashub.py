#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — منظومة KerasHub وKerasNLP
Keras 3 in Arabic — Kerashub

Book slug : keras-3-arabic-professional
Chapter   : ch-30-ch30-kerashub (lesson 30)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-30-ch30-kerashub.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — منظومة KerasHub وKerasNLP
Keras 3 in Arabic — Kerashub

Book slug : keras-3-arabic-professional
Chapter   : ch-30-ch30-kerashub (lesson 30)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-30-ch30-kerashub.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — منظومة KerasHub وKerasNLP
Keras 3 in Arabic — Kerashub

Book slug : keras-3-arabic-professional
Chapter   : ch-30-ch30-kerashub (lesson 30)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch30_kerashub/001_s30_2_loading_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.2 تحميل نموذج
# المصدر  : src/chapters/ch30_kerashub.md:33
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch30_kerashub/002_s30_4_backbone_classification_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.4 تصنيف نصّ فوق backbone
# المصدر  : src/chapters/ch30_kerashub.md:58
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch30_kerashub/003_s30_5_fine_tuning_fine_tuning.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.5 الضبط الدقيق (Fine-Tuning)
# المصدر  : src/chapters/ch30_kerashub.md:86
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch30_kerashub/004_backbone_measure_cost_before.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : قياس كلفة backbone قبل الالتزام
# المصدر  : src/chapters/ch30_kerashub.md:132
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch30_kerashub/005_unfreezing_freezing_gradual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : فكّ التجميد المتدرّج
# المصدر  : src/chapters/ch30_kerashub.md:172
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.2 تحميل نموذج
# المصدر  : src/chapters/ch30_kerashub.md:33
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.4 تصنيف نصّ فوق backbone
# المصدر  : src/chapters/ch30_kerashub.md:58
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.5 الضبط الدقيق (Fine-Tuning)
# المصدر  : src/chapters/ch30_kerashub.md:86
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : قياس كلفة backbone قبل الالتزام
# المصدر  : src/chapters/ch30_kerashub.md:132
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : فكّ التجميد المتدرّج
# المصدر  : src/chapters/ch30_kerashub.md:172
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — منظومة KerasHub وKerasNLP
Keras 3 in Arabic — Kerashub

Book slug : keras-3-arabic-professional
Chapter   : ch-30-ch30-kerashub (lesson 30)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch30_kerashub/001_s30_2_loading_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.2 تحميل نموذج
# المصدر  : src/chapters/ch30_kerashub.md:33
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch30_kerashub/002_s30_4_backbone_classification_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.4 تصنيف نصّ فوق backbone
# المصدر  : src/chapters/ch30_kerashub.md:58
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch30_kerashub/003_s30_5_fine_tuning_fine_tuning.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.5 الضبط الدقيق (Fine-Tuning)
# المصدر  : src/chapters/ch30_kerashub.md:86
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch30_kerashub/004_backbone_measure_cost_before.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : قياس كلفة backbone قبل الالتزام
# المصدر  : src/chapters/ch30_kerashub.md:132
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch30_kerashub/005_unfreezing_freezing_gradual.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : فكّ التجميد المتدرّج
# المصدر  : src/chapters/ch30_kerashub.md:172
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch30_kerashub.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.2 تحميل نموذج
# المصدر  : src/chapters/ch30_kerashub.md:33
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.4 تصنيف نصّ فوق backbone
# المصدر  : src/chapters/ch30_kerashub.md:58
# الترتيب : 2/5
# ──────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : 30.5 الضبط الدقيق (Fine-Tuning)
# المصدر  : src/chapters/ch30_kerashub.md:86
# الترتيب : 3/5
# ──────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : قياس كلفة backbone قبل الالتزام
# المصدر  : src/chapters/ch30_kerashub.md:132
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 30: منظومة KerasHub وKerasNLP
# القسم   : فكّ التجميد المتدرّج
# المصدر  : src/chapters/ch30_kerashub.md:172
# الترتيب : 5/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras_hub
import keras

# تحميل backbone مُدرَّب مسبقًا (لا تنزّل في بيئة معزولة)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
# أو تحميل النموذج الكامل مع رأس تصنيف
classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_base_en_uncased", num_classes=2)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

# ─── 1. backbone مُجمّد (استخراج ميزات)
backbone = keras_hub.models.BertBackbone.from_preset("bert_base_en_uncased")
backbone.trainable = False

# ─── 2. النموذج: نصّ → backbone → رأس تصنيف
inp = keras.Input(shape=(1,), dtype="string")
# في KerasHub، النموذج الكامل يتولّى المُرمِّز داخليًّا؛
# إن بُنيت يدويًّا، مرّر معرّفات الرموز مباشرةً:
x = backbone(inp)["sequence_output"]     # (B, L, hidden)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(32, activation="relu")(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 3. تدريب الرأس فقط
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# مرحلة 1: تدريب الرأس فقط (backbone مجمّد)
backbone.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

# مرحلة 2: فكّ تجميد backbone بمعدّل تعلّم أدنى بكثير
backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np
import time

def probe_model(build_fn, input_dim=16, n=100):
    """يُعطي معاملات وزمن استدلال وحجم أوزان."""
    m = build_fn()
    params = m.count_params()
    x = np.random.rand(n, input_dim).astype("float32")
    m.predict(x, verbose=0)                       # warm-up
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"params": params, "ms_per_sample": ms}

def build_small():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(1)])

def build_large():
    return keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(256, activation="relu"),
                             keras.layers.Dense(128, activation="relu"),
                             keras.layers.Dense(1)])

for name, fn in (("صغير", build_small), ("كبير", build_large)):
    r = probe_model(fn)
    print(f"{name}: {r['params']:,} params · {r['ms_per_sample']:.3f} ms/عيّنة")
# النتيجة المتوقعة: النموذج الكبير أبطأ بعدد أضعاف نسبة معاملاته —
# راجع هذا الرقم قبل اختيار backbone في الخدمة.

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-30-ch30-kerashub.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. نموذج محاكاة backbone + رأس
backbone = keras.Sequential([keras.Input(shape=(16,)),
                             keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(32, activation="relu")])
inp = keras.Input(shape=(16,))
x = backbone(inp)
out = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="mse")

# ─── 2. قياس قابلية التدريب في كلّ حالة
def trainable_report(m):
    return (sum(int(np.prod(w.shape)) for w in m.trainable_weights),
            sum(int(np.prod(w.shape)) for w in m.non_trainable_weights))

# حالة 1: مجمّد كاملًا
backbone.trainable = False
print("مجمّد كاملًا:", trainable_report(model))

# حالة 2: فكّ آخر طبقة فقط
backbone.trainable = False
backbone.layers[-1].trainable = True
print("آخر طبقة:", trainable_report(model))

# حالة 3: مفكوك كاملًا
backbone.trainable = True
print("مفكوك كاملًا:", trainable_report(model))
# النتيجة المتوقعة: معاملات قابلة للتدريب تتدرّج من صفر
# إلى الحجم الكامل — وهذا هو مسار الضبط الدقيق الجزئيّ.
