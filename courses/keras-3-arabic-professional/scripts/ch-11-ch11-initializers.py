#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التهيئة: نقطة الانطلاق
Keras 3 in Arabic — Initializers

Book slug : keras-3-arabic-professional
Chapter   : ch-11-ch11-initializers (lesson 11)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-11-ch11-initializers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التهيئة: نقطة الانطلاق
Keras 3 in Arabic — Initializers

Book slug : keras-3-arabic-professional
Chapter   : ch-11-ch11-initializers (lesson 11)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-11-ch11-initializers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التهيئة: نقطة الانطلاق
Keras 3 in Arabic — Initializers

Book slug : keras-3-arabic-professional
Chapter   : ch-11-ch11-initializers (lesson 11)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch11_initializers/001_examples_code.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:129
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch11_initializers/002_examples_code.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:159
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch11_initializers/003_examples_code.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:201
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch11_initializers/004_seed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : البذرة: كيف تجعل الشبكة نفسها في كل مرة؟
# المصدر  : src/chapters/ch11_initializers.md:282
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:129
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:159
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:201
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : البذرة: كيف تجعل الشبكة نفسها في كل مرة؟
# المصدر  : src/chapters/ch11_initializers.md:282
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — التهيئة: نقطة الانطلاق
Keras 3 in Arabic — Initializers

Book slug : keras-3-arabic-professional
Chapter   : ch-11-ch11-initializers (lesson 11)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch11_initializers/001_examples_code.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:129
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch11_initializers/002_examples_code.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:159
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch11_initializers/003_examples_code.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:201
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch11_initializers/004_seed.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : البذرة: كيف تجعل الشبكة نفسها في كل مرة؟
# المصدر  : src/chapters/ch11_initializers.md:282
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch11_initializers.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:129
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:159
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : أمثلة برمجية
# المصدر  : src/chapters/ch11_initializers.md:201
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 11: التهيئة: نقطة الانطلاق
# القسم   : البذرة: كيف تجعل الشبكة نفسها في كل مرة؟
# المصدر  : src/chapters/ch11_initializers.md:282
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة عامة حتى تكون الأوزان (وبالتالي الأرقام) متطابقة في كل تشغيل
keras.utils.set_random_seed(0)

# القياس: كيف يغيّر المُهيّئ تباين مخرجات الطبقة عمليًّا؟
x = np.random.default_rng(0).standard_normal((20000, 784)).astype("float32")

for name in ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]:
 layer = keras.layers.Dense(
 64,
 activation="relu",
 use_bias=False,
 kernel_initializer=name,
 )
 layer.build((None, 784))
 h = keras.ops.convert_to_numpy(layer(x))
 # النتيجة المتوقعة: E[h^2] ~= 0.93 مع Glorot و ~= 1.00 مع He
 w_np = keras.ops.convert_to_numpy(layer.kernel)
 print(f"{name:16s} std(W)={np.std(w_np):.5f} E[h^2]={np.mean(h**2):.4f}")

# النتيجة المتوقعة:
# glorot_uniform std(W)=0.04862 E[h^2]=0.9269
# glorot_normal std(W)=0.04869 E[h^2]=0.9298
# he_uniform std(W)=0.05046 E[h^2]=0.9960
# he_normal std(W)=0.05055 E[h^2]=0.9994

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras


# مُهيّئ مخصّص: يرث من Initializer ويُعرّف __call__(shape, dtype)
class ScaledGlorotNormal(keras.initializers.Initializer):
    """GlorotNormal مضروبًا في معامل تحجيم يحدّه من الأعلى."""

    def __init__(self, scale=0.5, seed=None):
        # الصنف الأساس Initializer لا يقبل seed في __init__،
        # لذلك نخزّن مولّد بذرة خاصًّا بنا بأنفسنا
        self.scale = scale
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def __call__(self, shape, dtype=None):
        # انحراف GlorotNormal: sqrt(2/(fan_in+fan_out)) مقسومًا على معامل البتر 0.8796
        fan_in, fan_out = shape[-2], shape[-1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        values = keras.random.truncated_normal(
            shape,
            mean=0.0,
            stddev=stddev,
            seed=self.seed_generator,
            dtype=dtype,
        )
        return values * self.scale

    def get_config(self):
        # بدون get_config لا يمكن حفظ النموذج في ملف .keras ثم إعادة تحميله
        return {
            "scale": self.scale,
            "seed": self.seed_generator.get_config()["seed"],
        }


init = ScaledGlorotNormal(scale=0.5, seed=42)
w = init((784, 64))
print("std =", float(np.std(keras.ops.convert_to_numpy(w)))) # النتيجة المتوقعة: std = 0.02427 (نصف GlorotNormal)
print("config =", init.get_config()) # النتيجة المتوقعة: {'scale': 0.5, 'seed': 42}

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

# المحرك الداخلي: كل التهيئات الاحتمالية تقريبًا مجرد نداءات إلى VarianceScaling
v = keras.initializers.VarianceScaling(
 scale=2.0,  # معامل القياس: 1.0 لـ Glorot و 2.0 لـ He
 mode="fan_in", # fan_in | fan_out | fan_avg
 distribution="truncated_normal",
 seed=7,
)
w = v((784, 64))
print("std =", float(keras.ops.convert_to_numpy(w).std()))
# النتيجة المتوقعة: std = 0.05056 (يساوي sqrt(2/784) = 0.05051 تقريبًا)

# بالمعطيات نفسها ينتج HeNormal مصفوفة مطابقة تمامًا (لأنهما يبدآن من نفس المحرّك)
w2 = keras.initializers.HeNormal(seed=7)((784, 64))
print("same =", bool(keras.ops.all(keras.ops.convert_to_numpy(w) == keras.ops.convert_to_numpy(w2))))
# النتيجة المتوقعة: same = True

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-11-ch11-initializers.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# 1) البذرة الصحيحة: نفس النتيجة في كل مرة
a = keras.initializers.GlorotUniform(seed=7)((4, 4))
b = keras.initializers.GlorotUniform(seed=7)((4, 4))
print("حتمي:", bool(np.allclose(a, b))) # النتيجة المتوقعة: حتمي: True

# 2) المولّد ذو الحالة: كل سحب يعطي قيمة مختلفة
gen = keras.random.SeedGenerator(seed=7)
c = keras.random.normal((4, 4), seed=gen)
d = keras.random.normal((4, 4), seed=gen)
print("مختلف:", bool(not np.allclose(c, d))) # النتيجة المتوقعة: مختلف: True

# 3) البذرة العامة تضبط كل شيء: المولّدات والتهيئة وDropout
keras.utils.set_random_seed(1234)
keras.utils.set_random_seed(1234)
print("global seed set")
