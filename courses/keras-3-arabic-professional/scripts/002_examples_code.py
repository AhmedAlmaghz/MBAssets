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
