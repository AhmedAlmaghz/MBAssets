# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.10 وضع الأمان (Safe Mode)
# المصدر  : src/chapters/ch35_saving_loading.md:230
# الترتيب : 7/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class DoubleLayer(keras.layers.Layer):
    """طبقة تُضاعف المُدخل — بديل آمن عن Lambda(lambda x: x * 2)."""

    def call(self, inputs):
        return inputs * 2.0

    def get_config(self):
        return super().get_config()

m = keras.Sequential([keras.Input(shape=(8,)), DoubleLayer()])
m.save("safe.keras")
m2 = keras.saving.load_model("safe.keras",
                             custom_objects={"DoubleLayer": DoubleLayer})
# النتيجة المتوقعة: الطبقة تُحمَّل في الوضع الآمن — لأنّها لا تُنفّذ كودًا مضمّنًا.
