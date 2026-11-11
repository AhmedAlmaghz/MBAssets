# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.6 الطبقات المخصّصة في الحفظ
# المصدر  : src/chapters/ch35_saving_loading.md:127
# الترتيب : 4/9
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs):
        return inputs @ self.w

    def get_config(self):
        return {**super().get_config(), "units": self.units}

# الحفظ
m = keras.Sequential([keras.Input(shape=(8,)), MyDense(4)])
m.save("custom_model.keras")

# التحميل
m_loaded = keras.saving.load_model("custom_model.keras",
                                   custom_objects={"MyDense": MyDense})
# النتيجة المتوقعة: الطبقة المخصّصة مُحمَّلة بنجاح.
