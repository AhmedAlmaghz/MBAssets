# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.4 الواجهة الثالثة: Subclassing
# المصدر  : src/chapters/ch04_model_architecture.md:135
# الترتيب : 6/10
# ──────────────────────────────────────────────────────────────────────────
import keras

class MyModel(keras.Model):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(units, activation="relu")
        self.dense2 = keras.layers.Dense(units, activation="relu")
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        # منطق مخصّص: Residual connection على مرحلتين
        h = self.dense1(x)
        h = self.dense2(h)
        return self.out(x + h)
