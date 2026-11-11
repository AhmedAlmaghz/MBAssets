# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.2 `Input` و`InputLayer`
# المصدر  : src/chapters/ch06_core_layers.md:42
# الترتيب : 1/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# الطريقة المستحسنة: Input
inputs = keras.Input(shape=(28, 28), name="image")

# المكافئ: InputLayer
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28, 28), name="image"),
    keras.layers.Dense(10),
])
print(model.input_shape)   # (None, 28, 28)
