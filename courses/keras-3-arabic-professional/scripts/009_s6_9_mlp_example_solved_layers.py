# ──────────────────────────────────────────────────────────────────────────
# الفصل 6: الطبقات الأساسية: Dense والتنشيط والتسطيح
# القسم   : 6.9 مثال محلول: شبكة MLP بثلاث طبقات
# المصدر  : src/chapters/ch06_core_layers.md:230
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

def build_mlp(input_dim=784, hidden=(256, 128), n_classes=10, dropout=0.3):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(hidden):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        x = keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
    outputs = keras.layers.Dense(n_classes, name="logits")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    return model


model = build_mlp()
model.summary(line_length=76)

# التحقّق اليدوي من المعاملات
expected = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
print("المتوقّع يدويًا:", expected)
print("فعليًا          :", model.count_params())
assert expected == model.count_params()
