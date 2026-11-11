# ──────────────────────────────────────────────────────────────────────────
# الفصل 3: الأوتور: اللغة التي يفكّر بها Keras
# القسم   : القابل للتعلّم وغير القابل
# المصدر  : src/chapters/ch03_tensors.md:246
# الترتيب : 11/12
# ──────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input((8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1),
])

print("قابل للتعلّم   :", len(model.trainable_variables))
print("غير قابل للتعلّم:", len(model.non_trainable_variables))
for w in model.non_trainable_variables:
    print("   •", w.name, w.shape)
