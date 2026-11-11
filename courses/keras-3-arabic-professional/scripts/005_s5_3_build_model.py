# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.3 بناء النموذج
# المصدر  : src/chapters/ch05_first_model.md:70
# الترتيب : 5/14
# ──────────────────────────────────────────────────────────────────────────
inputs = keras.Input(shape=(28, 28), name="image")

# التسطيح: من (28, 28) إلى (784,)
x = keras.layers.Flatten(name="flatten")(inputs)

# الطبقة الكثيفة الأولى: مستخرج الميزات
x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)

# الإسقاط: لا_connections عشوائية 20% من الوحدات
x = keras.layers.Dropout(0.2, name="dropout")(x)

# الطبقة الأخيرة: logits (بلا تنشيط)
outputs = keras.layers.Dense(10, name="logits")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")
model.summary()
