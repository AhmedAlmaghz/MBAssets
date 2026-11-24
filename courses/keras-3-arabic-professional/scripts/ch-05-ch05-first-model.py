#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — أول نموذج كامل: شرح سطر بسطر
Keras 3 in Arabic — First Model

Book slug : keras-3-arabic-professional
Chapter   : ch-05-ch05-first-model (lesson 5)
Code files: 26 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-05-ch05-first-model.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — أول نموذج كامل: شرح سطر بسطر
Keras 3 in Arabic — First Model

Book slug : keras-3-arabic-professional
Chapter   : ch-05-ch05-first-model (lesson 5)
Code files: 25 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-05-ch05-first-model.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — أول نموذج كامل: شرح سطر بسطر
Keras 3 in Arabic — First Model

Book slug : keras-3-arabic-professional
Chapter   : ch-05-ch05-first-model (lesson 5)
Code files: 24 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch05_first_model/001_s5_1_requirements_before_start.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.1 المتطلّبات قبل أن نبدأ
# المصدر  : src/chapters/ch05_first_model.md:17
# الترتيب : 1/14
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch05_first_model/002_s5_2_loading_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.2 جلب البيانات
# المصدر  : src/chapters/ch05_first_model.md:28
# الترتيب : 2/14
# ──────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch05_first_model/003_split_data_validation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : تقسيم بيانات التحقق
# المصدر  : src/chapters/ch05_first_model.md:48
# الترتيب : 3/14
# ──────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch05_first_model/004_float_conversion.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : التحويل إلى Float
# المصدر  : src/chapters/ch05_first_model.md:58
# الترتيب : 4/14
# ──────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch05_first_model/005_s5_3_build_model.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch05_first_model/007_s5_4_compile.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.4 التجميع: `compile`
# المصدر  : src/chapters/ch05_first_model.md:121
# الترتيب : 7/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch05_first_model/008_s5_5_fit_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.5 التدريب: `fit`
# المصدر  : src/chapters/ch05_first_model.md:150
# الترتيب : 8/14
# ──────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch05_first_model/009_fit_reading_output.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:188
# الترتيب : 9/14
# ──────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch05_first_model/010_fit_reading_output.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:198
# الترتيب : 10/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch05_first_model/011_evaluate_measure_data_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `evaluate`: قياس على بيانات لم يرها النموذج
# المصدر  : src/chapters/ch05_first_model.md:216
# الترتيب : 11/14
# ──────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch05_first_model/012_predict_production.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `predict`: إنتاج المخرجات
# المصدر  : src/chapters/ch05_first_model.md:229
# الترتيب : 12/14
# ──────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [012] code/chapters/ch05_first_model/013_s5_7_model_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.7 النموذج الكامل في ملف واحد
# المصدر  : src/chapters/ch05_first_model.md:251
# الترتيب : 13/14
# ──────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [021] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [022] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [023] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [024] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.1 المتطلّبات قبل أن نبدأ
# المصدر  : src/chapters/ch05_first_model.md:17
# الترتيب : 1/14
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.2 جلب البيانات
# المصدر  : src/chapters/ch05_first_model.md:28
# الترتيب : 2/14
# ──────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : تقسيم بيانات التحقق
# المصدر  : src/chapters/ch05_first_model.md:48
# الترتيب : 3/14
# ──────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : التحويل إلى Float
# المصدر  : src/chapters/ch05_first_model.md:58
# الترتيب : 4/14
# ──────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.4 التجميع: `compile`
# المصدر  : src/chapters/ch05_first_model.md:121
# الترتيب : 7/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.5 التدريب: `fit`
# المصدر  : src/chapters/ch05_first_model.md:150
# الترتيب : 8/14
# ──────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:188
# الترتيب : 9/14
# ──────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:198
# الترتيب : 10/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `evaluate`: قياس على بيانات لم يرها النموذج
# المصدر  : src/chapters/ch05_first_model.md:216
# الترتيب : 11/14
# ──────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `predict`: إنتاج المخرجات
# المصدر  : src/chapters/ch05_first_model.md:229
# الترتيب : 12/14
# ──────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.7 النموذج الكامل في ملف واحد
# المصدر  : src/chapters/ch05_first_model.md:251
# الترتيب : 13/14
# ──────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — أول نموذج كامل: شرح سطر بسطر
Keras 3 in Arabic — First Model

Book slug : keras-3-arabic-professional
Chapter   : ch-05-ch05-first-model (lesson 5)
Code files: 24 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch05_first_model/001_s5_1_requirements_before_start.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.1 المتطلّبات قبل أن نبدأ
# المصدر  : src/chapters/ch05_first_model.md:17
# الترتيب : 1/14
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch05_first_model/002_s5_2_loading_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.2 جلب البيانات
# المصدر  : src/chapters/ch05_first_model.md:28
# الترتيب : 2/14
# ──────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch05_first_model/003_split_data_validation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : تقسيم بيانات التحقق
# المصدر  : src/chapters/ch05_first_model.md:48
# الترتيب : 3/14
# ──────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch05_first_model/004_float_conversion.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : التحويل إلى Float
# المصدر  : src/chapters/ch05_first_model.md:58
# الترتيب : 4/14
# ──────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch05_first_model/005_s5_3_build_model.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch05_first_model/007_s5_4_compile.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.4 التجميع: `compile`
# المصدر  : src/chapters/ch05_first_model.md:121
# الترتيب : 7/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [007] code/chapters/ch05_first_model/008_s5_5_fit_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.5 التدريب: `fit`
# المصدر  : src/chapters/ch05_first_model.md:150
# الترتيب : 8/14
# ──────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [008] code/chapters/ch05_first_model/009_fit_reading_output.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:188
# الترتيب : 9/14
# ──────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [009] code/chapters/ch05_first_model/010_fit_reading_output.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:198
# الترتيب : 10/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [010] code/chapters/ch05_first_model/011_evaluate_measure_data_model.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `evaluate`: قياس على بيانات لم يرها النموذج
# المصدر  : src/chapters/ch05_first_model.md:216
# الترتيب : 11/14
# ──────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [011] code/chapters/ch05_first_model/012_predict_production.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `predict`: إنتاج المخرجات
# المصدر  : src/chapters/ch05_first_model.md:229
# الترتيب : 12/14
# ──────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [012] code/chapters/ch05_first_model/013_s5_7_model_complete.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.7 النموذج الكامل في ملف واحد
# المصدر  : src/chapters/ch05_first_model.md:251
# الترتيب : 13/14
# ──────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [013] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [014] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [015] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [016] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [017] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [018] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [019] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [020] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [021] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [022] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [023] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [024] src/chapters/ch05_first_model.md
# ────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.1 المتطلّبات قبل أن نبدأ
# المصدر  : src/chapters/ch05_first_model.md:17
# الترتيب : 1/14
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.2 جلب البيانات
# المصدر  : src/chapters/ch05_first_model.md:28
# الترتيب : 2/14
# ──────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : تقسيم بيانات التحقق
# المصدر  : src/chapters/ch05_first_model.md:48
# الترتيب : 3/14
# ──────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : التحويل إلى Float
# المصدر  : src/chapters/ch05_first_model.md:58
# الترتيب : 4/14
# ──────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.4 التجميع: `compile`
# المصدر  : src/chapters/ch05_first_model.md:121
# الترتيب : 7/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.5 التدريب: `fit`
# المصدر  : src/chapters/ch05_first_model.md:150
# الترتيب : 8/14
# ──────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:188
# الترتيب : 9/14
# ──────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : قراءة ناتج `fit`
# المصدر  : src/chapters/ch05_first_model.md:198
# الترتيب : 10/14
# ──────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `evaluate`: قياس على بيانات لم يرها النموذج
# المصدر  : src/chapters/ch05_first_model.md:216
# الترتيب : 11/14
# ──────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : `predict`: إنتاج المخرجات
# المصدر  : src/chapters/ch05_first_model.md:229
# الترتيب : 12/14
# ──────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 5: أول نموذج كامل: شرح سطر بسطر
# القسم   : 5.7 النموذج الكامل في ملف واحد
# المصدر  : src/chapters/ch05_first_model.md:251
# الترتيب : 13/14
# ──────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)   # تكرار النتائج

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("x_train:", x_train.shape, x_train.dtype)   # (60000, 28, 28) uint8
print("y_train:", y_train.shape, y_train.dtype)     # (60000,) uint8
print("y_train[:10]:", y_train[:10])                # [5 0 1 1 ...]

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# تقسيم عشوائي 80/20 من التدريب
idx = np.random.permutation(len(x_train))
split = int(0.8 * len(x_train))
x_tr, x_val = x_train[idx[:split]], x_train[idx[split:]]
y_tr, y_val = y_train[idx[:split]], y_train[idx[split:]]

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
x_tr = x_tr.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
history = model.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
print(history.history.keys())
# dict_keys(['loss', 'compile_metrics', 'val_loss', 'val_compile_metrics'])

print("أدنى خسارة تدريب :", min(history.history["loss"]))
print("أدنى خسارة تحقق :", min(history.history["val_loss"]))

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)
# {'loss': 0.0352, 'compile_metrics': 0.9891}

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
logits = model.predict(x_test[:5], verbose=0)
print("شكل المخرجات:", logits.shape)   # (5, 10)

probs = keras.ops.softmax(logits, axis=-1)
classes = keras.ops.argmax(probs, axis=-1)

for i in range(5):
    print(f"الصورة {i}: logits={np.round(logits[i], 2)}")
    print(f"          argmax={int(classes[i])} · الثقة={float(probs[i].max()):.4f}")

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/keras-3-arabic-professional/notebooks/ch-05-ch05-first-model.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
"""mnist_mlp.py — أول نموذج كامل، قابل للتشغيل من أعلى إلى أسفل."""
import numpy as np
import keras


def main():
    keras.utils.set_random_seed(42)

    # 1) البيانات
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    idx = np.random.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    x_tr, x_val = x_train[idx[:cut]], x_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]
    x_tr = x_tr.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # 2) النموذج
    inputs = keras.Input(shape=(28, 28), name="image")
    x = keras.layers.Flatten(name="flatten")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(10, name="logits")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_mlp")

    # 3) التجميع
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4) التدريب
    history = model.fit(
        x_tr, y_tr,
        epochs=20, batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    # 5) التقييم
    print("\n=== التقييم على مجموعة الاختبار ===")
    results = model.evaluate(x_test, y_test, return_dict=True)
    print("الخسارة :", round(float(results["loss"]), 4))
    print("الدقّة  :", round(float(results["accuracy"]), 4))

    # 6) تنبّؤ نموذجي
    logits = model.predict(x_test[:3], verbose=0)
    probs = keras.ops.softmax(logits, axis=-1)
    print("\n=== أمثلة على التنبّؤ ===")
    for i in range(3):
        print(f"الحقيقة={y_test[i]}  التنبّؤ={int(keras.ops.argmax(probs[i]))}  "
              f"الثقة={float(keras.ops.max(probs[i])):.3f}")

    # 7) الحفظ
    model.save("mnist_mlp.keras")
    print("\n✓ حُفظ النموذج في mnist_mlp.keras")


if __name__ == "__main__":
    main()
