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
