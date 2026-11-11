# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.5 التنفيذ في Keras: كتلة CNN معيارية
# المصدر  : src/chapters/ch19_cnn_layers.md:127
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)


def conv_block(x, filters, kernel_size=3, stride=1, use_norm=True, drop=0.0):
    """كتلة تلافيفية: الالتفاف ← (BatchNorm) ← التفعيل ← (Dropout)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=not use_norm,      # مع BN لا نحتاج تحيزًا (BN له إزاحة خاصة)
    )(x)
    if use_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if drop > 0:
        x = keras.layers.Dropout(drop)(x)
    return x


def build_cnn(input_shape=(96, 96, 3), num_classes=10, width=32, depth=4):
    """بنية CNN بسيطة مع تدهور المساحة تدريجيًا."""
    x = keras.layers.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    for i in range(depth):
        filters = width * (2 ** i)
        x = conv_block(x, filters, stride=2 if i > 0 else 1, drop=0.1)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=x, outputs=out)
    return model


model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# تحقّق من الأبعاد
probe = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    keras.layers.Conv2D(32, 3, padding="same"),
    keras.layers.MaxPooling2D(2),
])
print(probe.output_shape)   # النتيجة المتوقعة: (None, 48, 48, 32)
