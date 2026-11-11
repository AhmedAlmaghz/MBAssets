# ──────────────────────────────────────────────────────────────────────────
# الفصل 19: طبقات الشبكة التلافيفية في Keras
# القسم   : 19.7 مثال محلول: تصنيف قطع المعدّات
# المصدر  : src/chapters/ch19_cnn_layers.md:236
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import keras

def make_model():
    x = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # كتلة 1: استخراج الحواف
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 64

    # كتلة 2
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 32

    # كتلة 3
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)          # 16

    x = keras.layers.GlobalAveragePooling2D()(x)  # 128
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(x, out)


model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
