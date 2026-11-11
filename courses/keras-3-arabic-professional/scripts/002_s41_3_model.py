# ──────────────────────────────────────────────────────────────────────────
# الفصل 41: مشروع تصنيف صور إنتاجيّ
# القسم   : 41.3 النموذج
# المصدر  : src/chapters/ch41_project_vision.md:56
# الترتيب : 2/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_shape=(64, 64, 3), num_classes=2):
    """نموذج CNN بنقل معرفة (backbone بلا أوزان مسبقة لتجنّب التنزيل)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)

    # backbone: في الإنتاج — MobileNetV3(weights="imagenet")
    x = keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)   # بلا softmax — لوجيتس
    return keras.Model(inp, out)

model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج بنقل معرفة، دقّة > 95% على بيانات مُصنّعة.
