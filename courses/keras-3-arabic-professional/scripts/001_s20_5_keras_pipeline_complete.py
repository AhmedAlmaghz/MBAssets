# ──────────────────────────────────────────────────────────────────────────
# الفصل 20: شبكة CNN من الصفر
# القسم   : 20.5 التنفيذ في Keras: خطّ كامل من الصفر
# المصدر  : src/chapters/ch20_cnn_from_scratch.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات مُصنّعة: 2000 صورة 96x96 بثلاث قنوات
n = 2000
x = np.random.rand(n, 96, 96, 3).astype("float32") * 255.0
# أُنشئ تسميات: 80% سليمة و20% معيبة (غير متوازنة)
y = (np.random.rand(n) < 0.2).astype("int32")

# ─── 2. سياسة التعزيز: تحاكي إضاءة المصنع وزوايا الكاميرا
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.15),
])

# ─── 3. الشبكة: 3 كتل (Conv-BN-ReLU) مع تدهور الأبعاد
def conv_block(x, filters):
    """كتلة تلافيفية قياسية: الالتفاف ثم التطبيع ثم التنشيط."""
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    return x

inp = keras.Input(shape=(96, 96, 3))
h = keras.layers.Rescaling(1.0 / 255.0)(inp)      # المعالجة داخل النموذج
h = augment(h)                                     # التعزيز أثناء التدريب فقط
for f in (32, 64, 128):
    h = conv_block(h, f)
h = keras.layers.GlobalAveragePooling2D()(h)
h = keras.layers.Dense(64, activation="relu")(h)
h = keras.layers.Dropout(0.3)(h)
out = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)

# ─── 4. الموازنة: وزن الصنف النادر
cw = {0: 1.0, 1: 50.0}
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.Precision(name="prec"),
        keras.metrics.Recall(name="rec"),
    ],
)
history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2,
                    class_weight=cw, verbose=0)
# النتيجة المتوقعة: recall يرتفع قرب 0.9 مع انخفاض بسيط في precision
