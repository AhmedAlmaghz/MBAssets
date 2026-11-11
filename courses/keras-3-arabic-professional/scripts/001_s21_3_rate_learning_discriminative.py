# ──────────────────────────────────────────────────────────────────────────
# الفصل 21: نقل المعرفة والتدريب الدقيق
# القسم   : 21.3 معدّل التعلّم التمييزي
# المصدر  : src/chapters/ch21_transfer_learning.md:39
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_transfer_model(input_shape=(96, 96, 3), num_classes=2):
    """نموذج نقل: backbone مدرّب مسبقًا + رأس تصنيفي جديد."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    # backbone بلا أوزان مسبقة (لتجنّب تنزيل) — في الإنتاج: weights="imagenet"
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False                       # جمّد الشبكة القاعدة
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out), base

model, base = build_transfer_model()

# مرحلة 1: تدريب الرأس فقط بمعدّل تعلّم مرتفع
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# مرحلة 2: فكّ آخر كتلتين بمعدّل تعلّم أدنى
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# النتيجة المتوقعة: مرحلة 1 تعطي دقّة 85% بعد 5 دورات،
# ومرحلة 2 ترفعها إلى 90% بعد 3 دورات أخرى.
