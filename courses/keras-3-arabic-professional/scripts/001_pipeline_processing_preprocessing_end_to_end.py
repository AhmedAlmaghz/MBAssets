# ──────────────────────────────────────────────────────────────────────────
# الفصل 15: المعالجة المسبقة والتعزيز: هندسة المدخلات قبل الشبكة
# القسم   : خط معالجة مسبق متكامل
# المصدر  : src/chapters/ch15_preprocessing_augmentation.md:74
# الترتيب : 1/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ── طبقة معالجة مسبقة مدمجة
preprocess = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),      # 0..255  →  0..1
    keras.layers.RandomFlip("horizontal"),     # انعكاس أفقي
    keras.layers.RandomRotation(0.1),         # دوران حتى ±10% من زاوية
    keras.layers.RandomZoom(0.1),             # تكبير/تصغير حتى ±10%
    keras.layers.RandomContrast(0.1),         # تعديل التباين
    keras.layers.Normalization(),             # توحيد إحصاءات القنوات
], name="preprocess")

# ── نموذج صغير يتّصل بها
model = keras.Sequential([
    keras.layers.Input(shape=(96, 96, 3)),
    preprocess,
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# بيانات تجريبية
x = np.random.randint(0, 255, size=(256, 96, 96, 3), dtype=np.uint8)
y = np.random.randint(0, 4, size=256)

history = model.fit(x, y, batch_size=32, epochs=3, validation_split=0.2)
