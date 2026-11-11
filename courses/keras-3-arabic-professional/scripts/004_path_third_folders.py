# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الثالث: مجلدات صور
# المصدر  : src/chapters/ch14_data_pipeline.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)
