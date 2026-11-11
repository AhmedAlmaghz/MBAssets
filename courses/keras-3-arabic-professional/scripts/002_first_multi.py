# ──────────────────────────────────────────────────────────────────────────
# الفصل 17: الأوزان والمهام المتعدّدة: كيف توازن النموذج وتشقّه
# القسم   : النمط الأول: مدخلات متعدّدة
# المصدر  : src/chapters/ch17_multi_task_weights.md:106
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# مدخلات متعدّدة: صورة + متجه خصائص
img_input = keras.Input(shape=(32, 32, 3), name="image")
feat_input = keras.Input(shape=(10,), name="features")

x = keras.layers.Conv2D(32, 3, activation="relu")(img_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64, activation="relu")(x)

y = keras.layers.Dense(32, activation="relu")(feat_input)

merged = keras.layers.Concatenate()([x, y])
out = keras.layers.Dense(4, activation="softmax")(merged)

model = keras.Model(inputs=[img_input, feat_input], outputs=out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # التسميات أعداد صحيحة
    metrics=["accuracy"],
)
model.summary()

# بيانات تجريبية
imgs = np.random.rand(64, 32, 32, 3).astype("float32")
feats = np.random.rand(64, 10).astype("float32")
labels = np.random.randint(0, 4, size=64)

history = model.fit(
    {"image": imgs, "features": feats},
    labels,
    epochs=3,
    batch_size=8,
    verbose=0,
)
