# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : من `int` إلى `Embedding`: المسار الكامل
# المصدر  : src/chapters/ch16_text_layers.md:133
# الترتيب : 2/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

texts = np.array([
    "المنتج ممتاز وجودة عالية",
    "جودة المنتج ممتازة وسعر مناسب",
    "منتج سيئ لا أنصح به",
    "سيء جدا وباهظ الثمن",
    "رائع جدا وأعجبني",
    "سريع التوصيل وسعره جيد",
    "منتج ممتاز وسريع",
    "لا ينصح به أبدا",
] * 8)

labels = np.array([1, 1, 0, 0, 1, 1, 1, 0] * 8)

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=12,     # تثبيت الطول
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(input_dim=1000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),     # يطبّق القناع تلقائيًا
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(texts, labels, epochs=5, batch_size=8, verbose=0)

print(model.evaluate(texts, labels, verbose=0))
# النتيجة المتوقعة: loss وaccuracy على مجموعة صغيرة
