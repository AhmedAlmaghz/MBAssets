# ──────────────────────────────────────────────────────────────────────────
# الفصل 16: الطبقات النصية: من الجملة إلى المتجه
# القسم   : 16.5 مثال محلول: تحليل مشاعر مراجعات
# المصدر  : src/chapters/ch16_text_layers.md:292
# الترتيب : 7/10
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(7)

MAX_TOKENS = 20000
SEQ_LEN = 40

model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype="string"),
    keras.layers.TextVectorization(
        name="vec",
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        pad_to_max_tokens=True,
    ),
    keras.layers.Embedding(MAX_TOKENS, 64, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
