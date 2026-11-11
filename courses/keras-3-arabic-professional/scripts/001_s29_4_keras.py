# ──────────────────────────────────────────────────────────────────────────
# الفصل 29: المحوّلات وآلية الانتباه
# القسم   : 29.4 التنفيذ في Keras
# المصدر  : src/chapters/ch29_transformers.md:65
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات صغيرة
VOCAB = 500
SEQ_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2

# ─── 2. كتلة المحوّل
def transformer_block(x, num_heads, d_model, dropout=0.1):
    """كتلة محوّل: انتباه + MLP + تطبيع + اتصالات تخطّي."""
    # مرحلة الانتباه
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads,
        dropout=dropout)(y, y)
    x = keras.layers.Add()([x, y])
    # مرحلة MLP
    y = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = keras.layers.Dense(d_model * 2, activation="gelu")(y)
    y = keras.layers.Dense(d_model)(y)
    y = keras.layers.Dropout(dropout)(y)
    return keras.layers.Add()([x, y])

# ─── 3. النموذج: تضمين + ترميز موضعيّ + كتل محوّل
inp = keras.Input(shape=(SEQ_LEN,))
x = keras.layers.Embedding(VOCAB, D_MODEL)(inp)               # (B, L, D)
pos = keras.layers.Embedding(SEQ_LEN, D_MODEL)(
    keras.ops.arange(SEQ_LEN))
x = x + pos                                                    # ترميز موضعيّ
for _ in range(NUM_BLOCKS):
    x = transformer_block(x, NUM_HEADS, D_MODEL)
x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
out = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out)

# ─── 4. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32")
y_data = np.random.randint(0, 2, size=(n,)).astype("int32")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_data, y_data, epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
# النتيجة المتوقعة: دقّة ≈ 0.55–0.65 على بيانات عشوائية.
