# ──────────────────────────────────────────────────────────────────────────
# الفصل 43: مشروع تحليل المشاعر ونشره كخدمة
# القسم   : 43.4 النموذج
# المصدر  : src/chapters/ch43_project_nlp.md:84
# الترتيب : 3/8
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_classifier(vocab_size=2000, num_classes=3):
    """نموذج تصنيف نصّ: تضمين → LSTM → تصنيف."""
    inp = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inp)                                        # (B, 20)
    x = keras.layers.Embedding(vocab_size, 32)(x)              # (B, 20, 32)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)   # (B, 64)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)

model = build_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
# النتيجة المتوقعة: نموذج ثنائيّ الاتجاه بثلاثة أصناف.
