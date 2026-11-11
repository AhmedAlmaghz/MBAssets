# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : 4.6 مثال محلول: نموذج متعدّد المهام
# المصدر  : src/chapters/ch04_model_architecture.md:221
# الترتيب : 9/10
# ──────────────────────────────────────────────────────────────────────────
import keras

def build_model(input_dim=784, hidden=128):
    # ── الجذع المشترك
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = keras.layers.Dense(hidden, activation="relu", name="shared_1")(inputs)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(0.3, name="drop_1")(x)
    x = keras.layers.Dense(hidden // 2, activation="relu", name="shared_2")(x)

    # ── الرأسان
    cls = keras.layers.Dense(3, activation="softmax", name="category")(x)
    reg = keras.layers.Dense(1, name="price")(x)

    model = keras.Model(inputs=inputs, outputs=[cls, reg], name="multi_task")
    return model


model = build_model()
model.summary(line_length=78)
