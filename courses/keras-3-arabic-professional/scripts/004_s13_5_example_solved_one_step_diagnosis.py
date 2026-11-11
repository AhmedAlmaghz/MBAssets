# ──────────────────────────────────────────────────────────────────────────
# الفصل 13: التنظيم والقيود: كيف تمنع النموذج من الحفظ بدل التعلّم
# القسم   : 13.5 مثال محلول خطوة بخطوة: مصنّف تشخيص طبي
# المصدر  : src/chapters/ch13_regularization.md:240
# الترتيب : 4/5
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة تحاكي صور الأشعة (في الواقع: 9,000 صورة)
n = 800
x_all = np.random.rand(n, 64, 64, 3).astype("float32")
y_all = (np.random.rand(n) < 0.044).astype("int32")   # 4.4% إيجابية
x_train, y_train = x_all[:600], y_all[:600]
x_val, y_val = x_all[600:], y_all[600:]

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),

    keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),          # يثبّت إحصاءات: يطبّع التوزيعات
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),                 # إسقاط بعد كل كتلة
    keras.layers.SpatialDropout2D(0.25),        # إسقاط على مستوى القنوات

    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.30),

    keras.layers.GlobalAveragePooling2D(),      # يقلّص السعة بشدّة
    keras.layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.L2(1e-4)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
