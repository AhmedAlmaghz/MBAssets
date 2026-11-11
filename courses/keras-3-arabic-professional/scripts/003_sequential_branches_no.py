# ──────────────────────────────────────────────────────────────────────────
# الفصل 4: بنية النموذج: ثلاث طرق لتحديد الشبكة
# القسم   : الفرعين: حالة لا يستطيع Sequential التعبير عنها
# المصدر  : src/chapters/ch04_model_architecture.md:78
# الترتيب : 3/10
# ──────────────────────────────────────────────────────────────────────────
import keras

# مدخلان
img_in = keras.Input(shape=(32, 32, 3), name="image")
meta_in = keras.Input(shape=(5,), name="metadata")

# معالجة كل فرع
img_feat = keras.layers.Conv2D(16, 3, activation="relu")(img_in)
img_feat = keras.layers.GlobalAveragePooling2D()(img_feat)
meta_feat = keras.layers.Dense(16, activation="relu")(meta_in)

# الدمج
merged = keras.layers.Concatenate()([img_feat, meta_feat])
out = keras.layers.Dense(1, activation="sigmoid")(merged)

model = keras.Model(inputs=[img_in, meta_in], outputs=out)
