# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.7 تمارين
# المصدر  : src/chapters/ch07_activations.md:391
# الترتيب : 9/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]
model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="sigmoid"),
])
model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam")
