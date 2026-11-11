# ──────────────────────────────────────────────────────────────────────────
# الفصل 35: الحفظ والتحميل
# القسم   : 35.3 الحفظ بلا حالة المحسّن
# المصدر  : src/chapters/ch35_saving_loading.md:64
# الترتيب : 2/9
# ──────────────────────────────────────────────────────────────────────────
# حفظ الأوزان فقط
model.save_weights("model.weights.h5")

# إنشاء نموذج بنفس البنية ثم تحميل الأوزان
new_model = keras.Sequential([
    keras.Input(shape=(16,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),
])
new_model.load_weights("model.weights.h5")
# النتيجة المتوقعة: الأوزان مُحمَّلة، وحالة المحسّن مفقودة.
