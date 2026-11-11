# ──────────────────────────────────────────────────────────────────────────
# الفصل 7: دوال التنشيط: أين يموت التدرّج؟
# القسم   : 7.6 أخطاء شائعة
# المصدر  : src/chapters/ch07_activations.md:356
# الترتيب : 8/9
# ──────────────────────────────────────────────────────────────────────────
# [خاطئ عمدًا]: softmax مرتين + from_logits معًا
model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(3, activation="softmax"),                  # softmax هنا
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
