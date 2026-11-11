# ──────────────────────────────────────────────────────────────────────────
# الفصل 1: Keras 3: كيف تغيّر عالَم التعلّم العميق؟
# القسم   : 1.7 مثال محلول: ترقية نظام كشف الاحتيال
# المصدر  : src/chapters/ch01_keras3_world.md:139
# الترتيب : 2/2
# ──────────────────────────────────────────────────────────────────────────
import keras

# Keras 2 (قبل): from tensorflow import keras; import tensorflow as tf
# Keras 3 (بعد): استيراد موحّد + عمليات محايدة عن المحرّك
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),          # طبقة الإدخال صارت صريحة داخل الـSequential
    keras.layers.Dense(64, activation="relu"),  # التنشيط يُمرَّر كوسيط لا كطبقة منفصلة
    keras.layers.Dropout(0.2),                 # إسقاط عشوائي أثناء التدريب فقط
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],  # الاسم الصحيح في Keras 3 هو AUC
)
model.summary()  # النتيجة المتوقعة: ملخّص بعدد الأوزان (205) لكل طبقة Dense
