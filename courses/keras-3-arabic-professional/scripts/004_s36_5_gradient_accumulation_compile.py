# ──────────────────────────────────────────────────────────────────────────
# الفصل 36: تسريع التدريب
# القسم   : 36.5 التجميع بالتدرّج (Gradient Accumulation)
# المصدر  : src/chapters/ch36_acceleration.md:124
# الترتيب : 4/6
# ──────────────────────────────────────────────────────────────────────────
import keras

model.compile(
    # تجميع 4 دفعات صغيرة في تحديث واحد: خيار المُحسّن لا compile
    optimizer=keras.optimizers.Adam(1e-3, gradient_accumulation_steps=4),
    loss="sparse_categorical_crossentropy",
)
# النتيجة المتوقعة: نموذج يعمل كأنّ الدفعة 4× أكبر، بذاكرة أقلّ.
