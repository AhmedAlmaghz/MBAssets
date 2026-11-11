# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:562
# الترتيب : 11/11
# ──────────────────────────────────────────────────────────────────────────
import keras

ckpt = keras.callbacks.ModelCheckpoint(
    filepath="best.weights.h5", save_best_only=True, save_weights_only=True
)
