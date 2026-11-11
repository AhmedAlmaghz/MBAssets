# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.7 تمارين
# المصدر  : src/chapters/ch12_callbacks.md:546
# الترتيب : 10/11
# ──────────────────────────────────────────────────────────────────────────
import keras

class BadStopper(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs["val_loss"] > 0.5:
            self.model.stop_training = True
