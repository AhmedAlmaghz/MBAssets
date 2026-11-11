# ──────────────────────────────────────────────────────────────────────────
# الفصل 12: الـ Callbacks: أتمتة دورة التدريب
# القسم   : 12.5 كتابة `Callback` مخصّص
# المصدر  : src/chapters/ch12_callbacks.md:393
# الترتيب : 7/11
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

# بذرة ثابتة: بدونها يختلف مسار التدريب بين التشغيلات وتختلف الأرقام المطبوعة
keras.utils.set_random_seed(0)


class PlateauDetector(keras.callbacks.Callback):
    """يكشف الهضبة ويخفّض معدّل التعلّم، ويوقف التدريب بعد مهلة."""

    def __init__(self, patience=3, factor=0.5, min_lr=1e-6, max_wait=6):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_wait = max_wait
        # أعد ضبط الحالة في on_train_begin لتصبح الحالة قابلة لإعادة الاستخدام
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_train_begin(self, logs=None):
        # تصفير الحالة عند كل تشغيل جديد
        self.best = None
        self.wait = 0
        self.plateau_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # لا يوجد تحقّق: لا معنى للحديث عن الهضبة

        if self.best is None or val_loss < self.best - 1e-4:
            # تحسّن: سجّل أفضل قيمة وأعد ضبط العدّادات
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # بلغنا الهضبة: خفّض معدّل التعلّم
                current_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.plateau_count += 1
                self.wait = 0
                print(f"هضبة عند الحقبة {epoch}: lr {current_lr:.6f} -> {new_lr:.6f}")

                if self.plateau_count >= self.max_wait:
                    # فشلت كل المحاولات: أوقف التدريب
                    self.model.stop_training = True


model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

# بيانات ذات خطأ غير قابل للاختزال: النموذج يبلغ الحدّ الأدنى ثم يتذبذب حوله
rng = np.random.default_rng(0)
x = rng.standard_normal((128, 8)).astype("float32")
y = (x[:, 0] * 2.0 + rng.normal(0, 1.0, size=128)).astype("float32")

detector = PlateauDetector(patience=2, factor=0.5, max_wait=1)
history = model.fit(x, y, epochs=12, batch_size=16, validation_split=0.2,
                    callbacks=[detector], verbose=0)
print("عدد الحقبات المنفَّذة =", len(history.history["loss"]))
print("عدد مرّات اكتشاف الهضبة =", detector.plateau_count)
# النتيجة المتوقعة: عدد الحقبات المنفَّذة = 11 (توقّف بعد أول هضبة)
# النتيجة المتوقعة: عدد مرّات اكتشاف الهضبة = 1
