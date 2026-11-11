# ──────────────────────────────────────────────────────────────────────────
# الفصل 34: الطبقات والنماذج المخصّصة
# القسم   : 34.3 نموذج مخصّص بخطوة تدريب مخصّصة
# المصدر  : src/chapters/ch34_custom_layers.md:83
# الترتيب : 2/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

@keras.saving.register_keras_serializable(package="book")
class MyModel(keras.Model):
    """نموذج بخطوة تدريب مخصّصة (تدرّج يدويّ)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        # Keras 3 لا يوحّد شريط التدرّج بين المحرّكات: نأخذه من المحرّك النشط.
        # (TensorFlow: tf.GradientTape · PyTorch: torch.enable_grad + backward)
        import tensorflow as tf
        with keras.name_scope("custom_train"):
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # تحديث المقاييس وبناؤها قبل قراءة النتائج (تتجنّب «لم تُبنَ بعد»)
        self.update_metrics(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# استخدامها
model = MyModel()
x = np.random.rand(200, 8).astype("float32")
y = np.random.rand(200, 1).astype("float32")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=["mae"])
model.fit(x, y, epochs=5, batch_size=32, verbose=0)
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.
