# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : 31.4 التنفيذ في Keras
# المصدر  : src/chapters/ch31_vae.md:72
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 8
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المُرمِّز
encoder_input = keras.Input(shape=IMAGE_SHAPE, name="x")
x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(encoder_input)
x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Flatten()(x)
mu = keras.layers.Dense(LATENT_DIM, name="mu")(x)
log_var = keras.layers.Dense(LATENT_DIM, name="log_var")(x)

# ─── 3. حيلة إعادة التوزيع
def sample_z(args):
    """z = mu + sigma * epsilon، حيث sigma = exp(0.5 * log_var)."""
    mu_, log_var_ = args
    eps = keras.random.normal(shape=keras.ops.shape(mu_))
    return mu_ + keras.ops.exp(0.5 * log_var_) * eps

z = keras.layers.Lambda(sample_z)([mu, log_var])
encoder = keras.Model(encoder_input, [mu, log_var, z], name="encoder")

# ─── 4. المُفكّك
decoder_input = keras.Input(shape=(LATENT_DIM,), name="z")
y = keras.layers.Dense(7 * 7 * 32, activation="relu")(decoder_input)
y = keras.layers.Reshape((7, 7, 32))(y)
y = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(y)
y = keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(y)
decoder_output = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(y)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

# ─── 5. النموذج الكامل مع خسارة مركّبة
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.recon_tracker = keras.metrics.Mean(name="recon")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    def call(self, inputs):
        mu_, log_var_, z_ = self.encoder(inputs)
        return self.decoder(z_)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        # Keras 3 لا يوحّد شريط التدرّج: نأخذه من المحرّك النشط
        import tensorflow as tf
        with tf.GradientTape() as tape:
            mu_, log_var_, z_ = self.encoder(x)
            recon = self.decoder(z_)
            # خسارة إعادة البناء (متوسّطًا على الدفعة)
            recon_loss = keras.ops.mean(
                keras.ops.sum(keras.ops.square(x - recon), axis=[1, 2, 3]))
            # خسارة KL
            kl_loss = -0.5 * keras.ops.mean(
                keras.ops.sum(1 + log_var_ - keras.ops.square(mu_)
                              - keras.ops.exp(log_var_), axis=-1))
            total = recon_loss + kl_loss
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

# ─── 6. تدريب على بيانات مُصنّعة
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))
vae.fit(x_data, epochs=10, batch_size=32, verbose=0)
# النتيجة المتوقعة: recon ينخفض مع تقدّم الدورات، وkl يستقرّ حول قيمة محدّدة.
