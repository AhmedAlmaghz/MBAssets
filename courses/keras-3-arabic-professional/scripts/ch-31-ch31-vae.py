#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الترميز الآلي المتغيّر (VAE)
Keras 3 in Arabic — Vae

Book slug : keras-3-arabic-professional
Chapter   : ch-31-ch31-vae (lesson 31)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-31-ch31-vae.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الترميز الآلي المتغيّر (VAE)
Keras 3 in Arabic — Vae

Book slug : keras-3-arabic-professional
Chapter   : ch-31-ch31-vae (lesson 31)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-31-ch31-vae.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الترميز الآلي المتغيّر (VAE)
Keras 3 in Arabic — Vae

Book slug : keras-3-arabic-professional
Chapter   : ch-31-ch31-vae (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch31_vae/001_s31_4_keras.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch31_vae/002_detection_anomalies.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : كشف الشواذ: التطبيق العمليّ
# المصدر  : src/chapters/ch31_vae.md:193
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch31_vae/003_threshold_curve_precision.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : اختيار العتبة على منحنى الدقّة والاستدعاء
# المصدر  : src/chapters/ch31_vae.md:248
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch31_vae.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch31_vae.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch31_vae.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : كشف الشواذ: التطبيق العمليّ
# المصدر  : src/chapters/ch31_vae.md:193
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : اختيار العتبة على منحنى الدقّة والاستدعاء
# المصدر  : src/chapters/ch31_vae.md:248
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — الترميز الآلي المتغيّر (VAE)
Keras 3 in Arabic — Vae

Book slug : keras-3-arabic-professional
Chapter   : ch-31-ch31-vae (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch31_vae/001_s31_4_keras.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch31_vae/002_detection_anomalies.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : كشف الشواذ: التطبيق العمليّ
# المصدر  : src/chapters/ch31_vae.md:193
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch31_vae/003_threshold_curve_precision.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : اختيار العتبة على منحنى الدقّة والاستدعاء
# المصدر  : src/chapters/ch31_vae.md:248
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch31_vae.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch31_vae.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch31_vae.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : كشف الشواذ: التطبيق العمليّ
# المصدر  : src/chapters/ch31_vae.md:193
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 31: الترميز الآلي المتغيّر (VAE)
# القسم   : اختيار العتبة على منحنى الدقّة والاستدعاء
# المصدر  : src/chapters/ch31_vae.md:248
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. بيانات سليمة (تدريب) وبيانات شاذّة (اختبار)
n_normal = 400
x_normal = np.random.randn(n_normal, 28, 28, 1).astype("float32") * 0.3
x_anomaly = np.random.randn(50, 28, 28, 1).astype("float32") * 1.5

# ─── 2. VAE مصغّرة: encoder + decoder
LATENT = 4
enc_in = keras.Input(shape=(28, 28, 1))
z = keras.layers.Flatten()(enc_in)
mu = keras.layers.Dense(LATENT)(z)
log_var = keras.layers.Dense(LATENT)(z)
# المشفّر يُخرج μ وlog σ² فقط؛ سحب الضجيج يبقى خارج الرسم البياني
# كي يعمل مع أي حجم دفعة (لا شكل ثابت مدفون في العقدة)
enc = keras.Model(enc_in, [mu, log_var])

dec_in = keras.Input(shape=(LATENT,))
d = keras.layers.Dense(28 * 28, activation="sigmoid")(dec_in)
d = keras.layers.Reshape((28, 28, 1))(d)
dec = keras.Model(dec_in, d)
dec.compile(optimizer="adam", loss="mse")  # المشفّر-المشفوف يُدرَّب بخسارة إعادة البناء

# ─── 3. التدريب على السليم فقط
mu_n, lv_n = enc.predict(x_normal, verbose=0)
eps = np.random.default_rng(0).standard_normal(mu_n.shape).astype("float32")
z_train = mu_n + np.exp(0.5 * lv_n) * eps      # إعادة المعالجة
dec.fit(z_train, x_normal, epochs=10, batch_size=32, verbose=0)

# ─── 4. درجة الشذوذ = خطأ إعادة البناء
def anomaly_score(x):
    mu_, lv_ = enc.predict(x, verbose=0)
    eps_ = np.random.default_rng(1).standard_normal(mu_.shape).astype("float32")
    z_ = mu_ + np.exp(0.5 * lv_) * eps_
    recon = dec.predict(z_, verbose=0)
    return np.mean(np.square(x - recon), axis=(1, 2, 3))

s_normal = anomaly_score(x_normal[:100])
s_anomaly = anomaly_score(x_anomaly)
threshold = s_normal.mean() + 3 * s_normal.std()
recall = float((s_anomaly > threshold).mean())
print(f"threshold = {threshold:.4f}, recall = {recall:.2%}")
# النتيجة المتوقعة: recall مرتفع — الشواذ تفشل في إعادة البناء.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-31-ch31-vae.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def sweep_threshold(scores_normal, scores_anomaly):
    """يُجرب عتبات ويُعطي أزواج (precision, recall)."""
    results = []
    for q in (0.90, 0.95, 0.99, 0.995, 0.999):
        thr = float(np.quantile(scores_normal, q))
        tp = float((scores_anomaly > thr).mean())
        fp = float((scores_normal > thr).mean())
        prec = tp / max(tp + fp, 1e-8)
        results.append((q, thr, prec, tp))
    return results

rng = np.random.default_rng(42)
s_normal = rng.normal(0.010, 0.002, 400)
s_anomaly = rng.normal(0.045, 0.010, 60)
print(f"{'quantile':>9} {'threshold':>10} {'precision':>10} {'recall':>8}")
for q, thr, prec, rec in sweep_threshold(s_normal, s_anomaly):
    print(f"{q:>9.3f} {thr:>10.4f} {prec:>10.3f} {rec:>8.3f}")
# النتيجة المتوقعة: كلّما ارتفع quantile ارتفعت precision وانخفض recall
# — والاختيار قرار عمل لا قرار رياضيّ.
