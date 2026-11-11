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
