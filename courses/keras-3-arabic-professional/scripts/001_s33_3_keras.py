# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : 33.3 التنفيذ في Keras
# المصدر  : src/chapters/ch33_diffusion.md:58
# الترتيب : 1/4
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
TIMESTEPS = 100
BETA_START = 1e-4
BETA_END = 0.02
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. جدول التباين
betas = np.linspace(BETA_START, BETA_END, TIMESTEPS).astype("float32")
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas).astype("float32")

# ─── 3. شبكة تنبّأ الضجيج (UNet مبسّط)
def noise_network(x, t):
    """شبكة تتنبّأ بالضجيج في الخطوة t."""
    # تضمين الزمن
    t_embed = keras.layers.Embedding(TIMESTEPS, 32)(t)
    t_dense = keras.layers.Dense(28 * 28, activation="relu")(t_embed)
    t_reshaped = keras.layers.Reshape((28, 28, 1))(t_dense)
    x = keras.layers.Concatenate()([x, t_reshaped])
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(1, 3, padding="same")(x)
    return x

t_input = keras.Input(shape=(), name="t")
x_input = keras.Input(shape=IMAGE_SHAPE, name="x_t")
noise_pred = noise_network(x_input, t_input)
noise_model = keras.Model([x_input, t_input], noise_pred, name="noise_predictor")

# ─── 4. خطّ التدريب: لكلّ عيّنة، اختر خطوة عشوائيًّا وأضِف الضجيج
def forward_diffuse(x0, t):
    """أضِف ضجيجًا إلى x0 في الخطوة t."""
    # alpha_bar شكلها (batch,) — بلا إعادة تشكيل تبثّ على المحور الأخير
    # فتصير الصورة 32 قناة بدل قناة واحدة
    alpha_bar = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    noise = np.random.randn(*x0.shape).astype("float32")
    x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
    return x_t.astype("float32"), noise

# ─── 5. تدريب
noise_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
n = 200
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
for epoch in range(5):
    for step in range(n // 32):
        batch_x = x_data[np.random.randint(0, n, 32)]
        t_batch = np.random.randint(0, TIMESTEPS, size=32)
        x_t, noise = forward_diffuse(batch_x, t_batch)
        loss = noise_model.train_on_batch([x_t, t_batch], noise)
    print(f"epoch {epoch}: loss={loss:.4f}")
# النتيجة المتوقعة: خسارة MSE تنخفض بمرور الدورات.
