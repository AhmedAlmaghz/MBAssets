#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نماذج الانتشار (Diffusion Models)
Keras 3 in Arabic — Diffusion

Book slug : keras-3-arabic-professional
Chapter   : ch-33-ch33-diffusion (lesson 33)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-33-ch33-diffusion.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نماذج الانتشار (Diffusion Models)
Keras 3 in Arabic — Diffusion

Book slug : keras-3-arabic-professional
Chapter   : ch-33-ch33-diffusion (lesson 33)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-33-ch33-diffusion.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نماذج الانتشار (Diffusion Models)
Keras 3 in Arabic — Diffusion

Book slug : keras-3-arabic-professional
Chapter   : ch-33-ch33-diffusion (lesson 33)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch33_diffusion/001_s33_3_keras.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch33_diffusion/002_s33_4_generation_noise.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : 33.4 التوليد من ضجيج خالص
# المصدر  : src/chapters/ch33_diffusion.md:121
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch33_diffusion/003_variance_linear_versus_exponential.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : جدول التباين: خطّيّ مقابل أسّيّ
# المصدر  : src/chapters/ch33_diffusion.md:178
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch33_diffusion/004_ddpm_ddim_test_time_generation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : زمن التوليد: DDPM مقابل DDIM
# المصدر  : src/chapters/ch33_diffusion.md:210
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : 33.4 التوليد من ضجيج خالص
# المصدر  : src/chapters/ch33_diffusion.md:121
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : جدول التباين: خطّيّ مقابل أسّيّ
# المصدر  : src/chapters/ch33_diffusion.md:178
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : زمن التوليد: DDPM مقابل DDIM
# المصدر  : src/chapters/ch33_diffusion.md:210
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — نماذج الانتشار (Diffusion Models)
Keras 3 in Arabic — Diffusion

Book slug : keras-3-arabic-professional
Chapter   : ch-33-ch33-diffusion (lesson 33)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch33_diffusion/001_s33_3_keras.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch33_diffusion/002_s33_4_generation_noise.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : 33.4 التوليد من ضجيج خالص
# المصدر  : src/chapters/ch33_diffusion.md:121
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch33_diffusion/003_variance_linear_versus_exponential.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : جدول التباين: خطّيّ مقابل أسّيّ
# المصدر  : src/chapters/ch33_diffusion.md:178
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch33_diffusion/004_ddpm_ddim_test_time_generation.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : زمن التوليد: DDPM مقابل DDIM
# المصدر  : src/chapters/ch33_diffusion.md:210
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch33_diffusion.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : 33.4 التوليد من ضجيج خالص
# المصدر  : src/chapters/ch33_diffusion.md:121
# الترتيب : 2/4
# ──────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : جدول التباين: خطّيّ مقابل أسّيّ
# المصدر  : src/chapters/ch33_diffusion.md:178
# الترتيب : 3/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 33: نماذج الانتشار (Diffusion Models)
# القسم   : زمن التوليد: DDPM مقابل DDIM
# المصدر  : src/chapters/ch33_diffusion.md:210
# الترتيب : 4/4
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def generate(model, num_samples, steps=TIMESTEPS):
    """توليد صور من ضجيج خالص عبر إزالة تدريجيّة."""
    x = np.random.randn(num_samples, *IMAGE_SHAPE).astype("float32")
    for t in range(steps - 1, -1, -1):
        t_batch = np.full(num_samples, t, dtype="int32")
        pred_noise = model.predict([x, t_batch], verbose=0)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        # معادلة إزالة الضجيج
        x = (x - (1 - alpha) / np.sqrt(1 - alpha_bar) * pred_noise) / np.sqrt(alpha)
        # إضافة ضجيج (إلّا في الخطوة الأخيرة)
        if t > 0:
            x = x + np.sqrt(betas[t]) * np.random.randn(*x.shape).astype("float32")
    return np.clip(x, 0, 1)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import numpy as np

T = 1000

# ─── 1. جدول خطّيّ
betas_lin = np.linspace(1e-4, 0.02, T).astype("float64")
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# ─── 2. جدول أسّيّ (cosine schedule)
s = 0.008
steps = np.arange(T + 1) / T
f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cos = f / f[0]

for t in (1, 100, 500, 999):
    print(f"t={t:4d}  linear={alpha_bar_lin[t]:.4f}  cosine={alpha_bar_cos[t]:.4f}")
# النتيجة المتوقعة: الجدول الخطّيّ يفقد المعلومة أسرع في البداية؛
# الجدول الأسّيّ (cosine) يحفظ البنية أطول ثم يسرّع التدمير.

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-33-ch33-diffusion.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import time

def ddim_schedule(n_steps, t_max=1000):
    """يختار n_steps مواضع متساوية من الـ 1000 (جدول DDIM)."""
    return np.linspace(0, t_max - 1, n_steps, dtype=int)

def time_generation(n_steps, per_step_ms=12.0):
    """زمن توليد 10 صور بحسب عدد الخطوات."""
    return n_steps * per_step_ms / 1000.0

# ─── 1. مقارنة أعداد الخطوات
for n in (1000, 250, 50):
    steps = ddim_schedule(n)
    print(f"{n:5d} steps → {time_generation(n):7.2f}s لكلّ 10 صور")
# النتيجة المتوقعة: 12.0s مقابل 3.0s مقابل 0.6s — تسريع 20×.

# ─── 2. عدد النماذج المستخدمة: نفس النموذج في الحالتين
print(f"نموذج مُدرَّب واحد — DDIM يعيد استخدام خطوات مختارة: "
      f"{len(ddim_schedule(50))} من 1000")
