#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكات التوليد التنافسيّة (GAN)
Keras 3 in Arabic — Gan

Book slug : keras-3-arabic-professional
Chapter   : ch-32-ch32-gan (lesson 32)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-32-ch32-gan.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكات التوليد التنافسيّة (GAN)
Keras 3 in Arabic — Gan

Book slug : keras-3-arabic-professional
Chapter   : ch-32-ch32-gan (lesson 32)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-32-ch32-gan.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكات التوليد التنافسيّة (GAN)
Keras 3 in Arabic — Gan

Book slug : keras-3-arabic-professional
Chapter   : ch-32-ch32-gan (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch32_gan/001_s32_5_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.5 التنفيذ في Keras
# المصدر  : src/chapters/ch32_gan.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch32_gan/002_s32_9_diagnosis_stability_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.9 تشخيص استقرار التدريب
# المصدر  : src/chapters/ch32_gan.md:179
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch32_gan/003_detection_diversity_generator_collapse.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : كشف تنوّع المولّد (كشف انهيار الأنماط)
# المصدر  : src/chapters/ch32_gan.md:219
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch32_gan.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch32_gan.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch32_gan.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.5 التنفيذ في Keras
# المصدر  : src/chapters/ch32_gan.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.9 تشخيص استقرار التدريب
# المصدر  : src/chapters/ch32_gan.md:179
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : كشف تنوّع المولّد (كشف انهيار الأنماط)
# المصدر  : src/chapters/ch32_gan.md:219
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — شبكات التوليد التنافسيّة (GAN)
Keras 3 in Arabic — Gan

Book slug : keras-3-arabic-professional
Chapter   : ch-32-ch32-gan (lesson 32)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch32_gan/001_s32_5_keras.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.5 التنفيذ في Keras
# المصدر  : src/chapters/ch32_gan.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch32_gan/002_s32_9_diagnosis_stability_training.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.9 تشخيص استقرار التدريب
# المصدر  : src/chapters/ch32_gan.md:179
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch32_gan/003_detection_diversity_generator_collapse.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : كشف تنوّع المولّد (كشف انهيار الأنماط)
# المصدر  : src/chapters/ch32_gan.md:219
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch32_gan.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch32_gan.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch32_gan.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.5 التنفيذ في Keras
# المصدر  : src/chapters/ch32_gan.md:77
# الترتيب : 1/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : 32.9 تشخيص استقرار التدريب
# المصدر  : src/chapters/ch32_gan.md:179
# الترتيب : 2/3
# ──────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 32: شبكات التوليد التنافسيّة (GAN)
# القسم   : كشف تنوّع المولّد (كشف انهيار الأنماط)
# المصدر  : src/chapters/ch32_gan.md:219
# الترتيب : 3/3
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. إعدادات
LATENT_DIM = 64
IMAGE_SHAPE = (28, 28, 1)

# ─── 2. المولّد
generator_input = keras.Input(shape=(LATENT_DIM,))
x = keras.layers.Dense(7 * 7 * 64, activation="relu")(generator_input)
x = keras.layers.Reshape((7, 7, 64))(x)
x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
gen_out = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
generator = keras.Model(generator_input, gen_out, name="generator")

# ─── 3. المُميّز
disc_input = keras.Input(shape=IMAGE_SHAPE)
y = keras.layers.Conv2D(32, 3, strides=2, padding="same")(disc_input)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Conv2D(64, 3, strides=2, padding="same")(y)
y = keras.layers.LeakyReLU(negative_slope=0.2)(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dropout(0.3)(y)
disc_out = keras.layers.Dense(1, activation="sigmoid")(y)
discriminator = keras.Model(disc_input, disc_out, name="discriminator")

# ─── 4. بناء GAN الكاملة (المُميّز مجمّد عند تدريب المولّد)
discriminator.trainable = False
gan_input = keras.Input(shape=(LATENT_DIM,))
gan_out = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_out, name="gan")

# ─── 5. المُحسّنات
discriminator.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
                      loss="binary_crossentropy")
gan.compile(optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss="binary_crossentropy")

# ─── 6. حلقة التدريب
n = 500
x_data = np.random.rand(n, 28, 28, 1).astype("float32")
BATCH = 32
for epoch in range(5):
    for step in range(n // BATCH):
        # (أ) تدريب المُميّز على صور حقيقيّة ومولّدة
        real = x_data[np.random.randint(0, n, BATCH)]
        noise = np.random.randn(BATCH, LATENT_DIM).astype("float32")
        fake = generator.predict(noise, verbose=0)
        d_loss = discriminator.train_on_batch(
            np.concatenate([real, fake], axis=0),
            np.concatenate([np.ones((BATCH, 1)), np.zeros((BATCH, 1))]))
        # (ب) تدريب المولّد (المُميّز مجمّد)
        g_loss = gan.train_on_batch(noise, np.ones((BATCH, 1)))
    print(f"epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")
# النتيجة المتوقعة: d_loss و g_loss يتحرّكان ضمن نطاق؛
# إن انهار أحدَهما إلى صفر فذلك مؤشّر انهيار الأنماط.

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def diagnose_gan(d_losses, g_losses):
    """تشخيص استقرار GAN من تاريخ الخسارة."""
    d = np.array(d_losses[-10:])
    g = np.array(g_losses[-10:])
    diagnosis = []
    # 1. المُميّز يهيمن
    if d.mean() < 0.05:
        diagnosis.append("المُميّز يهيمن — قلّل خطواته أو قوّي المولّد")
    # 2. المولّد خدع بالكامل
    if g.mean() < 0.05:
        diagnosis.append("المولّد خدع المُميّز — راجع توازن البنيتين")
    # 3. تذبذب مرتفع
    if d.std() > 0.5 or g.std() > 0.5:
        diagnosis.append("تذبذب مرتفع — جرّب WGAN-GP أو خفّض lr")
    # 4. انهيار الأنماط (تقريبيّ)
    if len(set(np.round(d[-5:], 3))) == 1:
        diagnosis.append("الخسارة ثابتة — انهيار أنماط محتمل")
    return diagnosis or ["التدريب مستقرّ"]

d_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
g_hist = np.abs(np.random.randn(50)) * 0.3 + 0.6
for msg in diagnose_gan(d_hist, g_hist):
    print("-", msg)
# النتيجة المتوقعة: تشخيص واحد أو أكثر بحسب المنحنيات.

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-32-ch32-gan.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

keras.utils.set_random_seed(42)

# ─── 1. مولّد مصغّر (محاكاة)
generator = keras.Sequential([keras.Input(shape=(32,)),
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dense(8)])
# ─── 2. توليد 200 عيّنة من ضجيج عشوائيّ
z = np.random.randn(200, 32).astype("float32")
samples = np.array(generator.predict(z, verbose=0))

# ─── 3. قياس التنوّع: متوسّط المسافة بين عيّنتين عشوائيّتين
rng = np.random.default_rng(0)
idx = rng.choice(200, size=(500, 2), replace=True)   # أزواج عشوائية
dists = np.linalg.norm(samples[idx[:, 0]] - samples[idx[:, 1]], axis=1)
print(f"متوسّط المسافة = {dists.mean():.4f}, الانحراف = {dists.std():.4f}")

# ─── 4. التشخيص
if dists.mean() < 1e-3:
    print("تنبيه: تنوّع شبه معدوم — انهيار أنماط")
elif dists.std() < 1e-3:
    print("تنبيه: كلّ العيّنات متطابقة تقريبًا — انهيار أنماط مؤكّد")
else:
    print("التنوّع سليم")
# النتيجة المتوقعة: تنوّع سليم في البداية؛ راقبه كلّ عدة دورات.
