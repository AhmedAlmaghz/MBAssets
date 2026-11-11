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
