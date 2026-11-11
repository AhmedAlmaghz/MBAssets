# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الثاني: `PyDataset` — الطريق المحمول
# المصدر  : src/chapters/ch14_data_pipeline.md:93
# الترتيب : 2/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np


class SignalDataset(keras.utils.PyDataset):
    """مجموعة إشارات: تولّد موجات جيبية مع ضجيج، وتعيد دفعات جاهزة."""

    def __init__(self, samples=4096, batch_size=64, length=256, noise=0.1, seed=0):
        super().__init__()
        self.samples = samples
        self.batch_size = batch_size
        self.length = length
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        # مدّة الإشارة لكل عيّنة (0..1) — يُعاد توليدها في كل حقبة
        self.freqs = self.rng.uniform(0.5, 3.0, size=samples).astype("float32")
        self.phases = self.rng.uniform(0, 2 * np.pi, size=samples).astype("float32")

    def __len__(self):
        # عدد الدفعات لا عدد العيّنات
        return self.samples // self.batch_size

    def __getitem__(self, index):
        lo = index * self.batch_size
        hi = lo + self.batch_size
        t = np.linspace(0, 1, self.length, dtype="float32")[None, :]
        # يبثّ (broadcast) ترددات العيّنات على المحور الزمني
        sig = np.sin(2 * np.pi * self.freqs[lo:hi, None] * t + self.phases[lo:hi, None])
        sig = sig + self.noise * self.rng.normal(size=sig.shape).astype("float32")
        x = sig[:, :, None]          # (batch, length, 1)
        y = (self.freqs[lo:hi] > 1.5).astype("float32").reshape(-1, 1)  # (batch, 1) لمطابقة مخرَج النموذج
        return x.astype("float32"), y

    def on_epoch_end(self):
        # إعادة توليد الضجيج بين الحقبات: تعزيز بيانات implicit
        self.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))


train = SignalDataset(samples=2048, batch_size=64)
val = SignalDataset(samples=512, batch_size=64, seed=7)

model = keras.Sequential([
    keras.layers.Input(shape=(256, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train,
    validation_data=val,
    epochs=5,
    verbose=2,
)
