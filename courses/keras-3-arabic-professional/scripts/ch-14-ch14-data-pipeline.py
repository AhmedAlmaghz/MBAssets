#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — هندسة البيانات: من البايتات إلى الدفعات
Keras 3 in Arabic — Data Pipeline

Book slug : keras-3-arabic-professional
Chapter   : ch-14-ch14-data-pipeline (lesson 14)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-14-ch14-data-pipeline.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — هندسة البيانات: من البايتات إلى الدفعات
Keras 3 in Arabic — Data Pipeline

Book slug : keras-3-arabic-professional
Chapter   : ch-14-ch14-data-pipeline (lesson 14)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/keras-3-arabic-professional/code/ch-14-ch14-data-pipeline.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — هندسة البيانات: من البايتات إلى الدفعات
Keras 3 in Arabic — Data Pipeline

Book slug : keras-3-arabic-professional
Chapter   : ch-14-ch14-data-pipeline (lesson 14)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch14_data_pipeline/001_path_first_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الأول: بيانات في الذاكرة
# المصدر  : src/chapters/ch14_data_pipeline.md:64
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch14_data_pipeline/002_pydataset_path_second.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch14_data_pipeline/004_path_third_folders.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الثالث: مجلدات صور
# المصدر  : src/chapters/ch14_data_pipeline.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch14_data_pipeline/005_path_fourth_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الرابع: النصوص
# المصدر  : src/chapters/ch14_data_pipeline.md:205
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch14_data_pipeline/006_s14_5_example_solved_pipeline_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:244
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch14_data_pipeline/007_s14_5_example_solved_pipeline_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:266
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الأول: بيانات في الذاكرة
# المصدر  : src/chapters/ch14_data_pipeline.md:64
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الثالث: مجلدات صور
# المصدر  : src/chapters/ch14_data_pipeline.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الرابع: النصوص
# المصدر  : src/chapters/ch14_data_pipeline.md:205
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:244
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:266
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras 3 بالعربية — هندسة البيانات: من البايتات إلى الدفعات
Keras 3 in Arabic — Data Pipeline

Book slug : keras-3-arabic-professional
Chapter   : ch-14-ch14-data-pipeline (lesson 14)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/chapters/ch14_data_pipeline/001_path_first_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الأول: بيانات في الذاكرة
# المصدر  : src/chapters/ch14_data_pipeline.md:64
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [002] code/chapters/ch14_data_pipeline/002_pydataset_path_second.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] code/chapters/ch14_data_pipeline/004_path_third_folders.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الثالث: مجلدات صور
# المصدر  : src/chapters/ch14_data_pipeline.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [004] code/chapters/ch14_data_pipeline/005_path_fourth_text.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الرابع: النصوص
# المصدر  : src/chapters/ch14_data_pipeline.md:205
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [005] code/chapters/ch14_data_pipeline/006_s14_5_example_solved_pipeline_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:244
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [006] code/chapters/ch14_data_pipeline/007_s14_5_example_solved_pipeline_data.py
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:266
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [011] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [012] src/chapters/ch14_data_pipeline.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الأول: بيانات في الذاكرة
# المصدر  : src/chapters/ch14_data_pipeline.md:64
# الترتيب : 1/7
# ──────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الثالث: مجلدات صور
# المصدر  : src/chapters/ch14_data_pipeline.md:167
# الترتيب : 4/7
# ──────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : المسار الرابع: النصوص
# المصدر  : src/chapters/ch14_data_pipeline.md:205
# الترتيب : 5/7
# ──────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:244
# الترتيب : 6/7
# ──────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# الفصل 14: هندسة البيانات: من البايتات إلى الدفعات
# القسم   : 14.5 مثال محلول: خطّ بيانات بطيء
# المصدر  : src/chapters/ch14_data_pipeline.md:266
# الترتيب : 7/7
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras
import numpy as np

# بيانات اصطناعية صغيرة للتجربة
rng = np.random.default_rng(0)
x_train = rng.normal(size=(2000, 28, 28, 1)).astype("float32")
y_train = (x_train[:, 0, 0, 0] > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import keras

train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),     # تحجيم موحّد
    batch_size=32,
    labels="inferred",         # أسماء المجلدات = التسميات
    label_mode="categorical",  # one-hot
    shuffle=True,
    seed=42,
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32,
    labels="inferred",
    label_mode="categorical",
    shuffle=False,
)

model = keras.Sequential([
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
train_ds = keras.utils.text_dataset_from_directory(
    "data/train_text",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np

# محاكاة: كم يستغرق فكّ ضغط وتحجيم عيّنة واحدة؟
import io
import keras

t0 = time.perf_counter()
raw = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
small = keras.ops.image.resize(keras.ops.convert_to_tensor(raw), (224, 224))
t_resize = time.perf_counter() - t0

print(f"decode={t_decode*1e3:.2f}ms  resize={t_resize*1e3:.2f}ms")
print(f"الإنتاجية المتوقعة بـ8 أنوية: {8/max(t_decode+t_resize,1e-9):.0f} عيّنة/ثانية")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/keras-3-arabic-professional/notebooks/ch-14-ch14-data-pipeline.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import os

def predecode_to_memmap(src_dir, out_prefix, size=(224, 224), limit=None):
    """يقرأ الصور مرّة واحدة ويحفظها في ملف .npy (تحميل شبه فوري لاحقًا)."""
    from PIL import Image

    files = sorted(os.listdir(src_dir))[:limit]
    out = np.zeros((len(files), size[0], size[1], 3), dtype="uint8")
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(src_dir, fn)).convert("RGB").resize(size)
        out[i] = np.asarray(img)
    path = f"{out_prefix}.npy"
    np.save(path, out)
    print(f"حُفظ {len(files)} صورة في {path}")
    return path


# ثم التدريب يصبح:
# x = np.load("train_224.npy", mmap_mode="r")  ← قراءة شبه فورية
