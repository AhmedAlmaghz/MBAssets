#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — خط البيانات في Keras
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 23 — tf.data

Book slug : pytorch-keras-deep-learning
Chapter   : ch-23-ch23 (lesson 23)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-23-ch23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — خط البيانات في Keras
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 23 — tf.data

Book slug : pytorch-keras-deep-learning
Chapter   : ch-23-ch23 (lesson 23)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch23.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np

# أبسط حالة ممكنة: كل شيء في الذاكرة
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    validation_split=0.1,
                    epochs=10,
                    batch_size=128,
                    verbose=0)
print("last val acc:", round(history.history["val_accuracy"][-1], 4))

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch23.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
from pathlib import Path

class ImageFolderDataset(keras.utils.PyDataset):
    """يقرأ الصور من مجلدات: data/class_a/*.jpg، data/class_b/*.jpg ..."""

    def __init__(self, root, image_size=(64, 64), augment=False, seed=0):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = []
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        for label, cls in enumerate(self.classes):
            for img in sorted((self.root / cls).glob("*.jpg")):
                self.samples.append((img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(self.image_size)
        arr = np.asarray(img, dtype="float32") / 255.0
        if self.augment:                      # انعكاس أفقي بسيط
            if self.rng.random() < 0.5:
                arr = arr[:, ::-1, :]
        return arr, label

train_ds = ImageFolderDataset("data/train", augment=True)
val_ds = ImageFolderDataset("data/val")
print("train:", len(train_ds), "classes:", train_ds.classes)

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch23.html
# ────────────────────────────────────────────────────────────────────────
import keras

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal", seed=1),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=2),
    keras.layers.RandomZoom(height_factor=0.1, factor=0.1, seed=3),
    keras.layers.RandomRotation(0.05, seed=4),
], name="augment")

# داخل النموذج: يُطبَّق فقط أثناء التدريب (training=True)
model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    augment,
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])

# للتأكّد من أنّ التعزيز لا يعمل في الاستدلال
model.predict(np.zeros((2, 64, 64, 3), dtype="float32"), verbose=0)
print("model built:", model.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch23.html
# ────────────────────────────────────────────────────────────────────────
import keras

# طبقة تقطيع وتحويل: تتعلّم المعجم من التدريب فقط
vectorizer = keras.layers.TextVectorization(max_tokens=20_000,
                                            output_sequence_length=200,
                                            standardize="lower_and_strip_punctuation")
vectorizer.adapt(["نص تجريبي للتجربة", "جملة أخرى للتحقق"])

out = vectorizer(["مرحبا بالعالم!"]).numpy()
print("shape:", out.shape)   # (1, 200) — مملوءة بـ 0 بعد حدود المحتوى
print("tokens:", vectorizer.get_vocabulary()[:8])

# استخدامه داخل نموذج
model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype=keras.string),
    vectorizer,
    keras.layers.Embedding(input_dim=20_000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1),
])

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch23.html
# ────────────────────────────────────────────────────────────────────────
# نفس الفكرة بالمكتبتين

# PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

# Keras
model.fit(dataset, batch_size=64, shuffle=True, drop_last=True)   # عبر PyDataset

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch23.html
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
import keras

# خطّ A: التحسين داخل __getitem__ (على المعالج)
class SlowDataset(keras.utils.PyDataset):
    def __init__(self, x, y, aug=False):
        super().__init__()
        self.x, self.y, self.aug = x, y, aug
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        img = self.x[i]
        if self.aug:
            img = np.flip(img, axis=1)          # işlem على المعالج
        return img, self.y[i]

# خطّ B: التحسين داخل النموذج (على الجهاز)
def build(augment_in_model: bool):
    layers = [keras.layers.Input(shape=(64, 64, 3))]
    if augment_in_model:
        layers.append(keras.layers.RandomFlip("horizontal", seed=1))
    layers += [keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
               keras.layers.Flatten(), keras.layers.Dense(10)]
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m

x = np.random.rand(2000, 64, 64, 3).astype("float32")
y = np.random.randint(0, 10, 2000)

for name, ds, model in (("CPU augment", SlowDataset(x, y, True), build(False)),
                        ("GPU augment", SlowDataset(x, y, False), build(True))):
    t0 = time.perf_counter()
    model.fit(ds, epochs=2, verbose=0)
    print(f"{name}: {time.perf_counter() - t0:.2f}s")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-23-ch23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np

# أبسط حالة ممكنة: كل شيء في الذاكرة
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    validation_split=0.1,
                    epochs=10,
                    batch_size=128,
                    verbose=0)
print("last val acc:", round(history.history["val_accuracy"][-1], 4))

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-23-ch23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
from pathlib import Path

class ImageFolderDataset(keras.utils.PyDataset):
    """يقرأ الصور من مجلدات: data/class_a/*.jpg، data/class_b/*.jpg ..."""

    def __init__(self, root, image_size=(64, 64), augment=False, seed=0):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = []
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        for label, cls in enumerate(self.classes):
            for img in sorted((self.root / cls).glob("*.jpg")):
                self.samples.append((img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(self.image_size)
        arr = np.asarray(img, dtype="float32") / 255.0
        if self.augment:                      # انعكاس أفقي بسيط
            if self.rng.random() < 0.5:
                arr = arr[:, ::-1, :]
        return arr, label

train_ds = ImageFolderDataset("data/train", augment=True)
val_ds = ImageFolderDataset("data/val")
print("train:", len(train_ds), "classes:", train_ds.classes)

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-23-ch23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import keras

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal", seed=1),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=2),
    keras.layers.RandomZoom(height_factor=0.1, factor=0.1, seed=3),
    keras.layers.RandomRotation(0.05, seed=4),
], name="augment")

# داخل النموذج: يُطبَّق فقط أثناء التدريب (training=True)
model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    augment,
    keras.layers.Rescaling(1.0 / 255.0),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])

# للتأكّد من أنّ التعزيز لا يعمل في الاستدلال
model.predict(np.zeros((2, 64, 64, 3), dtype="float32"), verbose=0)
print("model built:", model.output_shape)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-23-ch23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras

# طبقة تقطيع وتحويل: تتعلّم المعجم من التدريب فقط
vectorizer = keras.layers.TextVectorization(max_tokens=20_000,
                                            output_sequence_length=200,
                                            standardize="lower_and_strip_punctuation")
vectorizer.adapt(["نص تجريبي للتجربة", "جملة أخرى للتحقق"])

out = vectorizer(["مرحبا بالعالم!"]).numpy()
print("shape:", out.shape)   # (1, 200) — مملوءة بـ 0 بعد حدود المحتوى
print("tokens:", vectorizer.get_vocabulary()[:8])

# استخدامه داخل نموذج
model = keras.Sequential([
    keras.layers.Input(shape=(1,), dtype=keras.string),
    vectorizer,
    keras.layers.Embedding(input_dim=20_000, output_dim=64, mask_zero=True),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1),
])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-23-ch23.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# نفس الفكرة بالمكتبتين

# PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

# Keras
model.fit(dataset, batch_size=64, shuffle=True, drop_last=True)   # عبر PyDataset

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-23-ch23.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import time
import numpy as np
import keras

# خطّ A: التحسين داخل __getitem__ (على المعالج)
class SlowDataset(keras.utils.PyDataset):
    def __init__(self, x, y, aug=False):
        super().__init__()
        self.x, self.y, self.aug = x, y, aug
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        img = self.x[i]
        if self.aug:
            img = np.flip(img, axis=1)          # işlem على المعالج
        return img, self.y[i]

# خطّ B: التحسين داخل النموذج (على الجهاز)
def build(augment_in_model: bool):
    layers = [keras.layers.Input(shape=(64, 64, 3))]
    if augment_in_model:
        layers.append(keras.layers.RandomFlip("horizontal", seed=1))
    layers += [keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
               keras.layers.Flatten(), keras.layers.Dense(10)]
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m

x = np.random.rand(2000, 64, 64, 3).astype("float32")
y = np.random.randint(0, 10, 2000)

for name, ds, model in (("CPU augment", SlowDataset(x, y, True), build(False)),
                        ("GPU augment", SlowDataset(x, y, False), build(True))):
    t0 = time.perf_counter()
    model.fit(ds, epochs=2, verbose=0)
    print(f"{name}: {time.perf_counter() - t0:.2f}s")
