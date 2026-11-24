#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — مشاريع ختامية وخارطة الطريق
Professional TensorFlow — Capstones

Book slug : tensorflow-professional-ar
Chapter   : ch-24-ch24-capstones (lesson 24)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/tensorflow-professional-ar/code/ch-24-ch24-capstones.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow الاحترافي — مشاريع ختامية وخارطة الطريق
Professional TensorFlow — Capstones

Book slug : tensorflow-professional-ar
Chapter   : ch-24-ch24-capstones (lesson 24)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book/src/chapters/ch24_capstones.md
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)          # تكرار قدر الإمكان

IMG_SIZE = (224, 224)
BATCH = 32

# خط بيانات بسيط: في الواقع أضف cache/prefetch واختبر الأنابيب
train_ds = keras.utils.image_dataset_from_directory(
    "data/train", image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="int", shuffle=True, seed=42)
val_ds = keras.utils.image_dataset_from_directory(
    "data/val", image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="int", shuffle=False).prefetch(tf.data.AUTOTUNE)

# MobileNetV2 يتوقع قيمًا في المجال [-1, 1]
preprocess = keras.layers.Rescaling(1 / 127.5, offset=-1)
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.02),
    keras.layers.RandomZoom(0.1),
])

# المرحلة الأولى: قاعدة مجمّدة ورأس جديد
base = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = augment(inputs)
x = preprocess(x)
x = base(x, training=False)              # BatchNorm يبقى في وضع الاستدلال
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
model.fit(train_ds, validation_data=val_ds, epochs=15,
          callbacks=callbacks)

# المرحلة الثانية: فكّ تجميد آخر الكتل بمعدّل تعلّم صغير
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=8,
          callbacks=callbacks)

# SavedModel للنشر؛ ثم يُحوَّل إلى LiteRT على الجهاز الطرفي
model.export("artifacts/defect_classifier")

# ────────────────────────────────────────────────────────────────────────
# [002] book/src/chapters/ch24_capstones.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)

# بيانات اصطناعية: ثلاث سنوات يومية لمتجر واحد
rng = np.random.default_rng(42)
t = np.arange(3 * 365)
sales = (50 + 10 * np.sin(2 * np.pi * t / 7) + 0.01 * t
         + rng.normal(0, 3, size=t.shape)).astype("float32")

def make_windows(series, input_len=60, horizon=14):
    """يحوّل سلسلة إلى أزواج (ماضٍ، مستقبل) دون خلط الزمن."""
    X, y = [], []
    for i in range(len(series) - input_len - horizon + 1):
        X.append(series[i:i + input_len])
        y.append(series[i + input_len:i + input_len + horizon])
    return np.expand_dims(np.array(X), -1), np.array(y)

X, y = make_windows(sales)

# تقسيم زمني: آخر 20% للاختبار النهائي فقط
cut = int(len(X) * 0.8)
X_train, y_train = X[:cut], y[:cut]
X_test, y_test = X[cut:], y[cut:]

# خط الأساس: الأسبوع نفسه من الأسبوع الماضي
naive = X_test[:, -7:, 0]
mae_naive = np.mean(np.abs(naive - y_test[:, :7]))
print(f"MAE للخط الأساسي (أسبوع واحد): {mae_naive:.2f}")

# التطبيع من التدريب فقط، لا من الاختبار
mu, sigma = X_train.mean(), X_train.std()
X_train_n = (X_train - mu) / sigma
X_test_n = (X_test - mu) / sigma

model = keras.Sequential([
    keras.Input(shape=(60, 1)),
    keras.layers.GRU(64),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(14),              # أفق أربعة عشر يومًا
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mae", metrics=["mae"])
model.fit(X_train_n, y_train, validation_split=0.1,
          epochs=20, batch_size=64, verbose=0)

mae_model = model.evaluate(X_test_n, y_test, verbose=0)[0]
print(f"MAE للنموذج: {mae_model:.2f}")
# قارن دائمًا بالخط الأساسي؛ التفوق ليس مضمونًا

# ────────────────────────────────────────────────────────────────────────
# [003] book/src/chapters/ch24_capstones.md
# ────────────────────────────────────────────────────────────────────────
import re
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)

def normalize_arabic(text):
    """تنظيف محافظ: يوحّد الحروف ولا يمسّ النفي."""
    text = re.sub(r"[\u064B-\u0652]", "", text)   # التشكيل
    text = re.sub(r"ـ+", "", text)                 # التطويل
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[٠-٩]", lambda m: str(ord(m.group()) - 0x0660), text)
    text = re.sub(r"http\S+|@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

texts = np.array([
    "الخدمة ممتازة والموظفون لطفاء جدًا",
    "لم يعجبني المنتج، التأخير غير مقبول",
    "جيد لكن السعر مرتفع قليلًا",
    "لا أنصح به أبدًا",
], dtype=object)
labels = np.array([2, 0, 1, 0], dtype="int32")

# في الواقع: adapt على مجموعة التدريب فقط، لا على الاختبار
vectorizer = layers.TextVectorization(
    max_tokens=30000, output_mode="int", output_sequence_length=64,
    standardize=normalize_arabic)
vectorizer.adapt(texts)

model = keras.Sequential([
    keras.Input(shape=(1,), dtype="string"),
    vectorizer,
    layers.Embedding(30000, 128, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),   # سلبي، محايد، إيجابي
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()   # المخرجات تعتمد على المفردات الفعلية

# ────────────────────────────────────────────────────────────────────────
# [004] book/src/chapters/ch24_capstones.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)

# بيانات اصطناعية شديدة الاختلال (توضيحية فقط)
rng = np.random.default_rng(42)
n = 20000
X = rng.normal(size=(n, 12)).astype("float32")
p_true = 1 / (1 + np.exp(-(2.0 * X[:, 0] + X[:, 1] - 3.0)))
y = (rng.random(n) < 0.005 * p_true / p_true.mean()).astype("int32")

# في الواقع: تقسيم زمني بفجوة تغطي تأخر الوسم
cut = int(n * 0.7)
X_train, y_train = X[:cut], y[:cut]
X_val, y_val = X[cut:], y[cut:]

pos = int(y_train.sum())
neg = len(y_train) - pos
class_weight = {0: 1.0, 1: float(neg / max(pos, 1))}

model = keras.Sequential([
    keras.Input(shape=(12,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),               # logit واحد بلا sigmoid
])
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.AUC(curve="PR", from_logits=True,
                               name="pr_auc")],
)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=256, class_weight=class_weight,
          verbose=0)

# العتبة تُختار على التحقق وفق التكلفة: 5 لتفويت الاحتيال، 1 لإنذار كاذب
logits = model.predict(X_val, verbose=0).ravel()
probs = 1 / (1 + np.exp(-logits))
for tau in (0.2, 0.5, 0.8):
    pred = (probs >= tau).astype(int)
    fn = int(((y_val == 1) & (pred == 0)).sum())
    fp = int(((y_val == 0) & (pred == 1)).sum())
    print(f"tau={tau:.1f}  FN={fn}  FP={fp}  cost={5 * fn + fp}")

brier = np.mean((probs - y_val) ** 2)
print(f"Brier score: {brier:.4f}")      # أقل أفضل، لكنه يتأثر بالاختلال

# ────────────────────────────────────────────────────────────────────────
# [005] book/src/chapters/ch24_capstones.md
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
import keras
from keras import layers
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["normal", "defect"]  # الفئة الموجبة: defect

def read_split(path, shuffle):
    ds = keras.utils.image_dataset_from_directory(
        path,
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=(224, 224),
        batch_size=32,
        shuffle=shuffle,
        seed=42,
    )

    def prepare(images, labels):
        images = keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    return ds.map(prepare, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = read_split("data/split/train", shuffle=True)
val_ds = read_split("data/split/validation", shuffle=False)

backbone = keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
backbone.trainable = False
inputs = keras.Input(shape=(224, 224, 3), name="image")
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
probability = layers.Dense(1, activation="sigmoid", name="defect_probability")(x)
model = keras.Model(inputs, probability)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
)
callbacks = [keras.callbacks.EarlyStopping(
    monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True
)]
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# ────────────────────────────────────────────────────────────────────────
# [006] book/src/chapters/ch24_capstones.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import layers

# هذه المصفوفات تأتي من سجل تفاعلات مقسوم زمنيًا ومن سلبية مأخوذة بعناية.
# شكل كل من المعرّفات والوسوم: (عدد التفاعلات، 1).
num_users, num_items, factors = 5000, 8000, 64
user_input = keras.Input(shape=(), dtype="int32", name="user_id")
item_input = keras.Input(shape=(), dtype="int32", name="item_id")
user_vector = layers.Embedding(num_users, factors)(user_input)
item_vector = layers.Embedding(num_items, factors)(item_input)
score = layers.Dot(axes=1, name="compatibility")([user_vector, item_vector])
ranker = keras.Model({"user_id": user_input, "item_id": item_input}, score)
ranker.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc")],
)
ranker.fit(
    {"user_id": train_users, "item_id": train_items}, train_labels,
    validation_data=({"user_id": val_users, "item_id": val_items}, val_labels),
    batch_size=1024, epochs=8,
)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/tensorflow-professional-ar/notebooks/ch-24-ch24-capstones.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import keras

keras.utils.set_random_seed(42)          # تكرار قدر الإمكان

IMG_SIZE = (224, 224)
BATCH = 32

# خط بيانات بسيط: في الواقع أضف cache/prefetch واختبر الأنابيب
train_ds = keras.utils.image_dataset_from_directory(
    "data/train", image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="int", shuffle=True, seed=42)
val_ds = keras.utils.image_dataset_from_directory(
    "data/val", image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="int", shuffle=False).prefetch(tf.data.AUTOTUNE)

# MobileNetV2 يتوقع قيمًا في المجال [-1, 1]
preprocess = keras.layers.Rescaling(1 / 127.5, offset=-1)
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.02),
    keras.layers.RandomZoom(0.1),
])

# المرحلة الأولى: قاعدة مجمّدة ورأس جديد
base = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = augment(inputs)
x = preprocess(x)
x = base(x, training=False)              # BatchNorm يبقى في وضع الاستدلال
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
model.fit(train_ds, validation_data=val_ds, epochs=15,
          callbacks=callbacks)

# المرحلة الثانية: فكّ تجميد آخر الكتل بمعدّل تعلّم صغير
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=8,
          callbacks=callbacks)

# SavedModel للنشر؛ ثم يُحوَّل إلى LiteRT على الجهاز الطرفي
model.export("artifacts/defect_classifier")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/tensorflow-professional-ar/notebooks/ch-24-ch24-capstones.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)

# بيانات اصطناعية: ثلاث سنوات يومية لمتجر واحد
rng = np.random.default_rng(42)
t = np.arange(3 * 365)
sales = (50 + 10 * np.sin(2 * np.pi * t / 7) + 0.01 * t
         + rng.normal(0, 3, size=t.shape)).astype("float32")

def make_windows(series, input_len=60, horizon=14):
    """يحوّل سلسلة إلى أزواج (ماضٍ، مستقبل) دون خلط الزمن."""
    X, y = [], []
    for i in range(len(series) - input_len - horizon + 1):
        X.append(series[i:i + input_len])
        y.append(series[i + input_len:i + input_len + horizon])
    return np.expand_dims(np.array(X), -1), np.array(y)

X, y = make_windows(sales)

# تقسيم زمني: آخر 20% للاختبار النهائي فقط
cut = int(len(X) * 0.8)
X_train, y_train = X[:cut], y[:cut]
X_test, y_test = X[cut:], y[cut:]

# خط الأساس: الأسبوع نفسه من الأسبوع الماضي
naive = X_test[:, -7:, 0]
mae_naive = np.mean(np.abs(naive - y_test[:, :7]))
print(f"MAE للخط الأساسي (أسبوع واحد): {mae_naive:.2f}")

# التطبيع من التدريب فقط، لا من الاختبار
mu, sigma = X_train.mean(), X_train.std()
X_train_n = (X_train - mu) / sigma
X_test_n = (X_test - mu) / sigma

model = keras.Sequential([
    keras.Input(shape=(60, 1)),
    keras.layers.GRU(64),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(14),              # أفق أربعة عشر يومًا
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mae", metrics=["mae"])
model.fit(X_train_n, y_train, validation_split=0.1,
          epochs=20, batch_size=64, verbose=0)

mae_model = model.evaluate(X_test_n, y_test, verbose=0)[0]
print(f"MAE للنموذج: {mae_model:.2f}")
# قارن دائمًا بالخط الأساسي؛ التفوق ليس مضمونًا

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/tensorflow-professional-ar/notebooks/ch-24-ch24-capstones.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import re
import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)

def normalize_arabic(text):
    """تنظيف محافظ: يوحّد الحروف ولا يمسّ النفي."""
    text = re.sub(r"[\u064B-\u0652]", "", text)   # التشكيل
    text = re.sub(r"ـ+", "", text)                 # التطويل
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[٠-٩]", lambda m: str(ord(m.group()) - 0x0660), text)
    text = re.sub(r"http\S+|@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

texts = np.array([
    "الخدمة ممتازة والموظفون لطفاء جدًا",
    "لم يعجبني المنتج، التأخير غير مقبول",
    "جيد لكن السعر مرتفع قليلًا",
    "لا أنصح به أبدًا",
], dtype=object)
labels = np.array([2, 0, 1, 0], dtype="int32")

# في الواقع: adapt على مجموعة التدريب فقط، لا على الاختبار
vectorizer = layers.TextVectorization(
    max_tokens=30000, output_mode="int", output_sequence_length=64,
    standardize=normalize_arabic)
vectorizer.adapt(texts)

model = keras.Sequential([
    keras.Input(shape=(1,), dtype="string"),
    vectorizer,
    layers.Embedding(30000, 128, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),   # سلبي، محايد، إيجابي
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()   # المخرجات تعتمد على المفردات الفعلية

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/tensorflow-professional-ar/notebooks/ch-24-ch24-capstones.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras

keras.utils.set_random_seed(42)

# بيانات اصطناعية شديدة الاختلال (توضيحية فقط)
rng = np.random.default_rng(42)
n = 20000
X = rng.normal(size=(n, 12)).astype("float32")
p_true = 1 / (1 + np.exp(-(2.0 * X[:, 0] + X[:, 1] - 3.0)))
y = (rng.random(n) < 0.005 * p_true / p_true.mean()).astype("int32")

# في الواقع: تقسيم زمني بفجوة تغطي تأخر الوسم
cut = int(n * 0.7)
X_train, y_train = X[:cut], y[:cut]
X_val, y_val = X[cut:], y[cut:]

pos = int(y_train.sum())
neg = len(y_train) - pos
class_weight = {0: 1.0, 1: float(neg / max(pos, 1))}

model = keras.Sequential([
    keras.Input(shape=(12,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1),               # logit واحد بلا sigmoid
])
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.AUC(curve="PR", from_logits=True,
                               name="pr_auc")],
)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=256, class_weight=class_weight,
          verbose=0)

# العتبة تُختار على التحقق وفق التكلفة: 5 لتفويت الاحتيال، 1 لإنذار كاذب
logits = model.predict(X_val, verbose=0).ravel()
probs = 1 / (1 + np.exp(-logits))
for tau in (0.2, 0.5, 0.8):
    pred = (probs >= tau).astype(int)
    fn = int(((y_val == 1) & (pred == 0)).sum())
    fp = int(((y_val == 0) & (pred == 1)).sum())
    print(f"tau={tau:.1f}  FN={fn}  FP={fp}  cost={5 * fn + fp}")

brier = np.mean((probs - y_val) ** 2)
print(f"Brier score: {brier:.4f}")      # أقل أفضل، لكنه يتأثر بالاختلال

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/tensorflow-professional-ar/notebooks/ch-24-ch24-capstones.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
import keras
from keras import layers
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["normal", "defect"]  # الفئة الموجبة: defect

def read_split(path, shuffle):
    ds = keras.utils.image_dataset_from_directory(
        path,
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=(224, 224),
        batch_size=32,
        shuffle=shuffle,
        seed=42,
    )

    def prepare(images, labels):
        images = keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    return ds.map(prepare, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = read_split("data/split/train", shuffle=True)
val_ds = read_split("data/split/validation", shuffle=False)

backbone = keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
backbone.trainable = False
inputs = keras.Input(shape=(224, 224, 3), name="image")
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
probability = layers.Dense(1, activation="sigmoid", name="defect_probability")(x)
model = keras.Model(inputs, probability)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
)
callbacks = [keras.callbacks.EarlyStopping(
    monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True
)]
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/tensorflow-professional-ar/notebooks/ch-24-ch24-capstones.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import keras
from keras import layers

# هذه المصفوفات تأتي من سجل تفاعلات مقسوم زمنيًا ومن سلبية مأخوذة بعناية.
# شكل كل من المعرّفات والوسوم: (عدد التفاعلات، 1).
num_users, num_items, factors = 5000, 8000, 64
user_input = keras.Input(shape=(), dtype="int32", name="user_id")
item_input = keras.Input(shape=(), dtype="int32", name="item_id")
user_vector = layers.Embedding(num_users, factors)(user_input)
item_vector = layers.Embedding(num_items, factors)(item_input)
score = layers.Dot(axes=1, name="compatibility")([user_vector, item_vector])
ranker = keras.Model({"user_id": user_input, "item_id": item_input}, score)
ranker.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc")],
)
ranker.fit(
    {"user_id": train_users, "item_id": train_items}, train_labels,
    validation_data=({"user_id": val_users, "item_id": val_items}, val_labels),
    batch_size=1024, epochs=8,
)
