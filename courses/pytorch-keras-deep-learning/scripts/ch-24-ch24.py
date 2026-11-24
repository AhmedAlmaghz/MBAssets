#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — النماذج الجاهزة وKerasHub
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 24 — Applications

Book slug : pytorch-keras-deep-learning
Chapter   : ch-24-ch24 (lesson 24)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-24-ch24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — النماذج الجاهزة وKerasHub
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 24 — Applications

Book slug : pytorch-keras-deep-learning
Chapter   : ch-24-ch24 (lesson 24)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch24.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

# 1) تحميل نموذج مدرَّب مسبقاً
base = keras.applications.EfficientNetV2B0(
    include_top=False,             # نُسقط المصنّف العلوي (1000 فئة)
    weights="imagenet",            # أو "imagenet" / None للتدريب من الصفر
    input_shape=(224, 224, 3),
    pooling="avg",                 # يجمع إلى متجه واحد
)

# 2) راقب عدد المعاملات
print("params:", base.count_params())
# الناتج: params: 7012512

# 3) بناء نموذجك فوقه
inputs = keras.layers.Input(shape=(224, 224, 3))
x = base(inputs, training=False)     # training=False ⇒ تجميد BatchNorm أثناء التدريب
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
               loss=keras.losses.SparseCategoricalCrossentropy(),
               metrics=["accuracy"])
model.summary(line_length=100)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch24.html
# ────────────────────────────────────────────────────────────────────────
# التدرّج الدقيق بطبقات مخصّصة لكل مجموعة معاملات (PyTorch)
import torch
from torch import nn

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# 1) تجميد كل شيء
for p in model.parameters():
    p.requires_grad = False

# 2) استبدال المصنّف العلوي
model.fc = nn.Linear(model.fc.in_features, 5)

# 3) فكّ تجميد الكتلة الأخيرة بمعدّل تعلّم أصغر
for p in model.layer4.parameters():
    p.requires_grad = True

backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
head_params = list(model.fc.parameters())

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},   # 10× أصغر
    {"params": head_params,    "lr": 1e-3},
], weight_decay=1e-4)

print("trainable tensors:", sum(1 for p in model.parameters() if p.requires_grad))

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch24.html
# ────────────────────────────────────────────────────────────────────────
# مثال: نموذج BERT جاهز للتقطيع (يتطلّب تثبيت keras_hub/keras-nlp)
# pip install -U keras-hub keras-nlp

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras_hub

classifier = keras_hub.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=3,
)
classifier.fit(train_ds, epochs=2)
print(classifier.evaluate(val_ds))

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch24.html
# ────────────────────────────────────────────────────────────────────────
# PyTorch + Hugging Face: نفس الفكرة
# pip install transformers datasets

from transformers import AutoModel, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# تصنيف مشاعر بثلاثة أسطر
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("I love this book!"))
# الناتج: [{'label': 'POSITIVE', 'score': 0.9998}]

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch24.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

base = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
                                           input_shape=(224, 224, 3), pooling="avg")
base.trainable = False                       # استخراج الميزات
print("frozen params:", base.count_params())  # 7012512

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    base,
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax"),
])
trainable = sum(keras.ops.size(w) for w in model.trainable_weights)
print("trainable in head:", trainable)        # 258570 تقريباً

model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

base.trainable = True                        # التدرّج الدقيق الكامل
model.compile(optimizer=keras.optimizers.AdamW(1e-5, weight_decay=1e-4),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# ملاحظة: أعد بناء النموذج أو استخدم جدول تعلّم أصغر

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-24-ch24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

# 1) تحميل نموذج مدرَّب مسبقاً
base = keras.applications.EfficientNetV2B0(
    include_top=False,             # نُسقط المصنّف العلوي (1000 فئة)
    weights="imagenet",            # أو "imagenet" / None للتدريب من الصفر
    input_shape=(224, 224, 3),
    pooling="avg",                 # يجمع إلى متجه واحد
)

# 2) راقب عدد المعاملات
print("params:", base.count_params())
# الناتج: params: 7012512

# 3) بناء نموذجك فوقه
inputs = keras.layers.Input(shape=(224, 224, 3))
x = base(inputs, training=False)     # training=False ⇒ تجميد BatchNorm أثناء التدريب
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
               loss=keras.losses.SparseCategoricalCrossentropy(),
               metrics=["accuracy"])
model.summary(line_length=100)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-24-ch24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# التدرّج الدقيق بطبقات مخصّصة لكل مجموعة معاملات (PyTorch)
import torch
from torch import nn

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

# 1) تجميد كل شيء
for p in model.parameters():
    p.requires_grad = False

# 2) استبدال المصنّف العلوي
model.fc = nn.Linear(model.fc.in_features, 5)

# 3) فكّ تجميد الكتلة الأخيرة بمعدّل تعلّم أصغر
for p in model.layer4.parameters():
    p.requires_grad = True

backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
head_params = list(model.fc.parameters())

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},   # 10× أصغر
    {"params": head_params,    "lr": 1e-3},
], weight_decay=1e-4)

print("trainable tensors:", sum(1 for p in model.parameters() if p.requires_grad))

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-24-ch24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# مثال: نموذج BERT جاهز للتقطيع (يتطلّب تثبيت keras_hub/keras-nlp)
# pip install -U keras-hub keras-nlp

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras_hub

classifier = keras_hub.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=3,
)
classifier.fit(train_ds, epochs=2)
print(classifier.evaluate(val_ds))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-24-ch24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# PyTorch + Hugging Face: نفس الفكرة
# pip install transformers datasets

from transformers import AutoModel, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# تصنيف مشاعر بثلاثة أسطر
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("I love this book!"))
# الناتج: [{'label': 'POSITIVE', 'score': 0.9998}]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-24-ch24.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

base = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",
                                           input_shape=(224, 224, 3), pooling="avg")
base.trainable = False                       # استخراج الميزات
print("frozen params:", base.count_params())  # 7012512

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Rescaling(1.0 / 255.0),
    base,
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax"),
])
trainable = sum(keras.ops.size(w) for w in model.trainable_weights)
print("trainable in head:", trainable)        # 258570 تقريباً

model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

base.trainable = True                        # التدرّج الدقيق الكامل
model.compile(optimizer=keras.optimizers.AdamW(1e-5, weight_decay=1e-4),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# ملاحظة: أعد بناء النموذج أو استخدم جدول تعلّم أصغر
