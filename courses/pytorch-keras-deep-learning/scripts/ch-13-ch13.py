#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التعلّم النقلي والتدرّج الدقيق
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 13 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-13-ch13 (lesson 13)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-13-ch13.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التعلّم النقلي والتدرّج الدقيق
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 13 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-13-ch13 (lesson 13)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch13.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 1) تحميل الأوزان المدرَّبة مسبقاً
weights = ResNet50_Weights.IMAGENET1K_V2
backbone = resnet50(weights=weights)

# 2) تجميد كل المعاملات: لن تتغيّر أثناء التدريب
for param in backbone.parameters():
    param.requires_grad = False

# 3) استبدال طبقة التصنيف بمُخرَج سمات (2048 قيمة)
backbone.fc = nn.Identity()

# 4) رأس جديد فوق العظام المجمّدة
head = nn.Sequential(
    nn.LazyLinear(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 6),
)

model = nn.Sequential(backbone, head)
print(sum(p.requires_grad for p in model.parameters()), "trainable tensors")
# الناتج: 4 trainable tensors

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch13.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras.applications import EfficientNetB0
from keras import layers

base = EfficientNetB0(
    include_top=False,          # نريد السمات لا طبقة التصنيف
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base.trainable = False        # تجميد العظام

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch13.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = nn.Identity()

head = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 6),
)


def build_param_groups(model, base_lr, head_lr, decay=0.9):
    """قسّم معاملات النموذج إلى مجموعات حسب العمق في العظام."""
    layer_ids = {id(p): i for i, p in enumerate(model.backbone.parameters())}
    num_layers = len(layer_ids)
    groups = {}
    for name, param in model.backbone.named_parameters():
        depth = layer_ids[id(param)]
        lr = base_lr * (decay ** (num_layers - 1 - depth))
        groups.setdefault(round(lr, 10), []).append(param)

    param_groups = [{"params": ps, "lr": lr} for lr, ps in groups.items()]
    param_groups.append({"params": list(model.head.parameters()), "lr": head_lr})
    return param_groups


model = nn.Sequential(backbone, head)
optimizer = torch.optim.AdamW(
    build_param_groups(model, base_lr=1e-5, head_lr=1e-3, decay=0.9),
    weight_decay=1e-4,
)
print(len(optimizer.param_groups), "learning-rate groups")
# الناتج: 7 learning-rate groups

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch13.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras.applications import ResNet50
from keras import layers

base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)                 #.training=False: لا تحديث إحصاءات BN
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(6, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# المرحلة 1: الرأس فقط
model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)

# المرحلة 2: افتح الكتلة الأخيرة فقط
for layer in base.layers[-10:]:
    layer.trainable = True
model.compile(keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# المرحلة 3: تدرّج كامل بمعدّل صغير جداً
base.trainable = True
model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch13.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras.applications import (
    EfficientNetB0, EfficientNetB3, MobileNetV3Small,
    ResNet50, ConvNeXtTiny, ViT,
)

models = {
    "effnet_b0": EfficientNetB0,
    "effnet_b3": EfficientNetB3,
    "mobilenet_v3": MobileNetV3Small,
    "resnet50": ResNet50,
    "convnext_tiny": ConvNeXtTiny,
    "vit": ViT,
}

for name, ctor in models.items():
    m = ctor(include_top=False, weights="imagenet")
    params = sum(p.numel() for p in m.parameters())
    print(f"{name:15s} {params/1e6:7.2f}M parameters")

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch13.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

model = nn.Sequential()
model.add_module("backbone", resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))
model.backbone.fc = nn.Identity()
model.add_module("head", nn.Sequential(
    nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6),
))

backbone_params = list(model.backbone.named_parameters())
third = len(backbone_params) // 3

groups = [
    {"params": [p for _, p in backbone_params[:third]], "lr": 1e-6},
    {"params": [p for _, p in backbone_params[third:]], "lr": 1e-5},
    {"params": list(model.head.parameters()), "lr": 1e-3},
]

optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
for g in optimizer.param_groups:
    print(f"lr={g['lr']:.0e} tensors={len(g['params'])}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-13-ch13.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 1) تحميل الأوزان المدرَّبة مسبقاً
weights = ResNet50_Weights.IMAGENET1K_V2
backbone = resnet50(weights=weights)

# 2) تجميد كل المعاملات: لن تتغيّر أثناء التدريب
for param in backbone.parameters():
    param.requires_grad = False

# 3) استبدال طبقة التصنيف بمُخرَج سمات (2048 قيمة)
backbone.fc = nn.Identity()

# 4) رأس جديد فوق العظام المجمّدة
head = nn.Sequential(
    nn.LazyLinear(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 6),
)

model = nn.Sequential(backbone, head)
print(sum(p.requires_grad for p in model.parameters()), "trainable tensors")
# الناتج: 4 trainable tensors

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-13-ch13.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import keras
from keras.applications import EfficientNetB0
from keras import layers

base = EfficientNetB0(
    include_top=False,          # نريد السمات لا طبقة التصنيف
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base.trainable = False        # تجميد العظام

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-13-ch13.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = nn.Identity()

head = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 6),
)


def build_param_groups(model, base_lr, head_lr, decay=0.9):
    """قسّم معاملات النموذج إلى مجموعات حسب العمق في العظام."""
    layer_ids = {id(p): i for i, p in enumerate(model.backbone.parameters())}
    num_layers = len(layer_ids)
    groups = {}
    for name, param in model.backbone.named_parameters():
        depth = layer_ids[id(param)]
        lr = base_lr * (decay ** (num_layers - 1 - depth))
        groups.setdefault(round(lr, 10), []).append(param)

    param_groups = [{"params": ps, "lr": lr} for lr, ps in groups.items()]
    param_groups.append({"params": list(model.head.parameters()), "lr": head_lr})
    return param_groups


model = nn.Sequential(backbone, head)
optimizer = torch.optim.AdamW(
    build_param_groups(model, base_lr=1e-5, head_lr=1e-3, decay=0.9),
    weight_decay=1e-4,
)
print(len(optimizer.param_groups), "learning-rate groups")
# الناتج: 7 learning-rate groups

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-13-ch13.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
from keras.applications import ResNet50
from keras import layers

base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)                 #.training=False: لا تحديث إحصاءات BN
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(6, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# المرحلة 1: الرأس فقط
model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)

# المرحلة 2: افتح الكتلة الأخيرة فقط
for layer in base.layers[-10:]:
    layer.trainable = True
model.compile(keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# المرحلة 3: تدرّج كامل بمعدّل صغير جداً
base.trainable = True
model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-13-ch13.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
from keras.applications import (
    EfficientNetB0, EfficientNetB3, MobileNetV3Small,
    ResNet50, ConvNeXtTiny, ViT,
)

models = {
    "effnet_b0": EfficientNetB0,
    "effnet_b3": EfficientNetB3,
    "mobilenet_v3": MobileNetV3Small,
    "resnet50": ResNet50,
    "convnext_tiny": ConvNeXtTiny,
    "vit": ViT,
}

for name, ctor in models.items():
    m = ctor(include_top=False, weights="imagenet")
    params = sum(p.numel() for p in m.parameters())
    print(f"{name:15s} {params/1e6:7.2f}M parameters")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-13-ch13.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

model = nn.Sequential()
model.add_module("backbone", resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))
model.backbone.fc = nn.Identity()
model.add_module("head", nn.Sequential(
    nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6),
))

backbone_params = list(model.backbone.named_parameters())
third = len(backbone_params) // 3

groups = [
    {"params": [p for _, p in backbone_params[:third]], "lr": 1e-6},
    {"params": [p for _, p in backbone_params[third:]], "lr": 1e-5},
    {"params": list(model.head.parameters()), "lr": 1e-3},
]

optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
for g in optimizer.param_groups:
    print(f"lr={g['lr']:.0e} tensors={len(g['params'])}")
