#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الشبكات التلافيفية CNN
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 12 — CNN

Book slug : pytorch-keras-deep-learning
Chapter   : ch-12-ch12 (lesson 12)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-12-ch12.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الشبكات التلافيفية CNN
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 12 — CNN

Book slug : pytorch-keras-deep-learning
Chapter   : ch-12-ch12 (lesson 12)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# صورة 4×4 بأرقام متتابعة يسهل تتبّعها بالعين
x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
# نواة 2×2: ترصد الفرق الأفقي (يمين موجب، يسار سالب)
k = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])


def conv_manual(x, k):
    """الالتفاف الحلقي: نفس المعادلة مكتوبةً بالحلقات."""
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = k.shape
    out = torch.zeros(n, c_out, h - kh + 1, w - kw + 1)
    for i in range(out.shape[2]):        # كل موضع رأسي
        for j in range(out.shape[3]):    # كل موضع أفقي
            patch = x[:, :, i:i + kh, j:j + kw]      # الرقعة (n, c_in, kh, kw)
            out[:, :, i, j] = (patch * k[None]).sum(dim=(1, 2))
    return out


manual = conv_manual(x, k)
library = F.conv2d(x, k)

print("manual:\n", manual)
print("library:\n", library)
print("max abs diff:", (manual - library).abs().max().item())
# الناتج: max abs diff: 0.0

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def out_size(i, k, s, p, d=1):
    """صيغة بُعد الخرج: نفس القاعدة المذكورة في النص."""
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


# مقارنة بين padding="same" والحشو اليدوي
conv_same = nn.Conv2d(3, 16, kernel_size=3, padding="same")
conv_manual = nn.Conv2d(3, 16, kernel_size=3, padding=1)

x = torch.randn(2, 3, 224, 224)
print("same   :", conv_same(x).shape)      # الناتج: torch.Size([2, 16, 224, 224])
print("manual :", conv_manual(x).shape)    # الناتج: torch.Size([2, 16, 224, 224])
print("formula:", out_size(224, 3, 1, 1))  # الناتج: 224

# الخطوة 2 تُنصّف البُعد
conv_s2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
print("stride2:", conv_s2(conv_same(x)).shape)  # الناتج: torch.Size([2, 32, 112, 112])
print("formula:", out_size(224, 3, 2, 1))      # الناتج: 112

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def receptive_field(layers):
    """layers: قائمة من (حجم النواة، الخطوة). يعيد (الحقل، القفزة)."""
    r, j = 1, 1
    for k, s in layers:
        r = r + (k - 1) * j   # الطبقة توسّع ما تراه الطبقة السابقة بمقدار قفزتها
        j = j * s             # القفزة تتضاعف بالخطوة
    return r, j


print(receptive_field([(3, 1), (3, 1), (3, 1)]))   # الناتج: (7, 1)
print(receptive_field([(3, 1), (3, 2), (3, 1)]))   # الناتج: (9, 2)
print(receptive_field([(7, 1)]))                   # الناتج: (7, 1)

# الطبقات الثلاث 3×3 تحتاج معاملات أكثر من-layer 7×7 واحدة
c3 = nn.Sequential(*[nn.Conv2d(3, 3, 3, padding=1) for _ in range(3)])
c7 = nn.Conv2d(3, 3, 7, padding=3)
print("params 3x3x3:", sum(p.numel() for p in c3.parameters()))  # الناتج: 252
print("params 7x7  :", sum(p.numel() for p in c7.parameters()))  # الناتج: 588

# التجميع: الأقصى والمتوسّط
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
print(nn.MaxPool2d(2)(x))    # الناتج: tensor([[[[4.]]]])
print(nn.AvgPool2d(2)(x))    # الناتج: tensor([[[[2.5000]]]])

# التجميع المتكيّف: يضمن الخرج المطلوب مهما كان حجم الدخل
adaptive = nn.AdaptiveAvgPool2d(1)
print(adaptive(torch.randn(2, 64, 31, 17)).shape)  # الناتج: torch.Size([2, 64, 1, 1])

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """شبكة تصنيف صغيرة: تدرّج في القنوات،-halving في المكان."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (N, 128, 1, 1) مهما كان حجم الصورة
            nn.Flatten(),               # (N, 128)
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # logits خام، بلا softmax
        )

    def forward(self, x):
        return self.head(self.features(x))


model = ConvNet(num_classes=10)
x = torch.randn(4, 3, 224, 224)
y = model(x)
print("logits shape:", y.shape)                 # الناتج: torch.Size([4, 10])
print("params:", sum(p.numel() for p in model.parameters()))
# الخسارة تُحسب مباشرة على الـlogits؛ لا تضيف softmax
loss = nn.CrossEntropyLoss()(y, torch.tensor([0, 1, 2, 3]))
print("loss:", round(loss.item(), 4))

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """كتلة ResNet أساسية: مساران يُجمعان عند النهاية."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # المسار السريع: 1×1 عند اختلاف الأبعاد أو الخطوة
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # الجمع العنصري بين المسارين
        return F.relu(out)


class MiniResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(x)


net = MiniResNet()
x = torch.randn(2, 3, 224, 224)
print("logits:", net(x).shape)   # الناتج: torch.Size([2, 10])

# تحقّق عملي من فكرة «الباقي»: لو صفّرنا أوزان الكتلة، يصبح F(x)=0 تقريباً
block = BasicBlock(64, 64).eval()
with torch.no_grad():
    for p in block.conv1.parameters():
        p.zero_()
    for p in block.bn1.parameters():
        p.zero_()
    for p in block.conv2.parameters():
        p.zero_()
    for p in block.bn2.parameters():
        p.zero_()
    y0 = block(x)
print("identity preserved:", torch.allclose(y0, F.relu(x), atol=1e-4))
# الناتج: identity preserved: True

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# Keras 3 متعدّد النواة: تأكّد من النواة المستخدمة قبل تشغيل الكود
print("keras backend:", keras.backend.backend())

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),

    layers.Conv2D(128, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),

    layers.GlobalAveragePooling2D(),     # ما يقابله AdaptiveAvgPool2d(1)
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),  # Keras تتوقّع احتمالات هنا
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.summary(line_length=80)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# ResNet50 مدرَّب على ImageNet، بدون رأس التصنيف
backbone = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,          # نحذف طبقة التصنيف الأصلية
    input_shape=(224, 224, 3),
    pooling=None,               # نُبقي خرائط الميزات مكانية بأنفسنا
)

inputs = keras.Input(shape=(224, 224, 3))
x = backbone(inputs, training=False)     # training=False يُجمّد BatchNorm
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax", name="fault_head")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.ops as ops

# رفع الأبعاد: تكرار بسيط مقابل الالتفاف المعكوس
up = nn.Upsample(scale_factor=2, mode="nearest")            # (N,C,H,W) -> (N,C,2H,2W)
deconv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
x = torch.randn(1, 64, 32, 32)
print(up(x).shape)         # الناتج: torch.Size([1, 64, 64, 64])
print(deconv(x).shape)     # الناتج: torch.Size([1, 32, 64, 64])

# من التصنيف إلى الكشف: نفس العمود الفقري، ورأس يBoxes
boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0], [12.0, 12.0, 62.0, 62.0], [80.0, 80.0, 120.0, 120.0]])
scores = torch.tensor([0.9, 0.8, 0.7])
print("IoU matrix:\n", ops.box_iou(boxes, boxes).round(decimals=2))
keep = ops.nms(boxes, scores, iou_threshold=0.5)
print("kept after NMS:", keep)   # الناتج: kept after NMS: tensor([0, 2])

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.body(x)).flatten(1)
        return self.fc(x)


net = TinyNet()
x = torch.randn(8, 3, 96, 96)
print(net(x).shape)                       # الناتج: torch.Size([8, 10])
print("params:", sum(p.numel() for p in net.parameters()))

net.pool = nn.AdaptiveAvgPool2d(4)
net.fc = nn.Linear(128 * 4 * 4, 10)
print(net(x).shape)                       # الناتج: torch.Size([8, 10])
print("params:", sum(p.numel() for p in net.parameters()))

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch12.html
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn


def conv2d_numpy(x, k, stride=1, pad=0):
    """x: (C,H,W) أو (N,C,H,W)، k: (Cout,Cin,Kh,Kw)."""
    x = np.asarray(x, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    n, c, h, w = x.shape
    cout, cin, kh, kw = k.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for ci in range(cin):
        for i in range(oh):
            for j in range(ow):
                patch = x[:, ci, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[:, :, i, j] += np.einsum("nkhw,khw->n", patch, k[None, ci, :, :, :])
    return out


rng = np.random.default_rng(0)
x = rng.standard_normal((1, 3, 5, 5)).astype(np.float32)
w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)

mine = conv2d_numpy(x, w, stride=2, pad=1)
layer = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
layer.weight.data = torch.from_numpy(w)
with torch.no_grad():
    theirs = layer(torch.from_numpy(x)).numpy()
print("shape mine :", mine.shape)
print("shape torch:", theirs.shape)
print("max abs diff:", np.abs(mine - theirs).max())

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# صورة 4×4 بأرقام متتابعة يسهل تتبّعها بالعين
x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
# نواة 2×2: ترصد الفرق الأفقي (يمين موجب، يسار سالب)
k = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])


def conv_manual(x, k):
    """الالتفاف الحلقي: نفس المعادلة مكتوبةً بالحلقات."""
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = k.shape
    out = torch.zeros(n, c_out, h - kh + 1, w - kw + 1)
    for i in range(out.shape[2]):        # كل موضع رأسي
        for j in range(out.shape[3]):    # كل موضع أفقي
            patch = x[:, :, i:i + kh, j:j + kw]      # الرقعة (n, c_in, kh, kw)
            out[:, :, i, j] = (patch * k[None]).sum(dim=(1, 2))
    return out


manual = conv_manual(x, k)
library = F.conv2d(x, k)

print("manual:\n", manual)
print("library:\n", library)
print("max abs diff:", (manual - library).abs().max().item())
# الناتج: max abs diff: 0.0

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def out_size(i, k, s, p, d=1):
    """صيغة بُعد الخرج: نفس القاعدة المذكورة في النص."""
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


# مقارنة بين padding="same" والحشو اليدوي
conv_same = nn.Conv2d(3, 16, kernel_size=3, padding="same")
conv_manual = nn.Conv2d(3, 16, kernel_size=3, padding=1)

x = torch.randn(2, 3, 224, 224)
print("same   :", conv_same(x).shape)      # الناتج: torch.Size([2, 16, 224, 224])
print("manual :", conv_manual(x).shape)    # الناتج: torch.Size([2, 16, 224, 224])
print("formula:", out_size(224, 3, 1, 1))  # الناتج: 224

# الخطوة 2 تُنصّف البُعد
conv_s2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
print("stride2:", conv_s2(conv_same(x)).shape)  # الناتج: torch.Size([2, 32, 112, 112])
print("formula:", out_size(224, 3, 2, 1))      # الناتج: 112

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


def receptive_field(layers):
    """layers: قائمة من (حجم النواة، الخطوة). يعيد (الحقل، القفزة)."""
    r, j = 1, 1
    for k, s in layers:
        r = r + (k - 1) * j   # الطبقة توسّع ما تراه الطبقة السابقة بمقدار قفزتها
        j = j * s             # القفزة تتضاعف بالخطوة
    return r, j


print(receptive_field([(3, 1), (3, 1), (3, 1)]))   # الناتج: (7, 1)
print(receptive_field([(3, 1), (3, 2), (3, 1)]))   # الناتج: (9, 2)
print(receptive_field([(7, 1)]))                   # الناتج: (7, 1)

# الطبقات الثلاث 3×3 تحتاج معاملات أكثر من-layer 7×7 واحدة
c3 = nn.Sequential(*[nn.Conv2d(3, 3, 3, padding=1) for _ in range(3)])
c7 = nn.Conv2d(3, 3, 7, padding=3)
print("params 3x3x3:", sum(p.numel() for p in c3.parameters()))  # الناتج: 252
print("params 7x7  :", sum(p.numel() for p in c7.parameters()))  # الناتج: 588

# التجميع: الأقصى والمتوسّط
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
print(nn.MaxPool2d(2)(x))    # الناتج: tensor([[[[4.]]]])
print(nn.AvgPool2d(2)(x))    # الناتج: tensor([[[[2.5000]]]])

# التجميع المتكيّف: يضمن الخرج المطلوب مهما كان حجم الدخل
adaptive = nn.AdaptiveAvgPool2d(1)
print(adaptive(torch.randn(2, 64, 31, 17)).shape)  # الناتج: torch.Size([2, 64, 1, 1])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """شبكة تصنيف صغيرة: تدرّج في القنوات،-halving في المكان."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (N, 128, 1, 1) مهما كان حجم الصورة
            nn.Flatten(),               # (N, 128)
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # logits خام، بلا softmax
        )

    def forward(self, x):
        return self.head(self.features(x))


model = ConvNet(num_classes=10)
x = torch.randn(4, 3, 224, 224)
y = model(x)
print("logits shape:", y.shape)                 # الناتج: torch.Size([4, 10])
print("params:", sum(p.numel() for p in model.parameters()))
# الخسارة تُحسب مباشرة على الـlogits؛ لا تضيف softmax
loss = nn.CrossEntropyLoss()(y, torch.tensor([0, 1, 2, 3]))
print("loss:", round(loss.item(), 4))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """كتلة ResNet أساسية: مساران يُجمعان عند النهاية."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # المسار السريع: 1×1 عند اختلاف الأبعاد أو الخطوة
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # الجمع العنصري بين المسارين
        return F.relu(out)


class MiniResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(x)


net = MiniResNet()
x = torch.randn(2, 3, 224, 224)
print("logits:", net(x).shape)   # الناتج: torch.Size([2, 10])

# تحقّق عملي من فكرة «الباقي»: لو صفّرنا أوزان الكتلة، يصبح F(x)=0 تقريباً
block = BasicBlock(64, 64).eval()
with torch.no_grad():
    for p in block.conv1.parameters():
        p.zero_()
    for p in block.bn1.parameters():
        p.zero_()
    for p in block.conv2.parameters():
        p.zero_()
    for p in block.bn2.parameters():
        p.zero_()
    y0 = block(x)
print("identity preserved:", torch.allclose(y0, F.relu(x), atol=1e-4))
# الناتج: identity preserved: True

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# Keras 3 متعدّد النواة: تأكّد من النواة المستخدمة قبل تشغيل الكود
print("keras backend:", keras.backend.backend())

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),

    layers.Conv2D(128, 3, padding="same", use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),

    layers.GlobalAveragePooling2D(),     # ما يقابله AdaptiveAvgPool2d(1)
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),  # Keras تتوقّع احتمالات هنا
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.summary(line_length=80)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers

# ResNet50 مدرَّب على ImageNet، بدون رأس التصنيف
backbone = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,          # نحذف طبقة التصنيف الأصلية
    input_shape=(224, 224, 3),
    pooling=None,               # نُبقي خرائط الميزات مكانية بأنفسنا
)

inputs = keras.Input(shape=(224, 224, 3))
x = backbone(inputs, training=False)     # training=False يُجمّد BatchNorm
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax", name="fault_head")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.ops as ops

# رفع الأبعاد: تكرار بسيط مقابل الالتفاف المعكوس
up = nn.Upsample(scale_factor=2, mode="nearest")            # (N,C,H,W) -> (N,C,2H,2W)
deconv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
x = torch.randn(1, 64, 32, 32)
print(up(x).shape)         # الناتج: torch.Size([1, 64, 64, 64])
print(deconv(x).shape)     # الناتج: torch.Size([1, 32, 64, 64])

# من التصنيف إلى الكشف: نفس العمود الفقري، ورأس يBoxes
boxes = torch.tensor([[10.0, 10.0, 60.0, 60.0], [12.0, 12.0, 62.0, 62.0], [80.0, 80.0, 120.0, 120.0]])
scores = torch.tensor([0.9, 0.8, 0.7])
print("IoU matrix:\n", ops.box_iou(boxes, boxes).round(decimals=2))
keep = ops.nms(boxes, scores, iou_threshold=0.5)
print("kept after NMS:", keep)   # الناتج: kept after NMS: tensor([0, 2])

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.body(x)).flatten(1)
        return self.fc(x)


net = TinyNet()
x = torch.randn(8, 3, 96, 96)
print(net(x).shape)                       # الناتج: torch.Size([8, 10])
print("params:", sum(p.numel() for p in net.parameters()))

net.pool = nn.AdaptiveAvgPool2d(4)
net.fc = nn.Linear(128 * 4 * 4, 10)
print(net(x).shape)                       # الناتج: torch.Size([8, 10])
print("params:", sum(p.numel() for p in net.parameters()))

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-12-ch12.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn


def conv2d_numpy(x, k, stride=1, pad=0):
    """x: (C,H,W) أو (N,C,H,W)، k: (Cout,Cin,Kh,Kw)."""
    x = np.asarray(x, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    n, c, h, w = x.shape
    cout, cin, kh, kw = k.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for ci in range(cin):
        for i in range(oh):
            for j in range(ow):
                patch = x[:, ci, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[:, :, i, j] += np.einsum("nkhw,khw->n", patch, k[None, ci, :, :, :])
    return out


rng = np.random.default_rng(0)
x = rng.standard_normal((1, 3, 5, 5)).astype(np.float32)
w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)

mine = conv2d_numpy(x, w, stride=2, pad=1)
layer = nn.Conv2d(3, 4, 3, stride=2, padding=1, bias=False)
layer.weight.data = torch.from_numpy(w)
with torch.no_grad():
    theirs = layer(torch.from_numpy(x)).numpy()
print("shape mine :", mine.shape)
print("shape torch:", theirs.shape)
print("max abs diff:", np.abs(mine - theirs).max())
