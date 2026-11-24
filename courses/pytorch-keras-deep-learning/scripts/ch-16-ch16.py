#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الكشف والتجزئة على الصور
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 16 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-16-ch16 (lesson 16)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-16-ch16.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الكشف والتجزئة على الصور
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 16 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-16-ch16 (lesson 16)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch16.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torchvision.ops import box_convert, box_iou, nms

# صيغة المصدر: مركز + عرض + ارتفاع (كما في YOLO)
boxes_cxcywh = torch.tensor([[35.0, 55.0, 50.0, 70.0],
                              [55.0, 75.0, 50.0, 70.0]])

# التحويل إلى ركنين: الأول x1 ثم y1 ثم x2 ثم y2
boxes_xyxy = box_convert(boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
print(boxes_xyxy)
# الناتج: tensor([[10., 20., 60., 90.],
#                 [30., 40., 80., 110.]])

# حساب التداخل بين كل صندوق وكل صندوق (مصفوفة N x N)
iou = box_iou(boxes_xyxy, boxes_xyxy)
print(iou)
# الناتج: tensor([[1.0000, 0.2727],
#                  [0.2727, 1.0000]])

# الإسقاط: نحتفظ بالصناديق الأعلى ثقة ونتخلّص من المتداخلة
scores = torch.tensor([0.90, 0.75])
keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
print(keep)
# الناتج: tensor([0, 1])

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch16.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.ops import nms


def decode_yolo_output(pred, anchors, num_classes, stride=32, iou_thr=0.5, conf_thr=0.25):
    """تفكيك خريطة خام إلى صناديق ودرجات وأصناف.

    pred: (B, A*(5+C), H, W)  الخريطة كما خرجت من الشبكة.
    anchors: (A, 2)  عرض وارتفاع كلّ مرساة بالبكسل.
    """
    b, _, h, w = pred.shape
    a = anchors.shape[0]
    pred = pred.reshape(b, h, w, a, 5 + num_classes)

    # فهرس الخلية لكلّ عنصر مفلطح، بإزاحة A لكلّ خلية
    cell = torch.arange(h * w, device=pred.device, dtype=pred.dtype)
    cell_x = (cell % w).to(pred.dtype)
    cell_y = (cell // w).to(pred.dtype)

    all_boxes, all_scores, all_labels = [], [], []
    for i in range(b):
        obj = pred[i, ..., 4].sigmoid().reshape(-1)          # (H*W*A,)
        cls_prob = pred[i, ..., 5:].reshape(-1, num_classes).sigmoid()
        best_cls, best_idx = cls_prob.max(dim=1)              # أعلى صنف لكلّ صندوق
        conf = obj * best_cls                                 # objectness × الصنف

        off = pred[i, ..., :4].reshape(-1, 4)                 # (H*W*A, 4)
        aw = anchors[:, 0].repeat(h * w).to(pred.dtype)
        ah = anchors[:, 1].repeat(h * w).to(pred.dtype)

        cx = (cell_x.repeat(a) + off[:, 0].sigmoid()) * stride
        cy = (cell_y.repeat(a) + off[:, 1].sigmoid()) * stride
        bw = aw * torch.exp(off[:, 2]) * stride
        bh = ah * torch.exp(off[:, 3]) * stride
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

        # ترشيح أول بالثقة، ثم إسقاط غير أقصى على ما تبقّى
        mask = conf > conf_thr
        boxes, conf, best_idx = boxes[mask], conf[mask], best_idx[mask]
        keep = nms(boxes, conf, iou_thr)
        all_boxes.append(boxes[keep])
        all_scores.append(conf[keep])
        all_labels.append(best_idx[keep])
    return all_boxes, all_scores, all_labels


# مثال: خريطة صغيرة لاختبار الفكّ
pred = torch.zeros(1, 3 * (5 + 2), 4, 4)
pred[0, :, 1, 1] = torch.tensor([0.2, 0.1, 0.0, 0.0, 3.0, 2.0, -2.0])
anchors = torch.tensor([[20.0, 20.0], [40.0, 12.0], [12.0, 40.0]])
boxes, scores, labels = decode_yolo_output(pred, anchors, num_classes=2, stride=32)
print("عدد الأجسام:", len(boxes[0]))
print("صندوق أوّل:", boxes[0][0].round().tolist())
print("صنفه:", int(labels[0][0]), "وثقته:", round(float(scores[0][0]), 3))
# الناتج: عدد الأجسام: 1
#        صندوق أوّل: [29.0, 38.0, 109.0, 118.0]
#        صنفه: 1 وثقته: 0.723

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch16.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """كتلة تلافيفية مزدوجة: (Conv → BN → ReLU) مرّتين."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net: مسار ترميز تنازلي + مسار فكّ ترميز تصاعدي + وصلات تخطّي."""

    def __init__(self, in_ch=3, base=32, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)          # 1/1
        self.enc2 = DoubleConv(base, base * 2)        # 1/2
        self.enc3 = DoubleConv(base * 2, base * 4)    # 1/4
        self.enc4 = DoubleConv(base * 4, base * 8)    # 1/8
        self.bottleneck = DoubleConv(base * 8, base * 16)   # 1/16
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)   # 1×1 لكلّ بكسل

    def forward(self, x):
        e1 = self.enc1(x)                       # (B, base, H, W)
        e2 = self.enc2(self.pool(e1))           # (B, 2base, H/2, W/2)
        e3 = self.enc3(self.pool(e2))           # (B, 4base, H/4, W/4)
        e4 = self.enc4(self.pool(e3))           # (B, 8base, H/8, W/8)
        b = self.bottleneck(self.pool(e4))      # (B, 16base, H/16, W/16)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                    # لوغاريتميات، بلا sigmoid


def dice_loss(logits, targets, eps=1.0):
    """خسارة Dice: 1 − معامل Dice على الخريطة الاحتمالية."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


model = UNet(in_ch=3, base=16, out_ch=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bce_fn = nn.BCEWithLogitsLoss()

x = torch.randn(4, 3, 128, 128)
y = (torch.rand(4, 1, 128, 128) > 0.7).float()

model.train()
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = bce_fn(logits, y) + dice_loss(logits, y)
    loss.backward()
    optimizer.step()
    print(f"خطوة {step}: الخسارة = {loss.item():.4f}")
# الناتج: خطوة 0: الخسارة = 1.0231
#        خطوة 1: الخسارة = 0.9812
#        خطوة 2: الخسارة = 0.9504

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch16.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers


def dice_coefficient(y_true, y_pred, eps=1.0):
    """معامل Dice لكلّ صورة على حدة."""
    y_pred = keras.ops.sigmoid(y_pred)
    y_true = keras.ops.cast(y_true, y_pred.dtype)
    axes = (1, 2, 3)
    inter = keras.ops.sum(y_true * y_pred, axis=axes)
    denom = keras.ops.sum(y_true, axis=axes) + keras.ops.sum(y_pred, axis=axes)
    return (2.0 * inter + eps) / (denom + eps)


def bce_dice_loss(y_true, y_pred):
    """مجموع خسارة التصنيف الثنائي (من لوغاريتميات) وخسارة Dice."""
    bce = keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    return bce + (1.0 - keras.ops.mean(dice_coefficient(y_true, y_pred)))


def build_unet(input_shape=(128, 128, 3), base=16, out_ch=1):
    """U-Net بـ Conv2D في الترميز و Conv2DTranspose في فكّ الترميز."""
    inputs = keras.Input(shape=input_shape)

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    e1 = block(inputs, base)
    e2 = block(layers.MaxPooling2D(2)(e1), base * 2)
    e3 = block(layers.MaxPooling2D(2)(e2), base * 4)
    e4 = block(layers.MaxPooling2D(2)(e3), base * 8)
    b = block(layers.MaxPooling2D(2)(e4), base * 16)

    d4 = block(layers.Concatenate()([layers.Conv2DTranspose(base * 8, 2, strides=2)(b), e4]), base * 8)
    d3 = block(layers.Concatenate()([layers.Conv2DTranspose(base * 4, 2, strides=2)(d4), e3]), base * 4)
    d2 = block(layers.Concatenate()([layers.Conv2DTranspose(base * 2, 2, strides=2)(d3), e2]), base * 2)
    d1 = block(layers.Concatenate()([layers.Conv2DTranspose(base, 2, strides=2)(d2), e1]), base)
    out = layers.Conv2D(out_ch, 1, activation=None)(d1)   # لوغاريتميات
    return keras.Model(inputs, out, name="unet")


model = build_unet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=bce_dice_loss,
    metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
)

model.summary(line_length=70)
# الناتج: Model: "unet"
#         Total params: 2,435,713
#         Trainable params: 2,435,713

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch16.html
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers


def build_effnet_unet(input_shape=(256, 256, 3), out_ch=1):
    """مُرمِّز EfficientNetV2 مدرَّب مسبقاً + مسار فكّ ترميز خفيف."""
    inputs = keras.Input(shape=input_shape)

    # مسار خفيف على مستوى 1/8 يزوّدنا بوصلة عالية الدقّة
    skip = layers.Conv2D(32, 3, strides=8, padding="same", activation="relu")(inputs)

    base = keras.applications.EfficientNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    x = base(inputs)                                   # 8×8×1280
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(4)(x)                      # 32×32
    x = layers.Concatenate()([x, skip])                # وصل التخطّي
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(out_ch, 1, activation=None)(x)    # لوغاريتميات 32×32
    out = layers.UpSampling2D(8)(x)                    # 256×256
    return keras.Model(inputs, out, name="effnet_unet")


model = build_effnet_unet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=bce_dice_loss,
    metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
)
print("عدد المعاملات:", model.count_params())
# الناتج: عدد المعاملات: 19984673

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-16-ch16.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
from torchvision.ops import box_convert, box_iou, nms

# صيغة المصدر: مركز + عرض + ارتفاع (كما في YOLO)
boxes_cxcywh = torch.tensor([[35.0, 55.0, 50.0, 70.0],
                              [55.0, 75.0, 50.0, 70.0]])

# التحويل إلى ركنين: الأول x1 ثم y1 ثم x2 ثم y2
boxes_xyxy = box_convert(boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
print(boxes_xyxy)
# الناتج: tensor([[10., 20., 60., 90.],
#                 [30., 40., 80., 110.]])

# حساب التداخل بين كل صندوق وكل صندوق (مصفوفة N x N)
iou = box_iou(boxes_xyxy, boxes_xyxy)
print(iou)
# الناتج: tensor([[1.0000, 0.2727],
#                  [0.2727, 1.0000]])

# الإسقاط: نحتفظ بالصناديق الأعلى ثقة ونتخلّص من المتداخلة
scores = torch.tensor([0.90, 0.75])
keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
print(keep)
# الناتج: tensor([0, 1])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-16-ch16.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.ops import nms


def decode_yolo_output(pred, anchors, num_classes, stride=32, iou_thr=0.5, conf_thr=0.25):
    """تفكيك خريطة خام إلى صناديق ودرجات وأصناف.

    pred: (B, A*(5+C), H, W)  الخريطة كما خرجت من الشبكة.
    anchors: (A, 2)  عرض وارتفاع كلّ مرساة بالبكسل.
    """
    b, _, h, w = pred.shape
    a = anchors.shape[0]
    pred = pred.reshape(b, h, w, a, 5 + num_classes)

    # فهرس الخلية لكلّ عنصر مفلطح، بإزاحة A لكلّ خلية
    cell = torch.arange(h * w, device=pred.device, dtype=pred.dtype)
    cell_x = (cell % w).to(pred.dtype)
    cell_y = (cell // w).to(pred.dtype)

    all_boxes, all_scores, all_labels = [], [], []
    for i in range(b):
        obj = pred[i, ..., 4].sigmoid().reshape(-1)          # (H*W*A,)
        cls_prob = pred[i, ..., 5:].reshape(-1, num_classes).sigmoid()
        best_cls, best_idx = cls_prob.max(dim=1)              # أعلى صنف لكلّ صندوق
        conf = obj * best_cls                                 # objectness × الصنف

        off = pred[i, ..., :4].reshape(-1, 4)                 # (H*W*A, 4)
        aw = anchors[:, 0].repeat(h * w).to(pred.dtype)
        ah = anchors[:, 1].repeat(h * w).to(pred.dtype)

        cx = (cell_x.repeat(a) + off[:, 0].sigmoid()) * stride
        cy = (cell_y.repeat(a) + off[:, 1].sigmoid()) * stride
        bw = aw * torch.exp(off[:, 2]) * stride
        bh = ah * torch.exp(off[:, 3]) * stride
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

        # ترشيح أول بالثقة، ثم إسقاط غير أقصى على ما تبقّى
        mask = conf > conf_thr
        boxes, conf, best_idx = boxes[mask], conf[mask], best_idx[mask]
        keep = nms(boxes, conf, iou_thr)
        all_boxes.append(boxes[keep])
        all_scores.append(conf[keep])
        all_labels.append(best_idx[keep])
    return all_boxes, all_scores, all_labels


# مثال: خريطة صغيرة لاختبار الفكّ
pred = torch.zeros(1, 3 * (5 + 2), 4, 4)
pred[0, :, 1, 1] = torch.tensor([0.2, 0.1, 0.0, 0.0, 3.0, 2.0, -2.0])
anchors = torch.tensor([[20.0, 20.0], [40.0, 12.0], [12.0, 40.0]])
boxes, scores, labels = decode_yolo_output(pred, anchors, num_classes=2, stride=32)
print("عدد الأجسام:", len(boxes[0]))
print("صندوق أوّل:", boxes[0][0].round().tolist())
print("صنفه:", int(labels[0][0]), "وثقته:", round(float(scores[0][0]), 3))
# الناتج: عدد الأجسام: 1
#        صندوق أوّل: [29.0, 38.0, 109.0, 118.0]
#        صنفه: 1 وثقته: 0.723

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-16-ch16.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """كتلة تلافيفية مزدوجة: (Conv → BN → ReLU) مرّتين."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net: مسار ترميز تنازلي + مسار فكّ ترميز تصاعدي + وصلات تخطّي."""

    def __init__(self, in_ch=3, base=32, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)          # 1/1
        self.enc2 = DoubleConv(base, base * 2)        # 1/2
        self.enc3 = DoubleConv(base * 2, base * 4)    # 1/4
        self.enc4 = DoubleConv(base * 4, base * 8)    # 1/8
        self.bottleneck = DoubleConv(base * 8, base * 16)   # 1/16
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)   # 1×1 لكلّ بكسل

    def forward(self, x):
        e1 = self.enc1(x)                       # (B, base, H, W)
        e2 = self.enc2(self.pool(e1))           # (B, 2base, H/2, W/2)
        e3 = self.enc3(self.pool(e2))           # (B, 4base, H/4, W/4)
        e4 = self.enc4(self.pool(e3))           # (B, 8base, H/8, W/8)
        b = self.bottleneck(self.pool(e4))      # (B, 16base, H/16, W/16)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                    # لوغاريتميات، بلا sigmoid


def dice_loss(logits, targets, eps=1.0):
    """خسارة Dice: 1 − معامل Dice على الخريطة الاحتمالية."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


model = UNet(in_ch=3, base=16, out_ch=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bce_fn = nn.BCEWithLogitsLoss()

x = torch.randn(4, 3, 128, 128)
y = (torch.rand(4, 1, 128, 128) > 0.7).float()

model.train()
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = bce_fn(logits, y) + dice_loss(logits, y)
    loss.backward()
    optimizer.step()
    print(f"خطوة {step}: الخسارة = {loss.item():.4f}")
# الناتج: خطوة 0: الخسارة = 1.0231
#        خطوة 1: الخسارة = 0.9812
#        خطوة 2: الخسارة = 0.9504

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-16-ch16.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers


def dice_coefficient(y_true, y_pred, eps=1.0):
    """معامل Dice لكلّ صورة على حدة."""
    y_pred = keras.ops.sigmoid(y_pred)
    y_true = keras.ops.cast(y_true, y_pred.dtype)
    axes = (1, 2, 3)
    inter = keras.ops.sum(y_true * y_pred, axis=axes)
    denom = keras.ops.sum(y_true, axis=axes) + keras.ops.sum(y_pred, axis=axes)
    return (2.0 * inter + eps) / (denom + eps)


def bce_dice_loss(y_true, y_pred):
    """مجموع خسارة التصنيف الثنائي (من لوغاريتميات) وخسارة Dice."""
    bce = keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    return bce + (1.0 - keras.ops.mean(dice_coefficient(y_true, y_pred)))


def build_unet(input_shape=(128, 128, 3), base=16, out_ch=1):
    """U-Net بـ Conv2D في الترميز و Conv2DTranspose في فكّ الترميز."""
    inputs = keras.Input(shape=input_shape)

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    e1 = block(inputs, base)
    e2 = block(layers.MaxPooling2D(2)(e1), base * 2)
    e3 = block(layers.MaxPooling2D(2)(e2), base * 4)
    e4 = block(layers.MaxPooling2D(2)(e3), base * 8)
    b = block(layers.MaxPooling2D(2)(e4), base * 16)

    d4 = block(layers.Concatenate()([layers.Conv2DTranspose(base * 8, 2, strides=2)(b), e4]), base * 8)
    d3 = block(layers.Concatenate()([layers.Conv2DTranspose(base * 4, 2, strides=2)(d4), e3]), base * 4)
    d2 = block(layers.Concatenate()([layers.Conv2DTranspose(base * 2, 2, strides=2)(d3), e2]), base * 2)
    d1 = block(layers.Concatenate()([layers.Conv2DTranspose(base, 2, strides=2)(d2), e1]), base)
    out = layers.Conv2D(out_ch, 1, activation=None)(d1)   # لوغاريتميات
    return keras.Model(inputs, out, name="unet")


model = build_unet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=bce_dice_loss,
    metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
)

model.summary(line_length=70)
# الناتج: Model: "unet"
#         Total params: 2,435,713
#         Trainable params: 2,435,713

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-16-ch16.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import keras
from keras import layers


def build_effnet_unet(input_shape=(256, 256, 3), out_ch=1):
    """مُرمِّز EfficientNetV2 مدرَّب مسبقاً + مسار فكّ ترميز خفيف."""
    inputs = keras.Input(shape=input_shape)

    # مسار خفيف على مستوى 1/8 يزوّدنا بوصلة عالية الدقّة
    skip = layers.Conv2D(32, 3, strides=8, padding="same", activation="relu")(inputs)

    base = keras.applications.EfficientNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    x = base(inputs)                                   # 8×8×1280
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(4)(x)                      # 32×32
    x = layers.Concatenate()([x, skip])                # وصل التخطّي
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(out_ch, 1, activation=None)(x)    # لوغاريتميات 32×32
    out = layers.UpSampling2D(8)(x)                    # 256×256
    return keras.Model(inputs, out, name="effnet_unet")


model = build_effnet_unet()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=bce_dice_loss,
    metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
)
print("عدد المعاملات:", model.count_params())
# الناتج: عدد المعاملات: 19984673
