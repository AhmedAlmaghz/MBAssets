#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — هندسة البيانات: من المصدر إلى الدفعة
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 09 — Dataset

Book slug : pytorch-keras-deep-learning
Chapter   : ch-09-ch09 (lesson 9)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-09-ch09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — هندسة البيانات: من المصدر إلى الدفعة
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 09 — Dataset

Book slug : pytorch-keras-deep-learning
Chapter   : ch-09-ch09 (lesson 9)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset

class WineDataset(Dataset):
    """مجموعة بيانات من ملف CSV: الأعمدة features والأخيرة label."""

    def __init__(self, csv_path, transform=None):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.x = torch.tensor(df.drop(columns="label").values, dtype=torch.float32)
        self.y = torch.tensor(df["label"].values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

# تجزئة عشوائية بسيطة ثم تطبيع على التدريب فقط
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("wine.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

train_ds = WineDataset("train.csv")
test_ds = WineDataset("test.csv")
print(len(train_ds), train_ds[0][0].shape, train_ds[0][1])

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
from torchvision import transforms

# مجموعة تحويلات للتدريب (تتضمّن randomness مقصوداً)
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),                       # يحوّل PIL → tensor في [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# مجموعة التقييم: حتمية تماماً
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# مهمّ: normalizefur إن كان في مجموعة التدريب، فليست في التقييم

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,                 # ترتيب عشوائي في التدريب
    num_workers=4,                # عمّال متعدّدون (0 = المعالج الرئيسي)
    pin_memory=True,              # ذاكرة قابلة للتثبيت ⇒ نقل أسرع إلى GPU
    persistent_workers=True,      # العمّال يبقون حيّين بين الحقبات (يوفّر بدءاً)
    prefetch_factor=2,            # عدد الدفعات التي يجهّزها كل عامل مسبقاً
    drop_last=True,               # حجم ثابت للدفعة ⇒ BatchNorm مستقر
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False,                # لا خلط في التقييم!
    num_workers=2,
    pin_memory=True,
)

for xb, yb in train_loader:
    xb = xb.cuda(non_blocking=True)   # النقل غير متزامن (يعمل مع pin_memory)
    yb = yb.cuda(non_blocking=True)
    break

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    """للنصوص: نُرجع sequences متطابقة الطول + قناع الانتباه + قائمة الأطوال."""
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i,:len(t)] = torch.tensor(t, dtype=torch.long)
    mask = (torch.arange(max_len)[None,:] < lengths[:, None])   # 1 يعني كلمة حقيقية
    return padded, mask, labels, lengths

loader = DataLoader(text_ds, batch_size=32, collate_fn=collate_fn, num_workers=2)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import WeightedRandomSampler

# 1) وزن كل عيّنة بمقلوب تكرار صفّها
counts = torch.bincount(torch.tensor([int(y) for y in train_ds.y]))
class_weights = 1.0 / counts.float()
sample_weights = class_weights[torch.tensor([int(y) for y in train_ds.y])]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
balanced_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)

# 2) أو: وزن الخسارة (أسهل، ولا يغيّر توزيع البيانات)
pos_weight = (counts[0] / counts[1]).float()
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# 3) أو: الـloss نفسه (Focal Loss) — يعطي وزناً أكبر للعناصر الصعبة
#    مفيد حين تكون الأصناف النادرة هي المهمّة

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import IterableDataset, DataLoader

class StreamingImages(IterableDataset):
    """يتدفّق على ملفات من قرص بلا تخزين الكل في الذاكرة."""
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.files[worker.id::worker.num_workers] if worker else self.files
        for f in files:
            img, label = load_image(f)
            if self.transform:
                img = self.transform(img)
            yield img, label

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
import time
from torch.utils.data import DataLoader

def first_batch_time(loader, n=5):
    it = iter(loader)
    next(it)                      # تسخين
    t0 = time.perf_counter()
    for _ in range(n):
        next(it)
    return (time.perf_counter() - t0) / n

for nw in (0, 4):
    loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                        num_workers=nw, pin_memory=(nw > 0))
    print(f"num_workers={nw}: {first_batch_time(loader)*1000:.1f} ms/batch")

sizes = [xb.shape[0] for xb, _ in loader]
print("unique batch sizes:", set(sizes))   # مع drop_last=True: {64}

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader, Dataset

class TabularDataset(Dataset):
    """مجموعة بيانات جدولية: تقرأ ملفاً كاملاً ثم تخدم عيّناته."""

    def __init__(self, rows, features, labels):
        self.x = torch.tensor(rows[:, features], dtype=torch.float32)
        self.y = torch.tensor(rows[:, labels], dtype=torch.long)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loaders(train_ds, val_ds, batch_size=64, workers=4):
    """منشئ تحميل: تخلط التدريب فقط، والتقييم بترتيب ثابت."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch09.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import IterableDataset


class StreamingCsvDataset(IterableDataset):
    """يقرأ الملف صفّاً صفّاً: ذاكرة ثابتة مهما كبر الملف."""

    def __init__(self, path, features, labels, transform=None):
        self.path = path
        self.features = features
        self.labels = labels
        self.transform = transform

    def __iter__(self):
        import csv
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                x = torch.tensor(
                    [float(row[c]) for c in self.features], dtype=torch.float32
                )
                y = torch.tensor(int(row[self.labels[0]]), dtype=torch.long)
                if self.transform:
                    x = self.transform(x)
                yield x, y

    def __len__(self):
        with open(self.path, encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset

class WineDataset(Dataset):
    """مجموعة بيانات من ملف CSV: الأعمدة features والأخيرة label."""

    def __init__(self, csv_path, transform=None):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.x = torch.tensor(df.drop(columns="label").values, dtype=torch.float32)
        self.y = torch.tensor(df["label"].values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

# تجزئة عشوائية بسيطة ثم تطبيع على التدريب فقط
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("wine.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

train_ds = WineDataset("train.csv")
test_ds = WineDataset("test.csv")
print(len(train_ds), train_ds[0][0].shape, train_ds[0][1])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from torchvision import transforms

# مجموعة تحويلات للتدريب (تتضمّن randomness مقصوداً)
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),                       # يحوّل PIL → tensor في [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# مجموعة التقييم: حتمية تماماً
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# مهمّ: normalizefur إن كان في مجموعة التدريب، فليست في التقييم

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,                 # ترتيب عشوائي في التدريب
    num_workers=4,                # عمّال متعدّدون (0 = المعالج الرئيسي)
    pin_memory=True,              # ذاكرة قابلة للتثبيت ⇒ نقل أسرع إلى GPU
    persistent_workers=True,      # العمّال يبقون حيّين بين الحقبات (يوفّر بدءاً)
    prefetch_factor=2,            # عدد الدفعات التي يجهّزها كل عامل مسبقاً
    drop_last=True,               # حجم ثابت للدفعة ⇒ BatchNorm مستقر
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False,                # لا خلط في التقييم!
    num_workers=2,
    pin_memory=True,
)

for xb, yb in train_loader:
    xb = xb.cuda(non_blocking=True)   # النقل غير متزامن (يعمل مع pin_memory)
    yb = yb.cuda(non_blocking=True)
    break

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    """للنصوص: نُرجع sequences متطابقة الطول + قناع الانتباه + قائمة الأطوال."""
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded[i,:len(t)] = torch.tensor(t, dtype=torch.long)
    mask = (torch.arange(max_len)[None,:] < lengths[:, None])   # 1 يعني كلمة حقيقية
    return padded, mask, labels, lengths

loader = DataLoader(text_ds, batch_size=32, collate_fn=collate_fn, num_workers=2)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import WeightedRandomSampler

# 1) وزن كل عيّنة بمقلوب تكرار صفّها
counts = torch.bincount(torch.tensor([int(y) for y in train_ds.y]))
class_weights = 1.0 / counts.float()
sample_weights = class_weights[torch.tensor([int(y) for y in train_ds.y])]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
balanced_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)

# 2) أو: وزن الخسارة (أسهل، ولا يغيّر توزيع البيانات)
pos_weight = (counts[0] / counts[1]).float()
lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# 3) أو: الـloss نفسه (Focal Loss) — يعطي وزناً أكبر للعناصر الصعبة
#    مفيد حين تكون الأصناف النادرة هي المهمّة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from torch.utils.data import IterableDataset, DataLoader

class StreamingImages(IterableDataset):
    """يتدفّق على ملفات من قرص بلا تخزين الكل في الذاكرة."""
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.files[worker.id::worker.num_workers] if worker else self.files
        for f in files:
            img, label = load_image(f)
            if self.transform:
                img = self.transform(img)
            yield img, label

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import time
from torch.utils.data import DataLoader

def first_batch_time(loader, n=5):
    it = iter(loader)
    next(it)                      # تسخين
    t0 = time.perf_counter()
    for _ in range(n):
        next(it)
    return (time.perf_counter() - t0) / n

for nw in (0, 4):
    loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                        num_workers=nw, pin_memory=(nw > 0))
    print(f"num_workers={nw}: {first_batch_time(loader)*1000:.1f} ms/batch")

sizes = [xb.shape[0] for xb, _ in loader]
print("unique batch sizes:", set(sizes))   # مع drop_last=True: {64}

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader, Dataset

class TabularDataset(Dataset):
    """مجموعة بيانات جدولية: تقرأ ملفاً كاملاً ثم تخدم عيّناته."""

    def __init__(self, rows, features, labels):
        self.x = torch.tensor(rows[:, features], dtype=torch.float32)
        self.y = torch.tensor(rows[:, labels], dtype=torch.long)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loaders(train_ds, val_ds, batch_size=64, workers=4):
    """منشئ تحميل: تخلط التدريب فقط، والتقييم بترتيب ثابت."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-09-ch09.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import IterableDataset


class StreamingCsvDataset(IterableDataset):
    """يقرأ الملف صفّاً صفّاً: ذاكرة ثابتة مهما كبر الملف."""

    def __init__(self, path, features, labels, transform=None):
        self.path = path
        self.features = features
        self.labels = labels
        self.transform = transform

    def __iter__(self):
        import csv
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                x = torch.tensor(
                    [float(row[c]) for c in self.features], dtype=torch.float32
                )
                y = torch.tensor(int(row[self.labels[0]]), dtype=torch.long)
                if self.transform:
                    x = self.transform(x)
                yield x, y

    def __len__(self):
        with open(self.path, encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)
