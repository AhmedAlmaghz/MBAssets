#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الشبكات المتكرّرة وتسلسليات البيانات
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 14 — RNN

Book slug : pytorch-keras-deep-learning
Chapter   : ch-14-ch14 (lesson 14)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-14-ch14.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الشبكات المتكرّرة وتسلسليات البيانات
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 14 — RNN

Book slug : pytorch-keras-deep-learning
Chapter   : ch-14-ch14 (lesson 14)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch14.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleRNNCell(nn.Module):
    """خلية RNN واحدة: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        # Xavier للأوزان العابرة للمدخل، وorthogonal للأوزان العابرة للزمن:
        # التهيئة الجيدة هنا تختصر زمن التدريب وتقلّل الانفجار العددي.
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.b)

    def forward(self, x, h_prev):
        # x: (batch, input_size)   |   h_prev: (batch, hidden_size)
        h = torch.tanh(x @ self.W_xh + h_prev @ self.W_hh + self.b)
        return h


cell = SimpleRNNCell(input_size=3, hidden_size=4)
x_seq = torch.randn(2, 5, 3)          # (batch, time=5, features=3)
h = torch.zeros(2, cell.hidden_size)   # الحالة الابتدائية صفرية
for t in range(x_seq.size(1)):
    h = cell(x_seq[:, t, :], h)        # خطوة زمنية واحدة

print("h_T shape:", h.shape)            # الناتج: torch.Size([2, 4])
print("h_T:", h)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch14.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """خلية LSTM ببواباتٍ صريحة — تطبيقٌ مباشر للمعادلات في القسم 14.3."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # نجمع البوابات الأربع في مصفوفتين كبيرتين: ترتيب القطع هو i, f, g, o
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, x, h_prev, c_prev):
        # gates: (batch, 4*hidden_size) = x W_ih^T + h W_hh^T + b
        gates = x @ self.weight_ih.t() + h_prev @ self.weight_hh.t() + self.bias
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_raw)   # بوابة الإدخال: كم أدخل؟
        f = torch.sigmoid(f_raw)   # بوابة النسيان: كم أحتفظ؟
        g = torch.tanh(g_raw)      # مرشّح المحتوى: ما الذي سأدخله؟
        o = torch.sigmoid(o_raw)   # بوابة الإخراج: كم أخرج؟

        c = f * c_prev + i * g     # c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h = o * torch.tanh(c)     # h_t = o_t ⊙ tanh(c_t)
        return h, c


torch.manual_seed(0)
cell = LSTMCell(input_size=3, hidden_size=4)
x_seq = torch.randn(2, 5, 3)
h = torch.zeros(2, 4)
c = torch.zeros(2, 4)
for t in range(x_seq.size(1)):
    h, c = cell(x_seq[:, t, :], h, c)

print("h_T shape:", h.shape)   # الناتج: torch.Size([2, 4])
print("c_T shape:", c.shape)   # الناتج: torch.Size([2, 4])

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch14.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- BPTT مقطوع: نحدّ عدد الخطوات التي يمرّ عبرها التدرّج ----
def truncated_bptt(model, x_seq, h_prev, c_prev, chunk=20):
    """يمرّ بالانتشار الأمامي على كل السلسلة، ثم يقتطع التدرّج إلى chunk خطوة."""
    total_loss = 0.0
    h, c = h_prev, c_prev
    for t in range(x_seq.size(1)):
        h, c = model.cell(x_seq[:, t, :], h, c)   # انتشار أمامي كامل
        if (t + 1) % chunk == 0:
            # نقتطع الرسم البياني عند كل chunk: التدرّج لا يعود أبعد من chunk خطوة
            total_loss = total_loss + h.sum()
            total_loss.backward()
            model.zero_grad(set_to_none=True)
    return total_loss


class TinyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=8):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.cell.hidden_size)
        c = torch.zeros(x.size(0), self.cell.hidden_size)
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], h, c)
        return self.head(h)


model = TinyLSTM(input_size=1, hidden_size=8)
x_seq = torch.randn(4, 100, 1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(3):
    h0 = torch.zeros(4, 8)
    c0 = torch.zeros(4, 8)
    loss = truncated_bptt(model, x_seq, h0, c0, chunk=20)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # قصّ التدرّج
    opt.step()
    print(f"step {step} loss {loss.item():.4f}")
# الناتج: step 0 loss ... / step 1 loss ... / step 2 loss ...

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch14.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class WindowedSeries(Dataset):
    """يحوّل سلسلة زمنية إلى أزواج (نافذة، هدف) بتمرير نافذة بطول step."""

    def __init__(self, series, step=24, horizon=1):
        # series: (T, features) — لا نغيّر التدرّج هنا، فقط ننسّق الأبعاد
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.step = step
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.step - self.horizon + 1

    def __getitem__(self, i):
        # X: (step, features) و y: (horizon, features)
        x = self.series[i:i + self.step]
        y = self.series[i + self.step:i + self.step + self.horizon]
        return x, y


def standardise(train, rest):
    """تطبيع بأstatistics التدريب فقط — التسرّب هنا خطأ كلاسيكي."""
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (rest - mean) / std


# سلسلة وهمية: طلب يومي + طقس، لاختبار المسار كاملاً
raw = torch.stack([
    torch.linspace(0, 50, 400) + torch.randn(400) * 3,      # الطلب
    torch.sin(torch.arange(400) / 7.0) * 5 + 20,             # درجة الحرارة
], dim=1)

n_train = int(0.7 * len(raw))
n_val = int(0.15 * len(raw))
train_raw, val_raw, test_raw = raw[:n_train], raw[n_train:n_train + n_val], raw[n_train + n_val:]

train, _ = standardise(train_raw, train_raw)
val, _ = standardise(train_raw, val_raw)
test, _ = standardise(train_raw, test_raw)

train_ds = WindowedSeries(train, step=24, horizon=1)
val_ds = WindowedSeries(val, step=24, horizon=1)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

xb, yb = next(iter(train_dl))
print("batch X:", tuple(xb.shape), "batch y:", tuple(yb.shape))
# الناتج: batch X: (32, 24, 2) batch y: (32, 1, 2)

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch14.html
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(0)

# نفس منطق النافذة المنزلقة، لكن بأسلوب Keras
def make_windows(series, step=24):
    X = np.stack([series[i:i + step] for i in range(len(series) - step)])
    y = series[step:]
    return X.astype("float32"), y.astype("float32")


series = np.cumsum(np.random.randn(600, 2).astype("float32"), axis=0)
X, y = make_windows(series, step=24)
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

model = keras.Sequential([
    keras.Input(shape=(24, 2)),
    layers.LSTM(64, return_sequences=True),          # يُعيد سلسلة كاملة
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),         # يُعيد آخر حالة فقط
    layers.Dense(2),                                  # توقّع خطّي لكل خاصية
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=0,
)

print("best val MAE:", float(min(history.history["val_mean_absolute_error"])))

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch14.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DemandForecaster(nn.Module):
    """GRU بطبقتين + رأس predicts horizon خطوة لكل خاصية."""

    def __init__(self, n_features=2, hidden=64, horizon=7):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden,
                          num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, horizon * n_features),
        )

    def forward(self, x):
        # x: (batch, step, features)  →  المخرج: (batch, horizon, features)
        out, _ = self.gru(x)
        y = self.head(out[:, -1, :])          # آخر حالة مخفيّة
        return y.view(-1, self.horizon, x.size(-1))


model = DemandForecaster(n_features=2, hidden=64, horizon=7).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
loss_fn = nn.HuberLoss(delta=1.0)            # delta=1.0 على بيانات مطبَّعة

Xb = torch.randn(32, 28, 2).to(device)
Yb = torch.randn(32, 7, 2).to(device)

for epoch in range(20):
    model.train()
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred = model(Xb)
        loss = loss_fn(pred, Yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    sched.step()

    model.eval()
    with torch.no_grad():
        val = loss_fn(model(Xb), Yb).item()
    if epoch % 5 == 0:
        print(f"epoch {epoch:02d} train {loss.item():.4f} val {val:.4f}")
# الناتج: epoch 00 train ... val ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-14-ch14.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleRNNCell(nn.Module):
    """خلية RNN واحدة: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        # Xavier للأوزان العابرة للمدخل، وorthogonal للأوزان العابرة للزمن:
        # التهيئة الجيدة هنا تختصر زمن التدريب وتقلّل الانفجار العددي.
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.b)

    def forward(self, x, h_prev):
        # x: (batch, input_size)   |   h_prev: (batch, hidden_size)
        h = torch.tanh(x @ self.W_xh + h_prev @ self.W_hh + self.b)
        return h


cell = SimpleRNNCell(input_size=3, hidden_size=4)
x_seq = torch.randn(2, 5, 3)          # (batch, time=5, features=3)
h = torch.zeros(2, cell.hidden_size)   # الحالة الابتدائية صفرية
for t in range(x_seq.size(1)):
    h = cell(x_seq[:, t, :], h)        # خطوة زمنية واحدة

print("h_T shape:", h.shape)            # الناتج: torch.Size([2, 4])
print("h_T:", h)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-14-ch14.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """خلية LSTM ببواباتٍ صريحة — تطبيقٌ مباشر للمعادلات في القسم 14.3."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # نجمع البوابات الأربع في مصفوفتين كبيرتين: ترتيب القطع هو i, f, g, o
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, x, h_prev, c_prev):
        # gates: (batch, 4*hidden_size) = x W_ih^T + h W_hh^T + b
        gates = x @ self.weight_ih.t() + h_prev @ self.weight_hh.t() + self.bias
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_raw)   # بوابة الإدخال: كم أدخل؟
        f = torch.sigmoid(f_raw)   # بوابة النسيان: كم أحتفظ؟
        g = torch.tanh(g_raw)      # مرشّح المحتوى: ما الذي سأدخله؟
        o = torch.sigmoid(o_raw)   # بوابة الإخراج: كم أخرج؟

        c = f * c_prev + i * g     # c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h = o * torch.tanh(c)     # h_t = o_t ⊙ tanh(c_t)
        return h, c


torch.manual_seed(0)
cell = LSTMCell(input_size=3, hidden_size=4)
x_seq = torch.randn(2, 5, 3)
h = torch.zeros(2, 4)
c = torch.zeros(2, 4)
for t in range(x_seq.size(1)):
    h, c = cell(x_seq[:, t, :], h, c)

print("h_T shape:", h.shape)   # الناتج: torch.Size([2, 4])
print("c_T shape:", c.shape)   # الناتج: torch.Size([2, 4])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-14-ch14.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- BPTT مقطوع: نحدّ عدد الخطوات التي يمرّ عبرها التدرّج ----
def truncated_bptt(model, x_seq, h_prev, c_prev, chunk=20):
    """يمرّ بالانتشار الأمامي على كل السلسلة، ثم يقتطع التدرّج إلى chunk خطوة."""
    total_loss = 0.0
    h, c = h_prev, c_prev
    for t in range(x_seq.size(1)):
        h, c = model.cell(x_seq[:, t, :], h, c)   # انتشار أمامي كامل
        if (t + 1) % chunk == 0:
            # نقتطع الرسم البياني عند كل chunk: التدرّج لا يعود أبعد من chunk خطوة
            total_loss = total_loss + h.sum()
            total_loss.backward()
            model.zero_grad(set_to_none=True)
    return total_loss


class TinyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=8):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.cell.hidden_size)
        c = torch.zeros(x.size(0), self.cell.hidden_size)
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], h, c)
        return self.head(h)


model = TinyLSTM(input_size=1, hidden_size=8)
x_seq = torch.randn(4, 100, 1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(3):
    h0 = torch.zeros(4, 8)
    c0 = torch.zeros(4, 8)
    loss = truncated_bptt(model, x_seq, h0, c0, chunk=20)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # قصّ التدرّج
    opt.step()
    print(f"step {step} loss {loss.item():.4f}")
# الناتج: step 0 loss ... / step 1 loss ... / step 2 loss ...

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-14-ch14.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class WindowedSeries(Dataset):
    """يحوّل سلسلة زمنية إلى أزواج (نافذة، هدف) بتمرير نافذة بطول step."""

    def __init__(self, series, step=24, horizon=1):
        # series: (T, features) — لا نغيّر التدرّج هنا، فقط ننسّق الأبعاد
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.step = step
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.step - self.horizon + 1

    def __getitem__(self, i):
        # X: (step, features) و y: (horizon, features)
        x = self.series[i:i + self.step]
        y = self.series[i + self.step:i + self.step + self.horizon]
        return x, y


def standardise(train, rest):
    """تطبيع بأstatistics التدريب فقط — التسرّب هنا خطأ كلاسيكي."""
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (rest - mean) / std


# سلسلة وهمية: طلب يومي + طقس، لاختبار المسار كاملاً
raw = torch.stack([
    torch.linspace(0, 50, 400) + torch.randn(400) * 3,      # الطلب
    torch.sin(torch.arange(400) / 7.0) * 5 + 20,             # درجة الحرارة
], dim=1)

n_train = int(0.7 * len(raw))
n_val = int(0.15 * len(raw))
train_raw, val_raw, test_raw = raw[:n_train], raw[n_train:n_train + n_val], raw[n_train + n_val:]

train, _ = standardise(train_raw, train_raw)
val, _ = standardise(train_raw, val_raw)
test, _ = standardise(train_raw, test_raw)

train_ds = WindowedSeries(train, step=24, horizon=1)
val_ds = WindowedSeries(val, step=24, horizon=1)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

xb, yb = next(iter(train_dl))
print("batch X:", tuple(xb.shape), "batch y:", tuple(yb.shape))
# الناتج: batch X: (32, 24, 2) batch y: (32, 1, 2)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-14-ch14.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(0)

# نفس منطق النافذة المنزلقة، لكن بأسلوب Keras
def make_windows(series, step=24):
    X = np.stack([series[i:i + step] for i in range(len(series) - step)])
    y = series[step:]
    return X.astype("float32"), y.astype("float32")


series = np.cumsum(np.random.randn(600, 2).astype("float32"), axis=0)
X, y = make_windows(series, step=24)
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

model = keras.Sequential([
    keras.Input(shape=(24, 2)),
    layers.LSTM(64, return_sequences=True),          # يُعيد سلسلة كاملة
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),         # يُعيد آخر حالة فقط
    layers.Dense(2),                                  # توقّع خطّي لكل خاصية
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=0,
)

print("best val MAE:", float(min(history.history["val_mean_absolute_error"])))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-14-ch14.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DemandForecaster(nn.Module):
    """GRU بطبقتين + رأس predicts horizon خطوة لكل خاصية."""

    def __init__(self, n_features=2, hidden=64, horizon=7):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden,
                          num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, horizon * n_features),
        )

    def forward(self, x):
        # x: (batch, step, features)  →  المخرج: (batch, horizon, features)
        out, _ = self.gru(x)
        y = self.head(out[:, -1, :])          # آخر حالة مخفيّة
        return y.view(-1, self.horizon, x.size(-1))


model = DemandForecaster(n_features=2, hidden=64, horizon=7).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
loss_fn = nn.HuberLoss(delta=1.0)            # delta=1.0 على بيانات مطبَّعة

Xb = torch.randn(32, 28, 2).to(device)
Yb = torch.randn(32, 7, 2).to(device)

for epoch in range(20):
    model.train()
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred = model(Xb)
        loss = loss_fn(pred, Yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    sched.step()

    model.eval()
    with torch.no_grad():
        val = loss_fn(model(Xb), Yb).item()
    if epoch % 5 == 0:
        print(f"epoch {epoch:02d} train {loss.item():.4f} val {val:.4f}")
# الناتج: epoch 00 train ... val ...
