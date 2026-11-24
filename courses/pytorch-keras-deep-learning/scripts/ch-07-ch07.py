#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — تشريح nn.Module
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 07 — nn.Module

Book slug : pytorch-keras-deep-learning
Chapter   : ch-07-ch07 (lesson 7)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-07-ch07.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — تشريح nn.Module
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 07 — nn.Module

Book slug : pytorch-keras-deep-learning
Chapter   : ch-07-ch07 (lesson 7)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class MLP(nn.Module):
    """شبكة بسيطة من ثلاث طبقات كثيفة."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()                     # تهيئة صنف الأب: إلزامية دائماً
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()                  # طبقة بلا معاملات
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

model = MLP(in_features=32, hidden=64, out_features=10)
print(model)
# الناتج:
# MLP (
#   (fc1): Linear(in_features=32, out_features=64, bias=True)
#   (act): ReLU()
#   (fc2): Linear(in_features=64, out_features=10, bias=True)
# )

x = torch.randn(8, 32)
y = model(x)
print(y.shape)        # الناتج: torch.Size([8, 10])

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        # الخطأ: شدّة عادية — لن تتعلّم، ولن تظهر في state_dict
        self.bad = torch.randn(4, 4)
        # الصحيح: Parameter — تتعلّم، وتُحفظ، وتُنقل مع النموذج
        self.good = nn.Parameter(torch.randn(4, 4))

        # شرط: إنشاء المعاملات داخل __init__ لا داخل forward
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x):
        return x @ self.good * self.scale

m = SmallNet()
print([name for name, _ in m.named_parameters()])   # الناتج: ['good']
print([name for name, _ in m.named_buffers()])      # الناتج: ['scale']
print(m.bad.requires_grad, m.good.requires_grad)    # الناتج: False True

# تغيير حالة التدرّج
m.bad.requires_grad_(True)
print([n for n, _ in m.named_parameters()])         # الناتج: []  (ليست parameter أصلاً)
print(m.bad.grad)                                  # الناتج: None

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Scaled(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        self.counter += 1                     # المخزن يُعدَّل أثناء التمرير
        return (x - self.running_mean) / torch.sqrt(self.running_var)

bn = nn.BatchNorm2d(4)
print("running_mean في state_dict:", "bn.running_mean" in bn.state_dict())
print("weight (معامل) موجود:", "bn.weight" in bn.state_dict())

s = Scaled(3)
print(s.state_dict().keys())
# الناتج: dict_keys(['running_mean', 'running_var', 'counter'])

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

m = Tiny()

calls = []
h = m.register_forward_hook(lambda mod, inp, out: calls.append(out.shape))
out = m(torch.randn(3, 4))
h.remove()                       # إزالة الخطّاف: عادةً desirable بعد القياس
print(out.shape, calls)          # الناتج: torch.Size([3, 2]) [torch.Size([3, 2])]

# تمرير الخطّاف إلى children: كل طبقة داخلية تُراقب
inner = []
h2 = m.fc.register_forward_hook(lambda mod, inp, out: inner.append(out.shape))
m(torch.randn(5, 4))
h2.remove()
print(inner)                     # الناتج: [torch.Size([5, 2])]

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from torch.nn import init

torch.manual_seed(0)

# --- التهيئة الافتراضية ---
lin = nn.Linear(64, 64)
print(lin.weight.std().item())      # الناتج: 0.125  تقريباً (1/sqrt(64))

# --- تهيئة يدوية موحّدة ---
init.xavier_uniform_(lin.weight)
init.zeros_(lin.bias)                # الانحياز يبدأ من الصفر
print(lin.weight.std().item())      # الناتج: 0.125  تقريباً أيضاً

# --- تهيئة مناسبة لـ ReLU في الطبقات التلافيفية ---
conv = nn.Conv2d(3, 16, kernel_size=3)
init.kaiming_normal_(conv.weight, nonlinearity="relu")
init.constant_(conv.bias, 0.0)
print(conv.weight.std().item())

# --- تهيئة خطية: مهمة جداً للشبكات العميقة ---
deep = nn.Linear(128, 128)
init.xavier_normal_(deep.weight, gain=1.0)
init.zeros_(deep.bias)

# إعادة التهيئة إلى حالة الطبقات الأصلية
lin.reset_parameters()
print(lin.bias.abs().sum().item())   # الناتج: 0.0

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

torch.manual_seed(0)
model = Net()

# 1) احفظ حالة النموذج
torch.save(model.state_dict(), "net_state.pt")

# 2) استعدها بأمان: weights_only=True يمنع تنفيذ كود أثناء التحميل
state = torch.load("net_state.pt", weights_only=True)
print(list(state.keys()))            # الناتج: ['fc.weight', 'fc.bias']
print(state["fc.weight"].shape)      # الناتج: torch.Size([2, 10])

# 3) عِد بناء النموذج ثم حمّل
model2 = Net()
missing, unexpected = model2.load_state_dict(state)
print(missing, unexpected)           # الناتج: [] []

# 4) تحميل مرن: عند اختلاف البنية
result = model2.load_state_dict(state, strict=False)
print(result.missing_keys, result.unexpected_keys)

# 5) نسخة قابلة للتعديل دون المساس بالنموذج
safe_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
safe_state["fc.bias"] = torch.zeros(2)

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

model = nn.Sequential(
    nn.Linear(20, 16),
    nn.ReLU(),
    nn.BatchNorm1d(16),
    nn.Linear(16, 3),
)

# --- النقل إلى الجهاز ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(next(model.parameters()).device)     # الناتج: cuda (أو cpu)

# --- وضع التدريب ووضع التقييم ---
model.train()
print(model[2].training)                    # الناتج: True
model.eval()
print(model[2].training)                    # الناتج: False
model.train()                               # ارجع لوضع التدريب قبل المتابعة

# --- تطبيق دالة على كل الطبقات ---
def init_all(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        init.zeros_(module.bias)

model.apply(init_all)

# --- تصفّح الشجرة ---
print([name for name, _ in model.named_children()])   # الناتج: ['0', '1', '2', '3']
print(len(list(model.modules())))                     # الناتج: 9 (النموذج + 4 طبقات + 4 توابع)

# --- تجميد جزء من النموذج (للتدرّب الدقيق) ---
for param in model[0].parameters():
    param.requires_grad_(False)
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(trainable)
# الناتج: ['2.weight', '2.bias', '3.weight', '3.bias']

# --- متجه واحد لكل المعاملات: مفيد للقياس والمقارنة ---
flat = parameters_to_vector(model.parameters())
print(flat.shape)          # الناتج: torch.Size([387])

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import keras

# Keras 3 فوق نواة torch: نفس المفاهيم بصيغة مختلفة
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(16, activation="relu",
                       kernel_initializer=keras.initializers.HeNormal(),
                       bias_initializer="zeros"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(3),
])

model.build((None, 20))
model.summary()                     # يعرض عدد المعاملاتTrainable وNon-trainable

# الأوزان القابلة للتدرّب مقابل غير القابلة
print(len(model.trainable_variables))        # الناتج: 4
print(len(model.non_trainable_variables))    # الناتج: 2 (إحصاءات التطبيع)

# الحفظ والاستعادة
model.save_weights("keras_net.weights.h5")
same = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(16, activation="relu",
                       kernel_initializer=keras.initializers.HeNormal(),
                       bias_initializer="zeros"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(3),
])
same.build((None, 20))
same.load_weights("keras_net.weights.h5")
print("تمت الاستعادة")

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class ScaleShift(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std * self.gamma + self.beta

m = ScaleShift(4)
print(list(m.state_dict().keys()))
# الناتج: ['gamma', 'beta', 'mean', 'std']

# تحويل المخزن إلى غير دائم
m.register_buffer("std", torch.ones(4), persistent=False)
print(list(m.state_dict().keys()))
# الناتج: ['gamma', 'beta', 'mean']

print(m(torch.randn(2, 4)).shape)   # الناتج: torch.Size([2, 4])

# ────────────────────────────────────────────────────────────────────────
# [010] src/chapters/ch07.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Two(torch.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = Two()
torch.save(model.state_dict(), "two.pt")

state = torch.load("two.pt", weights_only=True)
print(list(state.keys()))     # الناتج: ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']

class One(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 6)

    def forward(self, x):
        return self.fc1(x)

other = One()
report = other.load_state_dict(state, strict=False)
print(report.missing_keys)     # الناتج: []
print(report.unexpected_keys)  # الناتج: ['fc2.weight', 'fc2.bias']

# إثبات المرجعية
sd = model.state_dict()
print(sd["fc1.weight"] is model.fc1.weight)          # الناتج: True
safe = {k: v.detach().clone() for k, v in sd.items()}
safe["fc1.weight"].add_(1.0)
print(model.fc1.weight[0, 0].item())                 # الناتج: 0.2500 (لم يتغيّر)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class MLP(nn.Module):
    """شبكة بسيطة من ثلاث طبقات كثيفة."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()                     # تهيئة صنف الأب: إلزامية دائماً
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()                  # طبقة بلا معاملات
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

model = MLP(in_features=32, hidden=64, out_features=10)
print(model)
# الناتج:
# MLP (
#   (fc1): Linear(in_features=32, out_features=64, bias=True)
#   (act): ReLU()
#   (fc2): Linear(in_features=64, out_features=10, bias=True)
# )

x = torch.randn(8, 32)
y = model(x)
print(y.shape)        # الناتج: torch.Size([8, 10])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        # الخطأ: شدّة عادية — لن تتعلّم، ولن تظهر في state_dict
        self.bad = torch.randn(4, 4)
        # الصحيح: Parameter — تتعلّم، وتُحفظ، وتُنقل مع النموذج
        self.good = nn.Parameter(torch.randn(4, 4))

        # شرط: إنشاء المعاملات داخل __init__ لا داخل forward
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x):
        return x @ self.good * self.scale

m = SmallNet()
print([name for name, _ in m.named_parameters()])   # الناتج: ['good']
print([name for name, _ in m.named_buffers()])      # الناتج: ['scale']
print(m.bad.requires_grad, m.good.requires_grad)    # الناتج: False True

# تغيير حالة التدرّج
m.bad.requires_grad_(True)
print([n for n, _ in m.named_parameters()])         # الناتج: []  (ليست parameter أصلاً)
print(m.bad.grad)                                  # الناتج: None

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Scaled(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        self.counter += 1                     # المخزن يُعدَّل أثناء التمرير
        return (x - self.running_mean) / torch.sqrt(self.running_var)

bn = nn.BatchNorm2d(4)
print("running_mean في state_dict:", "bn.running_mean" in bn.state_dict())
print("weight (معامل) موجود:", "bn.weight" in bn.state_dict())

s = Scaled(3)
print(s.state_dict().keys())
# الناتج: dict_keys(['running_mean', 'running_var', 'counter'])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

m = Tiny()

calls = []
h = m.register_forward_hook(lambda mod, inp, out: calls.append(out.shape))
out = m(torch.randn(3, 4))
h.remove()                       # إزالة الخطّاف: عادةً desirable بعد القياس
print(out.shape, calls)          # الناتج: torch.Size([3, 2]) [torch.Size([3, 2])]

# تمرير الخطّاف إلى children: كل طبقة داخلية تُراقب
inner = []
h2 = m.fc.register_forward_hook(lambda mod, inp, out: inner.append(out.shape))
m(torch.randn(5, 4))
h2.remove()
print(inner)                     # الناتج: [torch.Size([5, 2])]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from torch.nn import init

torch.manual_seed(0)

# --- التهيئة الافتراضية ---
lin = nn.Linear(64, 64)
print(lin.weight.std().item())      # الناتج: 0.125  تقريباً (1/sqrt(64))

# --- تهيئة يدوية موحّدة ---
init.xavier_uniform_(lin.weight)
init.zeros_(lin.bias)                # الانحياز يبدأ من الصفر
print(lin.weight.std().item())      # الناتج: 0.125  تقريباً أيضاً

# --- تهيئة مناسبة لـ ReLU في الطبقات التلافيفية ---
conv = nn.Conv2d(3, 16, kernel_size=3)
init.kaiming_normal_(conv.weight, nonlinearity="relu")
init.constant_(conv.bias, 0.0)
print(conv.weight.std().item())

# --- تهيئة خطية: مهمة جداً للشبكات العميقة ---
deep = nn.Linear(128, 128)
init.xavier_normal_(deep.weight, gain=1.0)
init.zeros_(deep.bias)

# إعادة التهيئة إلى حالة الطبقات الأصلية
lin.reset_parameters()
print(lin.bias.abs().sum().item())   # الناتج: 0.0

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

torch.manual_seed(0)
model = Net()

# 1) احفظ حالة النموذج
torch.save(model.state_dict(), "net_state.pt")

# 2) استعدها بأمان: weights_only=True يمنع تنفيذ كود أثناء التحميل
state = torch.load("net_state.pt", weights_only=True)
print(list(state.keys()))            # الناتج: ['fc.weight', 'fc.bias']
print(state["fc.weight"].shape)      # الناتج: torch.Size([2, 10])

# 3) عِد بناء النموذج ثم حمّل
model2 = Net()
missing, unexpected = model2.load_state_dict(state)
print(missing, unexpected)           # الناتج: [] []

# 4) تحميل مرن: عند اختلاف البنية
result = model2.load_state_dict(state, strict=False)
print(result.missing_keys, result.unexpected_keys)

# 5) نسخة قابلة للتعديل دون المساس بالنموذج
safe_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
safe_state["fc.bias"] = torch.zeros(2)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

model = nn.Sequential(
    nn.Linear(20, 16),
    nn.ReLU(),
    nn.BatchNorm1d(16),
    nn.Linear(16, 3),
)

# --- النقل إلى الجهاز ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(next(model.parameters()).device)     # الناتج: cuda (أو cpu)

# --- وضع التدريب ووضع التقييم ---
model.train()
print(model[2].training)                    # الناتج: True
model.eval()
print(model[2].training)                    # الناتج: False
model.train()                               # ارجع لوضع التدريب قبل المتابعة

# --- تطبيق دالة على كل الطبقات ---
def init_all(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        init.zeros_(module.bias)

model.apply(init_all)

# --- تصفّح الشجرة ---
print([name for name, _ in model.named_children()])   # الناتج: ['0', '1', '2', '3']
print(len(list(model.modules())))                     # الناتج: 9 (النموذج + 4 طبقات + 4 توابع)

# --- تجميد جزء من النموذج (للتدرّب الدقيق) ---
for param in model[0].parameters():
    param.requires_grad_(False)
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(trainable)
# الناتج: ['2.weight', '2.bias', '3.weight', '3.bias']

# --- متجه واحد لكل المعاملات: مفيد للقياس والمقارنة ---
flat = parameters_to_vector(model.parameters())
print(flat.shape)          # الناتج: torch.Size([387])

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import keras

# Keras 3 فوق نواة torch: نفس المفاهيم بصيغة مختلفة
model = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(16, activation="relu",
                       kernel_initializer=keras.initializers.HeNormal(),
                       bias_initializer="zeros"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(3),
])

model.build((None, 20))
model.summary()                     # يعرض عدد المعاملاتTrainable وNon-trainable

# الأوزان القابلة للتدرّب مقابل غير القابلة
print(len(model.trainable_variables))        # الناتج: 4
print(len(model.non_trainable_variables))    # الناتج: 2 (إحصاءات التطبيع)

# الحفظ والاستعادة
model.save_weights("keras_net.weights.h5")
same = keras.Sequential([
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(16, activation="relu",
                       kernel_initializer=keras.initializers.HeNormal(),
                       bias_initializer="zeros"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(3),
])
same.build((None, 20))
same.load_weights("keras_net.weights.h5")
print("تمت الاستعادة")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class ScaleShift(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("std", torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std * self.gamma + self.beta

m = ScaleShift(4)
print(list(m.state_dict().keys()))
# الناتج: ['gamma', 'beta', 'mean', 'std']

# تحويل المخزن إلى غير دائم
m.register_buffer("std", torch.ones(4), persistent=False)
print(list(m.state_dict().keys()))
# الناتج: ['gamma', 'beta', 'mean']

print(m(torch.randn(2, 4)).shape)   # الناتج: torch.Size([2, 4])

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-07-ch07.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class Two(torch.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = Two()
torch.save(model.state_dict(), "two.pt")

state = torch.load("two.pt", weights_only=True)
print(list(state.keys()))     # الناتج: ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']

class One(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 6)

    def forward(self, x):
        return self.fc1(x)

other = One()
report = other.load_state_dict(state, strict=False)
print(report.missing_keys)     # الناتج: []
print(report.unexpected_keys)  # الناتج: ['fc2.weight', 'fc2.bias']

# إثبات المرجعية
sd = model.state_dict()
print(sd["fc1.weight"] is model.fc1.weight)          # الناتج: True
safe = {k: v.detach().clone() for k, v in sd.items()}
safe["fc1.weight"].add_(1.0)
print(model.fc1.weight[0, 0].item())                 # الناتج: 0.2500 (لم يتغيّر)
