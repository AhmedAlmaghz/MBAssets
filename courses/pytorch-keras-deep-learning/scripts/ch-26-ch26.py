#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الأداء والتسريع
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 26 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-26-ch26 (lesson 26)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-26-ch26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الأداء والتسريع
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 26 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-26-ch26 (lesson 26)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
# قياس زمن الخطوة بشكل صحيح (مع مزامنة!)
import time
import torch

def benchmark(model, loader, device="cuda", n_steps=300, warmup=30):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()

    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        lossf(model(xb), yb).backward()
        opt.step()
        if i == warmup:
            torch.cuda.synchronize()      # نقطة البداية المعيارية
            t0 = time.perf_counter()
        if i == warmup + n_steps:
            break
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps * 1000     # ملّي ثانية/خطوة

print(f"{benchmark(model, loader):.1f} ms/step")

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
import torch

scaler = torch.amp.GradScaler("cuda")          # PyTorch 2.14
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for xb, yb in loader:
    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = lossf(model(xb), yb)              # داخل autocast: fp16 تلقائياً

    scaler.scale(loss).backward()                # نضرب التدرّجات في scale
    scaler.unscale_(optimizer)                   # نعيد القسمة قبل القصّ
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                       # يتخطّى الخطوة إن ظهر inf/nan
    scaler.update()                              # تعدّل معامل التكبير

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
# قبل وبعد
model = model.cuda()
compiled = torch.compile(model)          # أو compiled = torch.compile(model, mode="max-autotune")

for xb, yb in loader:
    xb, yb = xb.cuda(), yb.cuda()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = lossf(compiled(xb), yb)
    loss.backward()
    opt.step()

# في Keras 3
model.compile(optimizer=keras.optimizers.AdamW(1e-3),
               loss="sparse_categorical_crossentropy",
               jit_compile=True)          # يعمل على نواتي torch و jax

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
# 1) TF32: دقة أقلّ في ضربات المصفوفات، سرعة أعلى بكثير على Ampere+
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2) channels_last: ذاكرة صور أسرع على Tensor Cores
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)

# 3) cudnn.benchmark: يبحث عن أسرع خوارزمية للأشكال الثابتة
torch.backends.cudnn.benchmark = True      # فعّله بعد ثبات شكل الإدخال

# 4) توسيع topical التمعّد:rops threads
torch.set_num_threads(min(8, torch.get_num_threads()))

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
# 1) التكميم الديناميكي (سريع التطبيق، على CPU)
import torch
from torch.ao.quantization import quantize_dynamic

quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print("size before:", os.path.getsize("model.pt"))
torch.save(quantized.state_dict(), "model_int8.pt")
print("size after:", os.path.getsize("model_int8.pt"))
# الناتج التقريبي: 100 MB ← 25 MB

# 2) التكميم الساكن (static) على GPU — يحتاج بيانات معايرة
from torch.ao.quantization import (get_default_qconfig_mapping,
                                    prepare, convert, calibrate)
qconfig = get_default_qconfig_mapping("x86")
prepared = prepare(model, qconfig)
calibration_loader = loader          # دفعة صغيرة Representative
with torch.no_grad():
    for i, (xb, _) in enumerate(calibration_loader):
        prepared(xb.cuda())
        if i >= 20:
            break
quantized_model = convert(prepared)

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, T=4):
    """الخسارة = مزيج بين softmax للتسميات الصعبة وsoft labels للمعلّم."""
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1 - alpha) * hard

# في حلقة التدريب:
with torch.no_grad():
    teacher_logits = teacher(xb)
student_logits = student(xb)
loss = distillation_loss(student_logits, teacher_logits, yb)
loss.backward()

# ────────────────────────────────────────────────────────────────────────
# [007] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        if i == 50:
            break

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))

# ────────────────────────────────────────────────────────────────────────
# [008] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")            # gloo على CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = Model().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])         # مزامنة التدرّجات

    # كل عملية ترى ثلث البيانات (DistributedSampler يتكفّل بذلك)
    loader = DataLoader(dataset, sampler=DistributedSampler(dataset))...
    dist.destroy_process_group()

# التشغيل:
# torchrun --nproc_per_node=4 train.py

# ────────────────────────────────────────────────────────────────────────
# [009] src/chapters/ch26.html
# ────────────────────────────────────────────────────────────────────────
import time
import torch


def measure(model, loader, device="cuda", steps=100):
    model.train()
    data_times, step_times = [], []
    it = iter(loader)

    for _ in range(steps):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y = batch
        x, y = x.to(device), y.to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer_step = model.zero_grad()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        data_times.append(t1 - t0)
        step_times.append(t2 - t1)

    d = sum(data_times) / steps
    s = sum(step_times) / steps
    print(f"data {d*1000:.1f} ms | step {s*1000:.1f} ms | util {(d/(d+s))*100:.0f}%")
    return d, s

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# قياس زمن الخطوة بشكل صحيح (مع مزامنة!)
import time
import torch

def benchmark(model, loader, device="cuda", n_steps=300, warmup=30):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()

    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        lossf(model(xb), yb).backward()
        opt.step()
        if i == warmup:
            torch.cuda.synchronize()      # نقطة البداية المعيارية
            t0 = time.perf_counter()
        if i == warmup + n_steps:
            break
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps * 1000     # ملّي ثانية/خطوة

print(f"{benchmark(model, loader):.1f} ms/step")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch

scaler = torch.amp.GradScaler("cuda")          # PyTorch 2.14
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for xb, yb in loader:
    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = lossf(model(xb), yb)              # داخل autocast: fp16 تلقائياً

    scaler.scale(loss).backward()                # نضرب التدرّجات في scale
    scaler.unscale_(optimizer)                   # نعيد القسمة قبل القصّ
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                       # يتخطّى الخطوة إن ظهر inf/nan
    scaler.update()                              # تعدّل معامل التكبير

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# قبل وبعد
model = model.cuda()
compiled = torch.compile(model)          # أو compiled = torch.compile(model, mode="max-autotune")

for xb, yb in loader:
    xb, yb = xb.cuda(), yb.cuda()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = lossf(compiled(xb), yb)
    loss.backward()
    opt.step()

# في Keras 3
model.compile(optimizer=keras.optimizers.AdamW(1e-3),
               loss="sparse_categorical_crossentropy",
               jit_compile=True)          # يعمل على نواتي torch و jax

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 1) TF32: دقة أقلّ في ضربات المصفوفات، سرعة أعلى بكثير على Ampere+
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2) channels_last: ذاكرة صور أسرع على Tensor Cores
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)

# 3) cudnn.benchmark: يبحث عن أسرع خوارزمية للأشكال الثابتة
torch.backends.cudnn.benchmark = True      # فعّله بعد ثبات شكل الإدخال

# 4) توسيع topical التمعّد:rops threads
torch.set_num_threads(min(8, torch.get_num_threads()))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 1) التكميم الديناميكي (سريع التطبيق، على CPU)
import torch
from torch.ao.quantization import quantize_dynamic

quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print("size before:", os.path.getsize("model.pt"))
torch.save(quantized.state_dict(), "model_int8.pt")
print("size after:", os.path.getsize("model_int8.pt"))
# الناتج التقريبي: 100 MB ← 25 MB

# 2) التكميم الساكن (static) على GPU — يحتاج بيانات معايرة
from torch.ao.quantization import (get_default_qconfig_mapping,
                                    prepare, convert, calibrate)
qconfig = get_default_qconfig_mapping("x86")
prepared = prepare(model, qconfig)
calibration_loader = loader          # دفعة صغيرة Representative
with torch.no_grad():
    for i, (xb, _) in enumerate(calibration_loader):
        prepared(xb.cuda())
        if i >= 20:
            break
quantized_model = convert(prepared)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, T=4):
    """الخسارة = مزيج بين softmax للتسميات الصعبة وsoft labels للمعلّم."""
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1 - alpha) * hard

# في حلقة التدريب:
with torch.no_grad():
    teacher_logits = teacher(xb)
student_logits = student(xb)
loss = distillation_loss(student_logits, teacher_logits, yb)
loss.backward()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        if i == 50:
            break

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")            # gloo على CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = Model().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])         # مزامنة التدرّجات

    # كل عملية ترى ثلث البيانات (DistributedSampler يتكفّل بذلك)
    loader = DataLoader(dataset, sampler=DistributedSampler(dataset))...
    dist.destroy_process_group()

# التشغيل:
# torchrun --nproc_per_node=4 train.py

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-26-ch26.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import time
import torch


def measure(model, loader, device="cuda", steps=100):
    model.train()
    data_times, step_times = [], []
    it = iter(loader)

    for _ in range(steps):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y = batch
        x, y = x.to(device), y.to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer_step = model.zero_grad()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        data_times.append(t1 - t0)
        step_times.append(t2 - t1)

    d = sum(data_times) / steps
    s = sum(step_times) / steps
    print(f"data {d*1000:.1f} ms | step {s*1000:.1f} ms | util {(d/(d+s))*100:.0f}%")
    return d, s
