#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء
Large Language Model Engineering — Diffusion

Book slug : llm-engineering-69-chapters
Chapter   : ch-43-chapter-43 (lesson 43)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-43-chapter-43.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء
Large Language Model Engineering — Diffusion

Book slug : llm-engineering-69-chapters
Chapter   : ch-43-chapter-43 (lesson 43)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-43-chapter-43.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء
Large Language Model Engineering — Diffusion

Book slug : llm-engineering-69-chapters
Chapter   : ch-43-chapter-43 (lesson 43)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_43.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 43: الفصل 43: نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_diffusion() -> None:
    clean, noise, alpha = ([0.2, -0.4, 0.8], [0.5, 0.3, -0.2], 0.75)
    noisy = [math.sqrt(alpha) * x + math.sqrt(1 - alpha) * n for x, n in zip(clean, noise)]
    estimated_noise = noise
    restored = [(x - math.sqrt(1 - alpha) * e) / math.sqrt(alpha) for x, e in zip(noisy, estimated_noise)]
    print('clean:', clean)
    print('noisy sample:', [round(x, 3) for x in noisy])
    print('reconstructed:', [round(x, 3) for x in restored])

if __name__ == "__main__":
    print("Chapter 43 lab: Add and reverse noise in a toy diffusion process")
    run_diffusion()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-43.md
# ────────────────────────────────────────────────────────────────────────
"""
Denoising Diffusion Probabilistic Model (DDPM) من الصفر
تدريب على توليد صور أرقام MNIST غير الموجّه (Unconditional)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) جدول الضوضاء (Noise Schedule) — ثابت، لا يُدرَّب
# ---------------------------------------------------------------------
@dataclass
class DiffusionSchedule:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def forward_diffusion_sample(
    x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    تطبيق مباشر للصيغة المغلقة: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
    يسمح بالقفز مباشرة لأي خطوة t دون محاكاة كل الخطوات الوسيطة.
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise


# ---------------------------------------------------------------------
# 2) تضمين الخطوة الزمنية (Sinusoidal Timestep Embedding)
# ---------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


# ---------------------------------------------------------------------
# 3) كتلة U-Net الأساسية مع حقن تضمين الزمن
# ---------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_projection = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # حقن معلومة الخطوة الزمنية في كل قناة من خريطة السمات
        h = h + self.time_projection(time_embed)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual_connection(x)


class SimpleUNet(nn.Module):
    """U-Net مصغّرة لتوليد صور MNIST (28x28، قناة واحدة)."""

    def __init__(self, image_channels: int = 1, base_channels: int = 64, time_embed_dim: int = 64):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        time_dim = time_embed_dim * 4

        # مسار التقليص (Downsampling)
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 28->14

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 14->7

        # عنق الزجاجة (Bottleneck)
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        # مسار التوسيع (Upsampling) مع اتصالات تخطي (Skip Connections)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 7->14
        self.up1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 14->28
        self.up2 = ResidualBlock(base_channels + base_channels, base_channels, time_dim)

        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embedding(timesteps)

        x0 = self.init_conv(x)
        d1 = self.down1(x0, time_embed)
        d1_down = self.downsample1(d1)

        d2 = self.down2(d1_down, time_embed)
        d2_down = self.downsample2(d2)

        bottleneck = self.bottleneck(d2_down, time_embed)

        u1 = self.upsample1(bottleneck)
        u1 = self.up1(torch.cat([u1, d2], dim=1), time_embed)  # اتصال تخطي مع d2

        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), time_embed)  # اتصال تخطي مع d1

        return self.final_conv(u2)


# ---------------------------------------------------------------------
# 4) حلقة التدريب
# ---------------------------------------------------------------------
def train_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, train_loader: DataLoader, num_epochs: int = 20
):
    model.to(device)
    schedule.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # اختيار خطوة زمنية عشوائية مختلفة لكل عنصر في الدفعة
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)

            # تطبيق العملية الأمامية: توليد الصورة المشوّشة والضوضاء الحقيقية
            x_t, true_noise = forward_diffusion_sample(images, t, schedule)

            # تنبؤ الشبكة بالضوضاء
            predicted_noise = model(x_t, t)

            # الخسارة المبسَّطة: MSE بين الضوضاء الحقيقية والمتنبأ بها
            loss = F.mse_loss(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.5f}")


# ---------------------------------------------------------------------
# 5) الاستدلال: التوليد عبر عكس عملية الانتشار خطوة بخطوة
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, image_shape: tuple, num_samples: int = 16
) -> torch.Tensor:
    """
    ابدأ من ضوضاء خالصة، وطبّق التنبؤ بشكل تكراري لعكس الانتشار
    خطوة بخطوة من T إلى 0.
    """
    model.eval()
    x = torch.randn((num_samples, *image_shape), device=device)

    for t_step in reversed(range(schedule.num_timesteps)):
        t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha_t = schedule.alphas[t_step]
        alpha_cumprod_t = schedule.alphas_cumprod[t_step]
        beta_t = schedule.betas[t_step]

        # حساب المتوسط μ_θ وفق الصيغة التحليلية المشتقة من ε_θ
        coefficient = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coefficient * predicted_noise)

        if t_step > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta_t)
            x = mean + variance * noise
        else:
            x = mean  # في الخطوة الأخيرة، لا نضيف ضوضاء إضافية

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------
# 6) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # تطبيع إلى [-1, 1] — شائع في تدريب Diffusion
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    schedule = DiffusionSchedule(num_timesteps=1000)
    model = SimpleUNet(image_channels=1, base_channels=64)

    print(f"عدد البارامترات: {sum(p.numel() for p in model.parameters()):,}")

    train_ddpm(model, schedule, train_loader, num_epochs=20)

    # توليد صور جديدة من ضوضاء خالصة
    generated_images = sample_ddpm(model, schedule, image_shape=(1, 28, 28), num_samples=16)

    # حفظ الشبكة الناتجة كصورة واحدة مجمّعة
    grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
    vutils.save_image(grid, "ddpm_generated_samples.png")
    print("\n✅ تم حفظ الصور المُولَّدة في ddpm_generated_samples.png")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-43.md
# ────────────────────────────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# استبدال الجدولة الافتراضية بـ DDIM لاستدلال أسرع (20-50 خطوة بدل 1000)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

image = pipeline(
    prompt="قطة فضائية ترتدي بدلة رائد فضاء تحت ضوء القمر، لوحة زيتية",
    num_inference_steps=30,
    guidance_scale=7.5,  # مقياس Classifier-Free Guidance
).images[0]

image.save("generated_cat.png")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-43.md
# ────────────────────────────────────────────────────────────────────────
class ConditionalUNet(SimpleUNet):
    """
    أضف Embedding لفئة الرقم (0-9)، وادمجه مع تضمين الخطوة الزمنية
    قبل حقنه في كل ResidualBlock — بنفس طريقة حقن النص في Stable Diffusion
    لكن بشكل مبسّط (Class Embedding بدل Cross-Attention مع نص كامل).
    """
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.class_embedding = nn.Embedding(num_classes + 1, kwargs.get("time_embed_dim", 64) * 4)
        # +1 لرمز "فارغ" (Unconditional) يُستخدم في Classifier-Free Guidance

    def forward(self, x, timesteps, class_labels):
        # ادمج class_embedding مع time_embedding قبل تمريرهما لكتل ResidualBlock
        ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 43: الفصل 43: نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_diffusion() -> None:
    clean, noise, alpha = ([0.2, -0.4, 0.8], [0.5, 0.3, -0.2], 0.75)
    noisy = [math.sqrt(alpha) * x + math.sqrt(1 - alpha) * n for x, n in zip(clean, noise)]
    estimated_noise = noise
    restored = [(x - math.sqrt(1 - alpha) * e) / math.sqrt(alpha) for x, e in zip(noisy, estimated_noise)]
    print('clean:', clean)
    print('noisy sample:', [round(x, 3) for x in noisy])
    print('reconstructed:', [round(x, 3) for x in restored])

if __name__ == "__main__":
    print("Chapter 43 lab: Add and reverse noise in a toy diffusion process")
    run_diffusion()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
Denoising Diffusion Probabilistic Model (DDPM) من الصفر
تدريب على توليد صور أرقام MNIST غير الموجّه (Unconditional)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) جدول الضوضاء (Noise Schedule) — ثابت، لا يُدرَّب
# ---------------------------------------------------------------------
@dataclass
class DiffusionSchedule:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def forward_diffusion_sample(
    x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    تطبيق مباشر للصيغة المغلقة: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
    يسمح بالقفز مباشرة لأي خطوة t دون محاكاة كل الخطوات الوسيطة.
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise


# ---------------------------------------------------------------------
# 2) تضمين الخطوة الزمنية (Sinusoidal Timestep Embedding)
# ---------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


# ---------------------------------------------------------------------
# 3) كتلة U-Net الأساسية مع حقن تضمين الزمن
# ---------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_projection = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # حقن معلومة الخطوة الزمنية في كل قناة من خريطة السمات
        h = h + self.time_projection(time_embed)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual_connection(x)


class SimpleUNet(nn.Module):
    """U-Net مصغّرة لتوليد صور MNIST (28x28، قناة واحدة)."""

    def __init__(self, image_channels: int = 1, base_channels: int = 64, time_embed_dim: int = 64):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        time_dim = time_embed_dim * 4

        # مسار التقليص (Downsampling)
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 28->14

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 14->7

        # عنق الزجاجة (Bottleneck)
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        # مسار التوسيع (Upsampling) مع اتصالات تخطي (Skip Connections)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 7->14
        self.up1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 14->28
        self.up2 = ResidualBlock(base_channels + base_channels, base_channels, time_dim)

        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embedding(timesteps)

        x0 = self.init_conv(x)
        d1 = self.down1(x0, time_embed)
        d1_down = self.downsample1(d1)

        d2 = self.down2(d1_down, time_embed)
        d2_down = self.downsample2(d2)

        bottleneck = self.bottleneck(d2_down, time_embed)

        u1 = self.upsample1(bottleneck)
        u1 = self.up1(torch.cat([u1, d2], dim=1), time_embed)  # اتصال تخطي مع d2

        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), time_embed)  # اتصال تخطي مع d1

        return self.final_conv(u2)


# ---------------------------------------------------------------------
# 4) حلقة التدريب
# ---------------------------------------------------------------------
def train_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, train_loader: DataLoader, num_epochs: int = 20
):
    model.to(device)
    schedule.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # اختيار خطوة زمنية عشوائية مختلفة لكل عنصر في الدفعة
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)

            # تطبيق العملية الأمامية: توليد الصورة المشوّشة والضوضاء الحقيقية
            x_t, true_noise = forward_diffusion_sample(images, t, schedule)

            # تنبؤ الشبكة بالضوضاء
            predicted_noise = model(x_t, t)

            # الخسارة المبسَّطة: MSE بين الضوضاء الحقيقية والمتنبأ بها
            loss = F.mse_loss(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.5f}")


# ---------------------------------------------------------------------
# 5) الاستدلال: التوليد عبر عكس عملية الانتشار خطوة بخطوة
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, image_shape: tuple, num_samples: int = 16
) -> torch.Tensor:
    """
    ابدأ من ضوضاء خالصة، وطبّق التنبؤ بشكل تكراري لعكس الانتشار
    خطوة بخطوة من T إلى 0.
    """
    model.eval()
    x = torch.randn((num_samples, *image_shape), device=device)

    for t_step in reversed(range(schedule.num_timesteps)):
        t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha_t = schedule.alphas[t_step]
        alpha_cumprod_t = schedule.alphas_cumprod[t_step]
        beta_t = schedule.betas[t_step]

        # حساب المتوسط μ_θ وفق الصيغة التحليلية المشتقة من ε_θ
        coefficient = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coefficient * predicted_noise)

        if t_step > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta_t)
            x = mean + variance * noise
        else:
            x = mean  # في الخطوة الأخيرة، لا نضيف ضوضاء إضافية

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------
# 6) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # تطبيع إلى [-1, 1] — شائع في تدريب Diffusion
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    schedule = DiffusionSchedule(num_timesteps=1000)
    model = SimpleUNet(image_channels=1, base_channels=64)

    print(f"عدد البارامترات: {sum(p.numel() for p in model.parameters()):,}")

    train_ddpm(model, schedule, train_loader, num_epochs=20)

    # توليد صور جديدة من ضوضاء خالصة
    generated_images = sample_ddpm(model, schedule, image_shape=(1, 28, 28), num_samples=16)

    # حفظ الشبكة الناتجة كصورة واحدة مجمّعة
    grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
    vutils.save_image(grid, "ddpm_generated_samples.png")
    print("\n✅ تم حفظ الصور المُولَّدة في ddpm_generated_samples.png")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# استبدال الجدولة الافتراضية بـ DDIM لاستدلال أسرع (20-50 خطوة بدل 1000)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

image = pipeline(
    prompt="قطة فضائية ترتدي بدلة رائد فضاء تحت ضوء القمر، لوحة زيتية",
    num_inference_steps=30,
    guidance_scale=7.5,  # مقياس Classifier-Free Guidance
).images[0]

image.save("generated_cat.png")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ConditionalUNet(SimpleUNet):
    """
    أضف Embedding لفئة الرقم (0-9)، وادمجه مع تضمين الخطوة الزمنية
    قبل حقنه في كل ResidualBlock — بنفس طريقة حقن النص في Stable Diffusion
    لكن بشكل مبسّط (Class Embedding بدل Cross-Attention مع نص كامل).
    """
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.class_embedding = nn.Embedding(num_classes + 1, kwargs.get("time_embed_dim", 64) * 4)
        # +1 لرمز "فارغ" (Unconditional) يُستخدم في Classifier-Free Guidance

    def forward(self, x, timesteps, class_labels):
        # ادمج class_embedding مع time_embedding قبل تمريرهما لكتل ResidualBlock
        ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء
Large Language Model Engineering — Diffusion

Book slug : llm-engineering-69-chapters
Chapter   : ch-43-chapter-43 (lesson 43)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_43.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 43: الفصل 43: نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_diffusion() -> None:
    clean, noise, alpha = ([0.2, -0.4, 0.8], [0.5, 0.3, -0.2], 0.75)
    noisy = [math.sqrt(alpha) * x + math.sqrt(1 - alpha) * n for x, n in zip(clean, noise)]
    estimated_noise = noise
    restored = [(x - math.sqrt(1 - alpha) * e) / math.sqrt(alpha) for x, e in zip(noisy, estimated_noise)]
    print('clean:', clean)
    print('noisy sample:', [round(x, 3) for x in noisy])
    print('reconstructed:', [round(x, 3) for x in restored])

if __name__ == "__main__":
    print("Chapter 43 lab: Add and reverse noise in a toy diffusion process")
    run_diffusion()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-43.md
# ────────────────────────────────────────────────────────────────────────
"""
Denoising Diffusion Probabilistic Model (DDPM) من الصفر
تدريب على توليد صور أرقام MNIST غير الموجّه (Unconditional)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) جدول الضوضاء (Noise Schedule) — ثابت، لا يُدرَّب
# ---------------------------------------------------------------------
@dataclass
class DiffusionSchedule:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def forward_diffusion_sample(
    x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    تطبيق مباشر للصيغة المغلقة: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
    يسمح بالقفز مباشرة لأي خطوة t دون محاكاة كل الخطوات الوسيطة.
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise


# ---------------------------------------------------------------------
# 2) تضمين الخطوة الزمنية (Sinusoidal Timestep Embedding)
# ---------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


# ---------------------------------------------------------------------
# 3) كتلة U-Net الأساسية مع حقن تضمين الزمن
# ---------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_projection = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # حقن معلومة الخطوة الزمنية في كل قناة من خريطة السمات
        h = h + self.time_projection(time_embed)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual_connection(x)


class SimpleUNet(nn.Module):
    """U-Net مصغّرة لتوليد صور MNIST (28x28، قناة واحدة)."""

    def __init__(self, image_channels: int = 1, base_channels: int = 64, time_embed_dim: int = 64):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        time_dim = time_embed_dim * 4

        # مسار التقليص (Downsampling)
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 28->14

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 14->7

        # عنق الزجاجة (Bottleneck)
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        # مسار التوسيع (Upsampling) مع اتصالات تخطي (Skip Connections)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 7->14
        self.up1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 14->28
        self.up2 = ResidualBlock(base_channels + base_channels, base_channels, time_dim)

        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embedding(timesteps)

        x0 = self.init_conv(x)
        d1 = self.down1(x0, time_embed)
        d1_down = self.downsample1(d1)

        d2 = self.down2(d1_down, time_embed)
        d2_down = self.downsample2(d2)

        bottleneck = self.bottleneck(d2_down, time_embed)

        u1 = self.upsample1(bottleneck)
        u1 = self.up1(torch.cat([u1, d2], dim=1), time_embed)  # اتصال تخطي مع d2

        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), time_embed)  # اتصال تخطي مع d1

        return self.final_conv(u2)


# ---------------------------------------------------------------------
# 4) حلقة التدريب
# ---------------------------------------------------------------------
def train_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, train_loader: DataLoader, num_epochs: int = 20
):
    model.to(device)
    schedule.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # اختيار خطوة زمنية عشوائية مختلفة لكل عنصر في الدفعة
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)

            # تطبيق العملية الأمامية: توليد الصورة المشوّشة والضوضاء الحقيقية
            x_t, true_noise = forward_diffusion_sample(images, t, schedule)

            # تنبؤ الشبكة بالضوضاء
            predicted_noise = model(x_t, t)

            # الخسارة المبسَّطة: MSE بين الضوضاء الحقيقية والمتنبأ بها
            loss = F.mse_loss(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.5f}")


# ---------------------------------------------------------------------
# 5) الاستدلال: التوليد عبر عكس عملية الانتشار خطوة بخطوة
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, image_shape: tuple, num_samples: int = 16
) -> torch.Tensor:
    """
    ابدأ من ضوضاء خالصة، وطبّق التنبؤ بشكل تكراري لعكس الانتشار
    خطوة بخطوة من T إلى 0.
    """
    model.eval()
    x = torch.randn((num_samples, *image_shape), device=device)

    for t_step in reversed(range(schedule.num_timesteps)):
        t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha_t = schedule.alphas[t_step]
        alpha_cumprod_t = schedule.alphas_cumprod[t_step]
        beta_t = schedule.betas[t_step]

        # حساب المتوسط μ_θ وفق الصيغة التحليلية المشتقة من ε_θ
        coefficient = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coefficient * predicted_noise)

        if t_step > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta_t)
            x = mean + variance * noise
        else:
            x = mean  # في الخطوة الأخيرة، لا نضيف ضوضاء إضافية

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------
# 6) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # تطبيع إلى [-1, 1] — شائع في تدريب Diffusion
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    schedule = DiffusionSchedule(num_timesteps=1000)
    model = SimpleUNet(image_channels=1, base_channels=64)

    print(f"عدد البارامترات: {sum(p.numel() for p in model.parameters()):,}")

    train_ddpm(model, schedule, train_loader, num_epochs=20)

    # توليد صور جديدة من ضوضاء خالصة
    generated_images = sample_ddpm(model, schedule, image_shape=(1, 28, 28), num_samples=16)

    # حفظ الشبكة الناتجة كصورة واحدة مجمّعة
    grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
    vutils.save_image(grid, "ddpm_generated_samples.png")
    print("\n✅ تم حفظ الصور المُولَّدة في ddpm_generated_samples.png")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-43.md
# ────────────────────────────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# استبدال الجدولة الافتراضية بـ DDIM لاستدلال أسرع (20-50 خطوة بدل 1000)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

image = pipeline(
    prompt="قطة فضائية ترتدي بدلة رائد فضاء تحت ضوء القمر، لوحة زيتية",
    num_inference_steps=30,
    guidance_scale=7.5,  # مقياس Classifier-Free Guidance
).images[0]

image.save("generated_cat.png")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-43.md
# ────────────────────────────────────────────────────────────────────────
class ConditionalUNet(SimpleUNet):
    """
    أضف Embedding لفئة الرقم (0-9)، وادمجه مع تضمين الخطوة الزمنية
    قبل حقنه في كل ResidualBlock — بنفس طريقة حقن النص في Stable Diffusion
    لكن بشكل مبسّط (Class Embedding بدل Cross-Attention مع نص كامل).
    """
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.class_embedding = nn.Embedding(num_classes + 1, kwargs.get("time_embed_dim", 64) * 4)
        # +1 لرمز "فارغ" (Unconditional) يُستخدم في Classifier-Free Guidance

    def forward(self, x, timesteps, class_labels):
        # ادمج class_embedding مع time_embedding قبل تمريرهما لكتل ResidualBlock
        ...

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 43: الفصل 43: نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_diffusion() -> None:
    clean, noise, alpha = ([0.2, -0.4, 0.8], [0.5, 0.3, -0.2], 0.75)
    noisy = [math.sqrt(alpha) * x + math.sqrt(1 - alpha) * n for x, n in zip(clean, noise)]
    estimated_noise = noise
    restored = [(x - math.sqrt(1 - alpha) * e) / math.sqrt(alpha) for x, e in zip(noisy, estimated_noise)]
    print('clean:', clean)
    print('noisy sample:', [round(x, 3) for x in noisy])
    print('reconstructed:', [round(x, 3) for x in restored])

if __name__ == "__main__":
    print("Chapter 43 lab: Add and reverse noise in a toy diffusion process")
    run_diffusion()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
Denoising Diffusion Probabilistic Model (DDPM) من الصفر
تدريب على توليد صور أرقام MNIST غير الموجّه (Unconditional)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) جدول الضوضاء (Noise Schedule) — ثابت، لا يُدرَّب
# ---------------------------------------------------------------------
@dataclass
class DiffusionSchedule:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def forward_diffusion_sample(
    x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    تطبيق مباشر للصيغة المغلقة: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
    يسمح بالقفز مباشرة لأي خطوة t دون محاكاة كل الخطوات الوسيطة.
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise


# ---------------------------------------------------------------------
# 2) تضمين الخطوة الزمنية (Sinusoidal Timestep Embedding)
# ---------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


# ---------------------------------------------------------------------
# 3) كتلة U-Net الأساسية مع حقن تضمين الزمن
# ---------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_projection = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # حقن معلومة الخطوة الزمنية في كل قناة من خريطة السمات
        h = h + self.time_projection(time_embed)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual_connection(x)


class SimpleUNet(nn.Module):
    """U-Net مصغّرة لتوليد صور MNIST (28x28، قناة واحدة)."""

    def __init__(self, image_channels: int = 1, base_channels: int = 64, time_embed_dim: int = 64):
        super().__init__()
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        time_dim = time_embed_dim * 4

        # مسار التقليص (Downsampling)
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 28->14

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 14->7

        # عنق الزجاجة (Bottleneck)
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)

        # مسار التوسيع (Upsampling) مع اتصالات تخطي (Skip Connections)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 7->14
        self.up1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 14->28
        self.up2 = ResidualBlock(base_channels + base_channels, base_channels, time_dim)

        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embedding(timesteps)

        x0 = self.init_conv(x)
        d1 = self.down1(x0, time_embed)
        d1_down = self.downsample1(d1)

        d2 = self.down2(d1_down, time_embed)
        d2_down = self.downsample2(d2)

        bottleneck = self.bottleneck(d2_down, time_embed)

        u1 = self.upsample1(bottleneck)
        u1 = self.up1(torch.cat([u1, d2], dim=1), time_embed)  # اتصال تخطي مع d2

        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), time_embed)  # اتصال تخطي مع d1

        return self.final_conv(u2)


# ---------------------------------------------------------------------
# 4) حلقة التدريب
# ---------------------------------------------------------------------
def train_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, train_loader: DataLoader, num_epochs: int = 20
):
    model.to(device)
    schedule.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # اختيار خطوة زمنية عشوائية مختلفة لكل عنصر في الدفعة
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)

            # تطبيق العملية الأمامية: توليد الصورة المشوّشة والضوضاء الحقيقية
            x_t, true_noise = forward_diffusion_sample(images, t, schedule)

            # تنبؤ الشبكة بالضوضاء
            predicted_noise = model(x_t, t)

            # الخسارة المبسَّطة: MSE بين الضوضاء الحقيقية والمتنبأ بها
            loss = F.mse_loss(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.5f}")


# ---------------------------------------------------------------------
# 5) الاستدلال: التوليد عبر عكس عملية الانتشار خطوة بخطوة
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_ddpm(
    model: nn.Module, schedule: DiffusionSchedule, image_shape: tuple, num_samples: int = 16
) -> torch.Tensor:
    """
    ابدأ من ضوضاء خالصة، وطبّق التنبؤ بشكل تكراري لعكس الانتشار
    خطوة بخطوة من T إلى 0.
    """
    model.eval()
    x = torch.randn((num_samples, *image_shape), device=device)

    for t_step in reversed(range(schedule.num_timesteps)):
        t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha_t = schedule.alphas[t_step]
        alpha_cumprod_t = schedule.alphas_cumprod[t_step]
        beta_t = schedule.betas[t_step]

        # حساب المتوسط μ_θ وفق الصيغة التحليلية المشتقة من ε_θ
        coefficient = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coefficient * predicted_noise)

        if t_step > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta_t)
            x = mean + variance * noise
        else:
            x = mean  # في الخطوة الأخيرة، لا نضيف ضوضاء إضافية

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------
# 6) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # تطبيع إلى [-1, 1] — شائع في تدريب Diffusion
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    schedule = DiffusionSchedule(num_timesteps=1000)
    model = SimpleUNet(image_channels=1, base_channels=64)

    print(f"عدد البارامترات: {sum(p.numel() for p in model.parameters()):,}")

    train_ddpm(model, schedule, train_loader, num_epochs=20)

    # توليد صور جديدة من ضوضاء خالصة
    generated_images = sample_ddpm(model, schedule, image_shape=(1, 28, 28), num_samples=16)

    # حفظ الشبكة الناتجة كصورة واحدة مجمّعة
    grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
    vutils.save_image(grid, "ddpm_generated_samples.png")
    print("\n✅ تم حفظ الصور المُولَّدة في ddpm_generated_samples.png")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# استبدال الجدولة الافتراضية بـ DDIM لاستدلال أسرع (20-50 خطوة بدل 1000)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

image = pipeline(
    prompt="قطة فضائية ترتدي بدلة رائد فضاء تحت ضوء القمر، لوحة زيتية",
    num_inference_steps=30,
    guidance_scale=7.5,  # مقياس Classifier-Free Guidance
).images[0]

image.save("generated_cat.png")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-43-chapter-43.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ConditionalUNet(SimpleUNet):
    """
    أضف Embedding لفئة الرقم (0-9)، وادمجه مع تضمين الخطوة الزمنية
    قبل حقنه في كل ResidualBlock — بنفس طريقة حقن النص في Stable Diffusion
    لكن بشكل مبسّط (Class Embedding بدل Cross-Attention مع نص كامل).
    """
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.class_embedding = nn.Embedding(num_classes + 1, kwargs.get("time_embed_dim", 64) * 4)
        # +1 لرمز "فارغ" (Unconditional) يُستخدم في Classifier-Free Guidance

    def forward(self, x, timesteps, class_labels):
        # ادمج class_embedding مع time_embedding قبل تمريرهما لكتل ResidualBlock
        ...
