#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج توليد الصور: Diffusion
Mastering Transformers and Large Language Models — Diffusion Models

Book slug : mastering-llms-transformers
Chapter   : ch-38-ch38-diffusion-models (lesson 38)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-38-ch38-diffusion-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج توليد الصور: Diffusion
Mastering Transformers and Large Language Models — Diffusion Models

Book slug : mastering-llms-transformers
Chapter   : ch-38-ch38-diffusion-models (lesson 38)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-38-ch38-diffusion-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج توليد الصور: Diffusion
Mastering Transformers and Large Language Models — Diffusion Models

Book slug : mastering-llms-transformers
Chapter   : ch-38-ch38-diffusion-models (lesson 38)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionScheduler:
    """جدولة الضوضاء (Noise Schedule) وعمليات forward/reverse الأساسية"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """الانتقال المباشر لخطوة t (عملية Forward بصيغة مغلقة)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_ac * x0 + sqrt_1m_ac * noise
        return x_t, noise

    @torch.no_grad()
    def sample_step(self, model, x_t, t, t_index, text_embeds=None):
        """خطوة واحدة من عملية الإزالة التراجعية (Reverse)"""
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1m_ac_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)

        predicted_noise = model(x_t, t, text_embeds)
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * predicted_noise / sqrt_1m_ac_t
        )
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_variance = beta_t
            return model_mean + torch.sqrt(posterior_variance) * noise

class SinusoidalTimeEmbedding(nn.Module):
    """ترميز الخطوة الزمنية t — نفس فكرة Positional Encoding من الفصل 11!"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SimpleConditionalUNet(nn.Module):
    """
    U-Net مبسّط موجَّه نصيًا (Text-Conditioned) — جوهر Stable Diffusion
    يتضمن: Cross-Attention مع نص (نفس آلية الفصل 10!) + ترميز زمني
    """
    def __init__(self, in_channels=4, base_channels=128, text_embed_dim=768, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU()
        )
        self.down1 = self._conv_block(in_channels, base_channels)
        self.down2 = self._conv_block(base_channels, base_channels * 2)

        # نفس MultiHeadAttention من الفصل 10، لكن هنا الاستعلام من الصورة والمفتاح/القيمة من النص
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_channels * 2, num_heads=8, batch_first=True,
            kdim=text_embed_dim, vdim=text_embed_dim
        )
        self.up1 = self._conv_block(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU()
        )

    def forward(self, x, t, text_embeds):
        time_emb = self.time_mlp(t)  # [B, time_dim]
        h = self.down1(x)
        h = self.down2(h)

        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C] كـ query
        attn_out, _ = self.cross_attn(query=h_flat, key=text_embeds, value=text_embeds)
        h = h_flat.permute(0, 2, 1).view(B, C, H, W) + attn_out.permute(0, 2, 1).view(B, C, H, W)

        h = self.up1(h)
        return self.out_conv(h)

def diffusion_training_step(model, scheduler, x0, text_embeds, optimizer):
    """خطوة تدريب واحدة كاملة"""
    batch_size = x0.shape[0]
    t = torch.randint(0, scheduler.timesteps, (batch_size,), device=x0.device)
    x_t, noise = scheduler.add_noise(x0, t)

    predicted_noise = model(x_t, t, text_embeds)
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def classifier_free_guidance_sample(model, x_t, t, text_embeds, null_embeds, guidance_scale=7.5):
    noise_cond = model(x_t, t, text_embeds)
    noise_uncond = model(x_t, t, null_embeds)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionScheduler:
    """جدولة الضوضاء (Noise Schedule) وعمليات forward/reverse الأساسية"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """الانتقال المباشر لخطوة t (عملية Forward بصيغة مغلقة)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_ac * x0 + sqrt_1m_ac * noise
        return x_t, noise

    @torch.no_grad()
    def sample_step(self, model, x_t, t, t_index, text_embeds=None):
        """خطوة واحدة من عملية الإزالة التراجعية (Reverse)"""
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1m_ac_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)

        predicted_noise = model(x_t, t, text_embeds)
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * predicted_noise / sqrt_1m_ac_t
        )
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_variance = beta_t
            return model_mean + torch.sqrt(posterior_variance) * noise

class SinusoidalTimeEmbedding(nn.Module):
    """ترميز الخطوة الزمنية t — نفس فكرة Positional Encoding من الفصل 11!"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SimpleConditionalUNet(nn.Module):
    """
    U-Net مبسّط موجَّه نصيًا (Text-Conditioned) — جوهر Stable Diffusion
    يتضمن: Cross-Attention مع نص (نفس آلية الفصل 10!) + ترميز زمني
    """
    def __init__(self, in_channels=4, base_channels=128, text_embed_dim=768, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU()
        )
        self.down1 = self._conv_block(in_channels, base_channels)
        self.down2 = self._conv_block(base_channels, base_channels * 2)

        # نفس MultiHeadAttention من الفصل 10، لكن هنا الاستعلام من الصورة والمفتاح/القيمة من النص
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_channels * 2, num_heads=8, batch_first=True,
            kdim=text_embed_dim, vdim=text_embed_dim
        )
        self.up1 = self._conv_block(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU()
        )

    def forward(self, x, t, text_embeds):
        time_emb = self.time_mlp(t)  # [B, time_dim]
        h = self.down1(x)
        h = self.down2(h)

        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C] كـ query
        attn_out, _ = self.cross_attn(query=h_flat, key=text_embeds, value=text_embeds)
        h = h_flat.permute(0, 2, 1).view(B, C, H, W) + attn_out.permute(0, 2, 1).view(B, C, H, W)

        h = self.up1(h)
        return self.out_conv(h)

def diffusion_training_step(model, scheduler, x0, text_embeds, optimizer):
    """خطوة تدريب واحدة كاملة"""
    batch_size = x0.shape[0]
    t = torch.randint(0, scheduler.timesteps, (batch_size,), device=x0.device)
    x_t, noise = scheduler.add_noise(x0, t)

    predicted_noise = model(x_t, t, text_embeds)
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def classifier_free_guidance_sample(model, x_t, t, text_embeds, null_embeds, guidance_scale=7.5):
    noise_cond = model(x_t, t, text_embeds)
    noise_uncond = model(x_t, t, null_embeds)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج توليد الصور: Diffusion
Mastering Transformers and Large Language Models — Diffusion Models

Book slug : mastering-llms-transformers
Chapter   : ch-38-ch38-diffusion-models (lesson 38)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionScheduler:
    """جدولة الضوضاء (Noise Schedule) وعمليات forward/reverse الأساسية"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """الانتقال المباشر لخطوة t (عملية Forward بصيغة مغلقة)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_ac * x0 + sqrt_1m_ac * noise
        return x_t, noise

    @torch.no_grad()
    def sample_step(self, model, x_t, t, t_index, text_embeds=None):
        """خطوة واحدة من عملية الإزالة التراجعية (Reverse)"""
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1m_ac_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)

        predicted_noise = model(x_t, t, text_embeds)
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * predicted_noise / sqrt_1m_ac_t
        )
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_variance = beta_t
            return model_mean + torch.sqrt(posterior_variance) * noise

class SinusoidalTimeEmbedding(nn.Module):
    """ترميز الخطوة الزمنية t — نفس فكرة Positional Encoding من الفصل 11!"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SimpleConditionalUNet(nn.Module):
    """
    U-Net مبسّط موجَّه نصيًا (Text-Conditioned) — جوهر Stable Diffusion
    يتضمن: Cross-Attention مع نص (نفس آلية الفصل 10!) + ترميز زمني
    """
    def __init__(self, in_channels=4, base_channels=128, text_embed_dim=768, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU()
        )
        self.down1 = self._conv_block(in_channels, base_channels)
        self.down2 = self._conv_block(base_channels, base_channels * 2)

        # نفس MultiHeadAttention من الفصل 10، لكن هنا الاستعلام من الصورة والمفتاح/القيمة من النص
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_channels * 2, num_heads=8, batch_first=True,
            kdim=text_embed_dim, vdim=text_embed_dim
        )
        self.up1 = self._conv_block(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU()
        )

    def forward(self, x, t, text_embeds):
        time_emb = self.time_mlp(t)  # [B, time_dim]
        h = self.down1(x)
        h = self.down2(h)

        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C] كـ query
        attn_out, _ = self.cross_attn(query=h_flat, key=text_embeds, value=text_embeds)
        h = h_flat.permute(0, 2, 1).view(B, C, H, W) + attn_out.permute(0, 2, 1).view(B, C, H, W)

        h = self.up1(h)
        return self.out_conv(h)

def diffusion_training_step(model, scheduler, x0, text_embeds, optimizer):
    """خطوة تدريب واحدة كاملة"""
    batch_size = x0.shape[0]
    t = torch.randint(0, scheduler.timesteps, (batch_size,), device=x0.device)
    x_t, noise = scheduler.add_noise(x0, t)

    predicted_noise = model(x_t, t, text_embeds)
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def classifier_free_guidance_sample(model, x_t, t, text_embeds, null_embeds, guidance_scale=7.5):
    noise_cond = model(x_t, t, text_embeds)
    noise_uncond = model(x_t, t, null_embeds)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch38-diffusion-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionScheduler:
    """جدولة الضوضاء (Noise Schedule) وعمليات forward/reverse الأساسية"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """الانتقال المباشر لخطوة t (عملية Forward بصيغة مغلقة)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_ac * x0 + sqrt_1m_ac * noise
        return x_t, noise

    @torch.no_grad()
    def sample_step(self, model, x_t, t, t_index, text_embeds=None):
        """خطوة واحدة من عملية الإزالة التراجعية (Reverse)"""
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_1m_ac_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)

        predicted_noise = model(x_t, t, text_embeds)
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * predicted_noise / sqrt_1m_ac_t
        )
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_variance = beta_t
            return model_mean + torch.sqrt(posterior_variance) * noise

class SinusoidalTimeEmbedding(nn.Module):
    """ترميز الخطوة الزمنية t — نفس فكرة Positional Encoding من الفصل 11!"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class SimpleConditionalUNet(nn.Module):
    """
    U-Net مبسّط موجَّه نصيًا (Text-Conditioned) — جوهر Stable Diffusion
    يتضمن: Cross-Attention مع نص (نفس آلية الفصل 10!) + ترميز زمني
    """
    def __init__(self, in_channels=4, base_channels=128, text_embed_dim=768, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU()
        )
        self.down1 = self._conv_block(in_channels, base_channels)
        self.down2 = self._conv_block(base_channels, base_channels * 2)

        # نفس MultiHeadAttention من الفصل 10، لكن هنا الاستعلام من الصورة والمفتاح/القيمة من النص
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_channels * 2, num_heads=8, batch_first=True,
            kdim=text_embed_dim, vdim=text_embed_dim
        )
        self.up1 = self._conv_block(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU()
        )

    def forward(self, x, t, text_embeds):
        time_emb = self.time_mlp(t)  # [B, time_dim]
        h = self.down1(x)
        h = self.down2(h)

        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C] كـ query
        attn_out, _ = self.cross_attn(query=h_flat, key=text_embeds, value=text_embeds)
        h = h_flat.permute(0, 2, 1).view(B, C, H, W) + attn_out.permute(0, 2, 1).view(B, C, H, W)

        h = self.up1(h)
        return self.out_conv(h)

def diffusion_training_step(model, scheduler, x0, text_embeds, optimizer):
    """خطوة تدريب واحدة كاملة"""
    batch_size = x0.shape[0]
    t = torch.randint(0, scheduler.timesteps, (batch_size,), device=x0.device)
    x_t, noise = scheduler.add_noise(x0, t)

    predicted_noise = model(x_t, t, text_embeds)
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def classifier_free_guidance_sample(model, x_t, t, text_embeds, null_embeds, guidance_scale=7.5):
    noise_cond = model(x_t, t, text_embeds)
    noise_uncond = model(x_t, t, null_embeds)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-38-ch38-diffusion-models.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
