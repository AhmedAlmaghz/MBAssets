#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مقدمة إلى PEFT: التدريب الفعّال من حيث البارامترات
Mastering Transformers and Large Language Models — Peft

Book slug : mastering-llms-transformers
Chapter   : ch-30-ch30-peft (lesson 30)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-30-ch30-peft.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مقدمة إلى PEFT: التدريب الفعّال من حيث البارامترات
Mastering Transformers and Large Language Models — Peft

Book slug : mastering-llms-transformers
Chapter   : ch-30-ch30-peft (lesson 30)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-30-ch30-peft.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مقدمة إلى PEFT: التدريب الفعّال من حيث البارامترات
Mastering Transformers and Large Language Models — Peft

Book slug : mastering-llms-transformers
Chapter   : ch-30-ch30-peft (lesson 30)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class AdapterLayer(nn.Module):
    """
    طبقة Adapter كلاسيكية (Houlsby et al., 2019)
    تُدرَج بعد طبقات Attention وFFN في كل كتلة محوّل.
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64,
                 init_scale: float = 1e-3):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()

        # تهيئة قريبة من الصفر لضمان أن الـ Adapter
        # لا يُخرِّب سلوك النموذج المُدرَّب مسبقًا في بداية التدريب
        # (near-identity initialization)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return residual + h  # Residual Connection إلزامي

class TransformerBlockWithAdapters(nn.Module):
    """
    كتلة محوّل قياسية + Adapters مُدمَجة بعد Attention وبعد FFN.
    الأوزان الأصلية (attention, ffn, norms) تُجمَّد بالكامل.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 adapter_dim: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_adapter = AdapterLayer(d_model, adapter_dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_adapter = AdapterLayer(d_model, adapter_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # Attention + Adapter
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        attn_out = self.attn_adapter(attn_out)
        x = self.norm1(x + attn_out)

        # FFN + Adapter
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_adapter(ffn_out)
        x = self.norm2(x + ffn_out)
        return x

def freeze_base_model_keep_adapters(model: nn.Module):
    """
    تجميد كل المعاملات ما عدا الـ Adapters.
    هذه هي الخطوة الجوهرية في كل تقنيات PEFT: تحديد
    ما الذي يبقى requires_grad=True بدقة.
    """
    trainable_params, total_params = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "adapter" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    pct = 100 * trainable_params / total_params
    print(f"معاملات قابلة للتدريب: {trainable_params:,} "
          f"({pct:.3f}% من الإجمالي {total_params:,})")
    return trainable_params, total_params

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PromptTuningWrapper(nn.Module):
    """
    Prompt Tuning: نتعلّم n_tokens متجهات افتراضية (soft prompts)
    تُضاف إلى بداية embedding الإدخال. لا نغيّر معمارية النموذج إطلاقًا.
    """
    def __init__(self, base_model: nn.Module, embedding_dim: int,
                 n_virtual_tokens: int = 20, vocab_size_for_init: int = 30000):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False  # تجميد كامل للنموذج الأساسي

        # تهيئة ذكية: أخذ عينة من embeddings حقيقية بدلًا من عشوائية بحتة
        # (Vocabulary-based initialization) تُسرّع التقارب بشكل ملحوظ
        self.soft_prompt = nn.Parameter(
            torch.randn(n_virtual_tokens, embedding_dim) * 0.01
        )
        self.n_virtual_tokens = n_virtual_tokens

    def forward(self, input_embeds: torch.Tensor, attention_mask=None):
        batch_size = input_embeds.shape[0]
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_input = torch.cat([prompt, input_embeds], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.n_virtual_tokens,
                                      device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model(inputs_embeds=full_input,
                                attention_mask=attention_mask)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PrefixTuningBlock(nn.Module):
    """
    Prefix Tuning: لكل طبقة محوّل، نُضيف بادئة مُتعلَّمة
    إلى K و V في طبقة Attention الخاصة بها (وليس فقط للإدخال).
    يُنتَج عبر شبكة MLP صغيرة (Re-parameterization) لتحسين الاستقرار.
    """
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 n_prefix_tokens: int = 10, mlp_hidden: int = 512):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_prefix_tokens = n_prefix_tokens

        # متجه بذرة صغير يُمرَّر عبر MLP لتوليد كل البادئات
        # (أثبتت التجارب أن التدريب المباشر للبادئات غير مستقر)
        embed_dim = n_layers * n_heads * head_dim * 2  # ×2 لـ K و V
        self.prefix_seed = nn.Parameter(torch.randn(n_prefix_tokens, 512))
        self.reparam_mlp = nn.Sequential(
            nn.Linear(512, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def get_prefix_kv(self, batch_size: int, device):
        prefix = self.reparam_mlp(self.prefix_seed)  # (n_prefix, embed_dim)
        prefix = prefix.view(self.n_prefix_tokens, self.n_layers,
                              2, self.n_heads, self.head_dim)
        prefix = prefix.permute(1, 2, 3, 0, 4)  # (n_layers, 2, n_heads, n_prefix, head_dim)
        prefix = prefix.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)
        past_key_values = []
        for layer_idx in range(self.n_layers):
            k = prefix[layer_idx, :, 0]  # (batch, n_heads, n_prefix, head_dim)
            v = prefix[layer_idx, :, 1]
            past_key_values.append((k, v))
        return past_key_values

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def apply_bitfit(model: nn.Module):
    """
    BitFit (Ben Zaken et al., 2021): تدريب الـ bias terms فقط!
    مفاجئ لكنه فعّال لمهام بسيطة/متوسطة التعقيد.
    نسبة المعاملات القابلة للتدريب غالبًا < 0.1%.
    """
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    print(f"BitFit: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def compare_memory_footprint(d_model=768, n_layers=12, n_heads=12,
                               d_ff=3072, adapter_dim=64):
    # Full Fine-Tuning
    total_params = 110_000_000  # BERT-base تقريبًا
    full_ft_trainable = total_params

    # Adapter Tuning: adapter واحد بعد Attention وواحد بعد FFN لكل طبقة
    params_per_adapter = 2 * (d_model * adapter_dim + adapter_dim * d_model) \
                        + 2 * adapter_dim  # + biases تقريبية
    adapters_total = params_per_adapter * n_layers * 2  # ×2 attention+ffn

    print(f"Full FT معاملات قابلة للتدريب: {full_ft_trainable:,}")
    print(f"Adapter Tuning معاملات قابلة للتدريب: {adapters_total:,}")
    print(f"النسبة: {100 * adapters_total / full_ft_trainable:.3f}%")

    # تقدير الذاكرة (Adam: 2 إضافيين لكل معامل + gradient)
    bytes_per_param_fp32 = 4
    full_ft_memory_gb = full_ft_trainable * bytes_per_param_fp32 * 4 / 1e9  # weights+grad+2 adam states
    adapter_memory_gb = adapters_total * bytes_per_param_fp32 * 4 / 1e9

    print(f"\nذاكرة تدريب Full FT (تقريبية): {full_ft_memory_gb:.3f} GB")
    print(f"ذاكرة تدريب Adapter (تقريبية): {adapter_memory_gb:.5f} GB")

compare_memory_footprint()
# الناتج التقريبي:
# Full FT معاملات قابلة للتدريب: 110,000,000
# Adapter Tuning معاملات قابلة للتدريب: 2,433,024
# النسبة: 2.212%
# ذاكرة تدريب Full FT: 1.760 GB
# ذاكرة تدريب Adapter: 0.039 GB

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class AdapterLayer(nn.Module):
    """
    طبقة Adapter كلاسيكية (Houlsby et al., 2019)
    تُدرَج بعد طبقات Attention وFFN في كل كتلة محوّل.
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64,
                 init_scale: float = 1e-3):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()

        # تهيئة قريبة من الصفر لضمان أن الـ Adapter
        # لا يُخرِّب سلوك النموذج المُدرَّب مسبقًا في بداية التدريب
        # (near-identity initialization)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return residual + h  # Residual Connection إلزامي

class TransformerBlockWithAdapters(nn.Module):
    """
    كتلة محوّل قياسية + Adapters مُدمَجة بعد Attention وبعد FFN.
    الأوزان الأصلية (attention, ffn, norms) تُجمَّد بالكامل.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 adapter_dim: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_adapter = AdapterLayer(d_model, adapter_dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_adapter = AdapterLayer(d_model, adapter_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # Attention + Adapter
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        attn_out = self.attn_adapter(attn_out)
        x = self.norm1(x + attn_out)

        # FFN + Adapter
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_adapter(ffn_out)
        x = self.norm2(x + ffn_out)
        return x

def freeze_base_model_keep_adapters(model: nn.Module):
    """
    تجميد كل المعاملات ما عدا الـ Adapters.
    هذه هي الخطوة الجوهرية في كل تقنيات PEFT: تحديد
    ما الذي يبقى requires_grad=True بدقة.
    """
    trainable_params, total_params = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "adapter" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    pct = 100 * trainable_params / total_params
    print(f"معاملات قابلة للتدريب: {trainable_params:,} "
          f"({pct:.3f}% من الإجمالي {total_params:,})")
    return trainable_params, total_params

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PromptTuningWrapper(nn.Module):
    """
    Prompt Tuning: نتعلّم n_tokens متجهات افتراضية (soft prompts)
    تُضاف إلى بداية embedding الإدخال. لا نغيّر معمارية النموذج إطلاقًا.
    """
    def __init__(self, base_model: nn.Module, embedding_dim: int,
                 n_virtual_tokens: int = 20, vocab_size_for_init: int = 30000):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False  # تجميد كامل للنموذج الأساسي

        # تهيئة ذكية: أخذ عينة من embeddings حقيقية بدلًا من عشوائية بحتة
        # (Vocabulary-based initialization) تُسرّع التقارب بشكل ملحوظ
        self.soft_prompt = nn.Parameter(
            torch.randn(n_virtual_tokens, embedding_dim) * 0.01
        )
        self.n_virtual_tokens = n_virtual_tokens

    def forward(self, input_embeds: torch.Tensor, attention_mask=None):
        batch_size = input_embeds.shape[0]
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_input = torch.cat([prompt, input_embeds], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.n_virtual_tokens,
                                      device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model(inputs_embeds=full_input,
                                attention_mask=attention_mask)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PrefixTuningBlock(nn.Module):
    """
    Prefix Tuning: لكل طبقة محوّل، نُضيف بادئة مُتعلَّمة
    إلى K و V في طبقة Attention الخاصة بها (وليس فقط للإدخال).
    يُنتَج عبر شبكة MLP صغيرة (Re-parameterization) لتحسين الاستقرار.
    """
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 n_prefix_tokens: int = 10, mlp_hidden: int = 512):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_prefix_tokens = n_prefix_tokens

        # متجه بذرة صغير يُمرَّر عبر MLP لتوليد كل البادئات
        # (أثبتت التجارب أن التدريب المباشر للبادئات غير مستقر)
        embed_dim = n_layers * n_heads * head_dim * 2  # ×2 لـ K و V
        self.prefix_seed = nn.Parameter(torch.randn(n_prefix_tokens, 512))
        self.reparam_mlp = nn.Sequential(
            nn.Linear(512, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def get_prefix_kv(self, batch_size: int, device):
        prefix = self.reparam_mlp(self.prefix_seed)  # (n_prefix, embed_dim)
        prefix = prefix.view(self.n_prefix_tokens, self.n_layers,
                              2, self.n_heads, self.head_dim)
        prefix = prefix.permute(1, 2, 3, 0, 4)  # (n_layers, 2, n_heads, n_prefix, head_dim)
        prefix = prefix.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)
        past_key_values = []
        for layer_idx in range(self.n_layers):
            k = prefix[layer_idx, :, 0]  # (batch, n_heads, n_prefix, head_dim)
            v = prefix[layer_idx, :, 1]
            past_key_values.append((k, v))
        return past_key_values

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def apply_bitfit(model: nn.Module):
    """
    BitFit (Ben Zaken et al., 2021): تدريب الـ bias terms فقط!
    مفاجئ لكنه فعّال لمهام بسيطة/متوسطة التعقيد.
    نسبة المعاملات القابلة للتدريب غالبًا < 0.1%.
    """
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    print(f"BitFit: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def compare_memory_footprint(d_model=768, n_layers=12, n_heads=12,
                               d_ff=3072, adapter_dim=64):
    # Full Fine-Tuning
    total_params = 110_000_000  # BERT-base تقريبًا
    full_ft_trainable = total_params

    # Adapter Tuning: adapter واحد بعد Attention وواحد بعد FFN لكل طبقة
    params_per_adapter = 2 * (d_model * adapter_dim + adapter_dim * d_model) \
                        + 2 * adapter_dim  # + biases تقريبية
    adapters_total = params_per_adapter * n_layers * 2  # ×2 attention+ffn

    print(f"Full FT معاملات قابلة للتدريب: {full_ft_trainable:,}")
    print(f"Adapter Tuning معاملات قابلة للتدريب: {adapters_total:,}")
    print(f"النسبة: {100 * adapters_total / full_ft_trainable:.3f}%")

    # تقدير الذاكرة (Adam: 2 إضافيين لكل معامل + gradient)
    bytes_per_param_fp32 = 4
    full_ft_memory_gb = full_ft_trainable * bytes_per_param_fp32 * 4 / 1e9  # weights+grad+2 adam states
    adapter_memory_gb = adapters_total * bytes_per_param_fp32 * 4 / 1e9

    print(f"\nذاكرة تدريب Full FT (تقريبية): {full_ft_memory_gb:.3f} GB")
    print(f"ذاكرة تدريب Adapter (تقريبية): {adapter_memory_gb:.5f} GB")

compare_memory_footprint()
# الناتج التقريبي:
# Full FT معاملات قابلة للتدريب: 110,000,000
# Adapter Tuning معاملات قابلة للتدريب: 2,433,024
# النسبة: 2.212%
# ذاكرة تدريب Full FT: 1.760 GB
# ذاكرة تدريب Adapter: 0.039 GB

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مقدمة إلى PEFT: التدريب الفعّال من حيث البارامترات
Mastering Transformers and Large Language Models — Peft

Book slug : mastering-llms-transformers
Chapter   : ch-30-ch30-peft (lesson 30)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-1
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
# [002] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class AdapterLayer(nn.Module):
    """
    طبقة Adapter كلاسيكية (Houlsby et al., 2019)
    تُدرَج بعد طبقات Attention وFFN في كل كتلة محوّل.
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64,
                 init_scale: float = 1e-3):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()

        # تهيئة قريبة من الصفر لضمان أن الـ Adapter
        # لا يُخرِّب سلوك النموذج المُدرَّب مسبقًا في بداية التدريب
        # (near-identity initialization)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return residual + h  # Residual Connection إلزامي

class TransformerBlockWithAdapters(nn.Module):
    """
    كتلة محوّل قياسية + Adapters مُدمَجة بعد Attention وبعد FFN.
    الأوزان الأصلية (attention, ffn, norms) تُجمَّد بالكامل.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 adapter_dim: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_adapter = AdapterLayer(d_model, adapter_dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_adapter = AdapterLayer(d_model, adapter_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # Attention + Adapter
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        attn_out = self.attn_adapter(attn_out)
        x = self.norm1(x + attn_out)

        # FFN + Adapter
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_adapter(ffn_out)
        x = self.norm2(x + ffn_out)
        return x

def freeze_base_model_keep_adapters(model: nn.Module):
    """
    تجميد كل المعاملات ما عدا الـ Adapters.
    هذه هي الخطوة الجوهرية في كل تقنيات PEFT: تحديد
    ما الذي يبقى requires_grad=True بدقة.
    """
    trainable_params, total_params = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "adapter" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    pct = 100 * trainable_params / total_params
    print(f"معاملات قابلة للتدريب: {trainable_params:,} "
          f"({pct:.3f}% من الإجمالي {total_params:,})")
    return trainable_params, total_params

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PromptTuningWrapper(nn.Module):
    """
    Prompt Tuning: نتعلّم n_tokens متجهات افتراضية (soft prompts)
    تُضاف إلى بداية embedding الإدخال. لا نغيّر معمارية النموذج إطلاقًا.
    """
    def __init__(self, base_model: nn.Module, embedding_dim: int,
                 n_virtual_tokens: int = 20, vocab_size_for_init: int = 30000):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False  # تجميد كامل للنموذج الأساسي

        # تهيئة ذكية: أخذ عينة من embeddings حقيقية بدلًا من عشوائية بحتة
        # (Vocabulary-based initialization) تُسرّع التقارب بشكل ملحوظ
        self.soft_prompt = nn.Parameter(
            torch.randn(n_virtual_tokens, embedding_dim) * 0.01
        )
        self.n_virtual_tokens = n_virtual_tokens

    def forward(self, input_embeds: torch.Tensor, attention_mask=None):
        batch_size = input_embeds.shape[0]
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_input = torch.cat([prompt, input_embeds], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.n_virtual_tokens,
                                      device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model(inputs_embeds=full_input,
                                attention_mask=attention_mask)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PrefixTuningBlock(nn.Module):
    """
    Prefix Tuning: لكل طبقة محوّل، نُضيف بادئة مُتعلَّمة
    إلى K و V في طبقة Attention الخاصة بها (وليس فقط للإدخال).
    يُنتَج عبر شبكة MLP صغيرة (Re-parameterization) لتحسين الاستقرار.
    """
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 n_prefix_tokens: int = 10, mlp_hidden: int = 512):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_prefix_tokens = n_prefix_tokens

        # متجه بذرة صغير يُمرَّر عبر MLP لتوليد كل البادئات
        # (أثبتت التجارب أن التدريب المباشر للبادئات غير مستقر)
        embed_dim = n_layers * n_heads * head_dim * 2  # ×2 لـ K و V
        self.prefix_seed = nn.Parameter(torch.randn(n_prefix_tokens, 512))
        self.reparam_mlp = nn.Sequential(
            nn.Linear(512, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def get_prefix_kv(self, batch_size: int, device):
        prefix = self.reparam_mlp(self.prefix_seed)  # (n_prefix, embed_dim)
        prefix = prefix.view(self.n_prefix_tokens, self.n_layers,
                              2, self.n_heads, self.head_dim)
        prefix = prefix.permute(1, 2, 3, 0, 4)  # (n_layers, 2, n_heads, n_prefix, head_dim)
        prefix = prefix.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)
        past_key_values = []
        for layer_idx in range(self.n_layers):
            k = prefix[layer_idx, :, 0]  # (batch, n_heads, n_prefix, head_dim)
            v = prefix[layer_idx, :, 1]
            past_key_values.append((k, v))
        return past_key_values

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def apply_bitfit(model: nn.Module):
    """
    BitFit (Ben Zaken et al., 2021): تدريب الـ bias terms فقط!
    مفاجئ لكنه فعّال لمهام بسيطة/متوسطة التعقيد.
    نسبة المعاملات القابلة للتدريب غالبًا < 0.1%.
    """
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    print(f"BitFit: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def compare_memory_footprint(d_model=768, n_layers=12, n_heads=12,
                               d_ff=3072, adapter_dim=64):
    # Full Fine-Tuning
    total_params = 110_000_000  # BERT-base تقريبًا
    full_ft_trainable = total_params

    # Adapter Tuning: adapter واحد بعد Attention وواحد بعد FFN لكل طبقة
    params_per_adapter = 2 * (d_model * adapter_dim + adapter_dim * d_model) \
                        + 2 * adapter_dim  # + biases تقريبية
    adapters_total = params_per_adapter * n_layers * 2  # ×2 attention+ffn

    print(f"Full FT معاملات قابلة للتدريب: {full_ft_trainable:,}")
    print(f"Adapter Tuning معاملات قابلة للتدريب: {adapters_total:,}")
    print(f"النسبة: {100 * adapters_total / full_ft_trainable:.3f}%")

    # تقدير الذاكرة (Adam: 2 إضافيين لكل معامل + gradient)
    bytes_per_param_fp32 = 4
    full_ft_memory_gb = full_ft_trainable * bytes_per_param_fp32 * 4 / 1e9  # weights+grad+2 adam states
    adapter_memory_gb = adapters_total * bytes_per_param_fp32 * 4 / 1e9

    print(f"\nذاكرة تدريب Full FT (تقريبية): {full_ft_memory_gb:.3f} GB")
    print(f"ذاكرة تدريب Adapter (تقريبية): {adapter_memory_gb:.5f} GB")

compare_memory_footprint()
# الناتج التقريبي:
# Full FT معاملات قابلة للتدريب: 110,000,000
# Adapter Tuning معاملات قابلة للتدريب: 2,433,024
# النسبة: 2.212%
# ذاكرة تدريب Full FT: 1.760 GB
# ذاكرة تدريب Adapter: 0.039 GB

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-07-peft-alignment/ch30-peft.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

class AdapterLayer(nn.Module):
    """
    طبقة Adapter كلاسيكية (Houlsby et al., 2019)
    تُدرَج بعد طبقات Attention وFFN في كل كتلة محوّل.
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64,
                 init_scale: float = 1e-3):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()

        # تهيئة قريبة من الصفر لضمان أن الـ Adapter
        # لا يُخرِّب سلوك النموذج المُدرَّب مسبقًا في بداية التدريب
        # (near-identity initialization)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.up_proj(h)
        return residual + h  # Residual Connection إلزامي

class TransformerBlockWithAdapters(nn.Module):
    """
    كتلة محوّل قياسية + Adapters مُدمَجة بعد Attention وبعد FFN.
    الأوزان الأصلية (attention, ffn, norms) تُجمَّد بالكامل.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 adapter_dim: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_adapter = AdapterLayer(d_model, adapter_dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_adapter = AdapterLayer(d_model, adapter_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # Attention + Adapter
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        attn_out = self.attn_adapter(attn_out)
        x = self.norm1(x + attn_out)

        # FFN + Adapter
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_adapter(ffn_out)
        x = self.norm2(x + ffn_out)
        return x

def freeze_base_model_keep_adapters(model: nn.Module):
    """
    تجميد كل المعاملات ما عدا الـ Adapters.
    هذه هي الخطوة الجوهرية في كل تقنيات PEFT: تحديد
    ما الذي يبقى requires_grad=True بدقة.
    """
    trainable_params, total_params = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "adapter" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    pct = 100 * trainable_params / total_params
    print(f"معاملات قابلة للتدريب: {trainable_params:,} "
          f"({pct:.3f}% من الإجمالي {total_params:,})")
    return trainable_params, total_params

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PromptTuningWrapper(nn.Module):
    """
    Prompt Tuning: نتعلّم n_tokens متجهات افتراضية (soft prompts)
    تُضاف إلى بداية embedding الإدخال. لا نغيّر معمارية النموذج إطلاقًا.
    """
    def __init__(self, base_model: nn.Module, embedding_dim: int,
                 n_virtual_tokens: int = 20, vocab_size_for_init: int = 30000):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False  # تجميد كامل للنموذج الأساسي

        # تهيئة ذكية: أخذ عينة من embeddings حقيقية بدلًا من عشوائية بحتة
        # (Vocabulary-based initialization) تُسرّع التقارب بشكل ملحوظ
        self.soft_prompt = nn.Parameter(
            torch.randn(n_virtual_tokens, embedding_dim) * 0.01
        )
        self.n_virtual_tokens = n_virtual_tokens

    def forward(self, input_embeds: torch.Tensor, attention_mask=None):
        batch_size = input_embeds.shape[0]
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_input = torch.cat([prompt, input_embeds], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.n_virtual_tokens,
                                      device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model(inputs_embeds=full_input,
                                attention_mask=attention_mask)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class PrefixTuningBlock(nn.Module):
    """
    Prefix Tuning: لكل طبقة محوّل، نُضيف بادئة مُتعلَّمة
    إلى K و V في طبقة Attention الخاصة بها (وليس فقط للإدخال).
    يُنتَج عبر شبكة MLP صغيرة (Re-parameterization) لتحسين الاستقرار.
    """
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 n_prefix_tokens: int = 10, mlp_hidden: int = 512):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_prefix_tokens = n_prefix_tokens

        # متجه بذرة صغير يُمرَّر عبر MLP لتوليد كل البادئات
        # (أثبتت التجارب أن التدريب المباشر للبادئات غير مستقر)
        embed_dim = n_layers * n_heads * head_dim * 2  # ×2 لـ K و V
        self.prefix_seed = nn.Parameter(torch.randn(n_prefix_tokens, 512))
        self.reparam_mlp = nn.Sequential(
            nn.Linear(512, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def get_prefix_kv(self, batch_size: int, device):
        prefix = self.reparam_mlp(self.prefix_seed)  # (n_prefix, embed_dim)
        prefix = prefix.view(self.n_prefix_tokens, self.n_layers,
                              2, self.n_heads, self.head_dim)
        prefix = prefix.permute(1, 2, 3, 0, 4)  # (n_layers, 2, n_heads, n_prefix, head_dim)
        prefix = prefix.unsqueeze(1).expand(-1, batch_size, -1, -1, -1, -1)
        past_key_values = []
        for layer_idx in range(self.n_layers):
            k = prefix[layer_idx, :, 0]  # (batch, n_heads, n_prefix, head_dim)
            v = prefix[layer_idx, :, 1]
            past_key_values.append((k, v))
        return past_key_values

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def apply_bitfit(model: nn.Module):
    """
    BitFit (Ben Zaken et al., 2021): تدريب الـ bias terms فقط!
    مفاجئ لكنه فعّال لمهام بسيطة/متوسطة التعقيد.
    نسبة المعاملات القابلة للتدريب غالبًا < 0.1%.
    """
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    print(f"BitFit: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    return model

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def compare_memory_footprint(d_model=768, n_layers=12, n_heads=12,
                               d_ff=3072, adapter_dim=64):
    # Full Fine-Tuning
    total_params = 110_000_000  # BERT-base تقريبًا
    full_ft_trainable = total_params

    # Adapter Tuning: adapter واحد بعد Attention وواحد بعد FFN لكل طبقة
    params_per_adapter = 2 * (d_model * adapter_dim + adapter_dim * d_model) \
                        + 2 * adapter_dim  # + biases تقريبية
    adapters_total = params_per_adapter * n_layers * 2  # ×2 attention+ffn

    print(f"Full FT معاملات قابلة للتدريب: {full_ft_trainable:,}")
    print(f"Adapter Tuning معاملات قابلة للتدريب: {adapters_total:,}")
    print(f"النسبة: {100 * adapters_total / full_ft_trainable:.3f}%")

    # تقدير الذاكرة (Adam: 2 إضافيين لكل معامل + gradient)
    bytes_per_param_fp32 = 4
    full_ft_memory_gb = full_ft_trainable * bytes_per_param_fp32 * 4 / 1e9  # weights+grad+2 adam states
    adapter_memory_gb = adapters_total * bytes_per_param_fp32 * 4 / 1e9

    print(f"\nذاكرة تدريب Full FT (تقريبية): {full_ft_memory_gb:.3f} GB")
    print(f"ذاكرة تدريب Adapter (تقريبية): {adapter_memory_gb:.5f} GB")

compare_memory_footprint()
# الناتج التقريبي:
# Full FT معاملات قابلة للتدريب: 110,000,000
# Adapter Tuning معاملات قابلة للتدريب: 2,433,024
# النسبة: 2.212%
# ذاكرة تدريب Full FT: 1.760 GB
# ذاكرة تدريب Adapter: 0.039 GB

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-30-ch30-peft.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
