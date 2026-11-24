#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خليط الخبراء: Mixture of Experts
Mastering Transformers and Large Language Models — Mixture Of Experts

Book slug : mastering-llms-transformers
Chapter   : ch-15-ch15-mixture-of-experts (lesson 15)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-15-ch15-mixture-of-experts.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خليط الخبراء: Mixture of Experts
Mastering Transformers and Large Language Models — Mixture Of Experts

Book slug : mastering-llms-transformers
Chapter   : ch-15-ch15-mixture-of-experts (lesson 15)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-15-ch15-mixture-of-experts.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خليط الخبراء: Mixture of Experts
Mastering Transformers and Large Language Models — Mixture Of Experts

Book slug : mastering-llms-transformers
Chapter   : ch-15-ch15-mixture-of-experts (lesson 15)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch15-mixture-of-experts.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch15-mixture-of-experts.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. خبير واحد (Expert) — شبكة FeedForward قياسية
# ============================================================
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 2. طبقة MoE كاملة مع Top-K Routing وLoad Balancing
# ============================================================
class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        العائد: (output, aux_loss)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)   # (batch*seq_len, d_model) لتبسيط التوجيه

        # 1. حساب درجات الـ Router
        router_logits = self.router(x_flat)              # (N_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # 2. اختيار أفضل K خبراء لكل رمز
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات على الخبراء المُختارين فقط
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 3. حساب مخرج كل رمز بتجميع مخرجات الخبراء المُختارين (موزونة)
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # أي الرموز اختارت هذا الخبير ضمن أفضل K؟
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (N_tokens,)

            if expert_mask.sum() == 0:
                continue  # لا رموز لهذا الخبير في هذه الدفعة

            # الوزن المقابل لهذا الخبير لكل رمز اختاره
            expert_weight_positions = (top_k_indices == expert_idx)
            expert_weights = (top_k_probs * expert_weight_positions).sum(dim=-1)  # (N_tokens,)

            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            output[expert_mask] += expert_weights[expert_mask].unsqueeze(-1) * expert_output

        output = output.view(batch_size, seq_len, d_model)

        # 4. حساب Load Balancing Loss
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(self, router_probs, top_k_indices):
        """
        L_balance = N * Σ f_i * P_i
        f_i: نسبة الرموز الفعلية المُوجَّهة للخبير i
        P_i: متوسط احتمال الـ Router للخبير i عبر كل الرموز
        """
        num_tokens = router_probs.shape[0]

        # P_i: متوسط احتمال الـ router لكل خبير (عبر كل الرموز، وليس فقط المُختارة)
        P = router_probs.mean(dim=0)   # (num_experts,)

        # f_i: نسبة الرموز التي اختارت هذا الخبير ضمن Top-K فعليًا
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        # expert_mask: (num_tokens, top_k, num_experts) → نجمع على top_k
        f = expert_mask.sum(dim=1).mean(dim=0) / self.top_k  # (num_experts,)

        aux_loss = self.num_experts * torch.sum(f * P)
        return aux_loss

# ============================================================
# 3. طبقة Transformer كاملة تستخدم MoE بدلاً من FFN القياسية
# ============================================================
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)  # من الفصل 10
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)

    def forward(self, x, mask=None):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, mask)
        x = x + attn_out

        normed = self.ln2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

# ============================================================
# 4. مفهوم DeepSeek-V3 المتقدم: خبراء مشتركون + خبراء متخصصون
# ============================================================
class DeepSeekStyleMoELayer(nn.Module):
    """
    تطبيق مبسّط لفكرة DeepSeek-V3: خبراء مشتركون (shared experts)
    مُفعَّلون دائمًا لكل رمز (يلتقطون المعرفة العامة المشتركة)،
    بالإضافة لخبراء متخصصين (routed experts) يُختارون ديناميكيًا.
    """
    def __init__(self, d_model, d_ff, num_shared_experts=2,
                 num_routed_experts=64, top_k_routed=6, dropout=0.1):
        super().__init__()

        # خبراء مشتركون: نشطون دائمًا لكل رمز، بدون توجيه
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_shared_experts)
        ])

        # خبراء متخصصون: يُختار منهم عبر Top-K Routing
        self.routed_moe = MoELayer(
            d_model, d_ff, num_experts=num_routed_experts,
            top_k=top_k_routed, dropout=dropout
        )

    def forward(self, x):
        # مساهمة الخبراء المشتركين (دائمًا نشطة)
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # مساهمة الخبراء المتخصصين (عبر التوجيه الديناميكي)
        routed_output, aux_loss = self.routed_moe(x)

        return shared_output + routed_output, aux_loss

def compute_moe_efficiency_stats(num_experts, top_k, expert_params, router_params):
    """حساب نسبة السعة الكلية إلى التكلفة الفعلية في MoE."""
    total_params = num_experts * expert_params + router_params
    active_params = top_k * expert_params + router_params

    efficiency_ratio = total_params / active_params

    print(f"إجمالي المعاملات (السعة): {total_params:,}")
    print(f"المعاملات النشطة لكل رمز (التكلفة): {active_params:,}")
    print(f"نسبة السعة إلى التكلفة: {efficiency_ratio:.2f}x")

    return total_params, active_params, efficiency_ratio

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch15-mixture-of-experts.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. خبير واحد (Expert) — شبكة FeedForward قياسية
# ============================================================
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 2. طبقة MoE كاملة مع Top-K Routing وLoad Balancing
# ============================================================
class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        العائد: (output, aux_loss)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)   # (batch*seq_len, d_model) لتبسيط التوجيه

        # 1. حساب درجات الـ Router
        router_logits = self.router(x_flat)              # (N_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # 2. اختيار أفضل K خبراء لكل رمز
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات على الخبراء المُختارين فقط
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 3. حساب مخرج كل رمز بتجميع مخرجات الخبراء المُختارين (موزونة)
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # أي الرموز اختارت هذا الخبير ضمن أفضل K؟
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (N_tokens,)

            if expert_mask.sum() == 0:
                continue  # لا رموز لهذا الخبير في هذه الدفعة

            # الوزن المقابل لهذا الخبير لكل رمز اختاره
            expert_weight_positions = (top_k_indices == expert_idx)
            expert_weights = (top_k_probs * expert_weight_positions).sum(dim=-1)  # (N_tokens,)

            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            output[expert_mask] += expert_weights[expert_mask].unsqueeze(-1) * expert_output

        output = output.view(batch_size, seq_len, d_model)

        # 4. حساب Load Balancing Loss
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(self, router_probs, top_k_indices):
        """
        L_balance = N * Σ f_i * P_i
        f_i: نسبة الرموز الفعلية المُوجَّهة للخبير i
        P_i: متوسط احتمال الـ Router للخبير i عبر كل الرموز
        """
        num_tokens = router_probs.shape[0]

        # P_i: متوسط احتمال الـ router لكل خبير (عبر كل الرموز، وليس فقط المُختارة)
        P = router_probs.mean(dim=0)   # (num_experts,)

        # f_i: نسبة الرموز التي اختارت هذا الخبير ضمن Top-K فعليًا
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        # expert_mask: (num_tokens, top_k, num_experts) → نجمع على top_k
        f = expert_mask.sum(dim=1).mean(dim=0) / self.top_k  # (num_experts,)

        aux_loss = self.num_experts * torch.sum(f * P)
        return aux_loss

# ============================================================
# 3. طبقة Transformer كاملة تستخدم MoE بدلاً من FFN القياسية
# ============================================================
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)  # من الفصل 10
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)

    def forward(self, x, mask=None):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, mask)
        x = x + attn_out

        normed = self.ln2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

# ============================================================
# 4. مفهوم DeepSeek-V3 المتقدم: خبراء مشتركون + خبراء متخصصون
# ============================================================
class DeepSeekStyleMoELayer(nn.Module):
    """
    تطبيق مبسّط لفكرة DeepSeek-V3: خبراء مشتركون (shared experts)
    مُفعَّلون دائمًا لكل رمز (يلتقطون المعرفة العامة المشتركة)،
    بالإضافة لخبراء متخصصين (routed experts) يُختارون ديناميكيًا.
    """
    def __init__(self, d_model, d_ff, num_shared_experts=2,
                 num_routed_experts=64, top_k_routed=6, dropout=0.1):
        super().__init__()

        # خبراء مشتركون: نشطون دائمًا لكل رمز، بدون توجيه
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_shared_experts)
        ])

        # خبراء متخصصون: يُختار منهم عبر Top-K Routing
        self.routed_moe = MoELayer(
            d_model, d_ff, num_experts=num_routed_experts,
            top_k=top_k_routed, dropout=dropout
        )

    def forward(self, x):
        # مساهمة الخبراء المشتركين (دائمًا نشطة)
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # مساهمة الخبراء المتخصصين (عبر التوجيه الديناميكي)
        routed_output, aux_loss = self.routed_moe(x)

        return shared_output + routed_output, aux_loss

def compute_moe_efficiency_stats(num_experts, top_k, expert_params, router_params):
    """حساب نسبة السعة الكلية إلى التكلفة الفعلية في MoE."""
    total_params = num_experts * expert_params + router_params
    active_params = top_k * expert_params + router_params

    efficiency_ratio = total_params / active_params

    print(f"إجمالي المعاملات (السعة): {total_params:,}")
    print(f"المعاملات النشطة لكل رمز (التكلفة): {active_params:,}")
    print(f"نسبة السعة إلى التكلفة: {efficiency_ratio:.2f}x")

    return total_params, active_params, efficiency_ratio

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — خليط الخبراء: Mixture of Experts
Mastering Transformers and Large Language Models — Mixture Of Experts

Book slug : mastering-llms-transformers
Chapter   : ch-15-ch15-mixture-of-experts (lesson 15)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch15-mixture-of-experts.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch15-mixture-of-experts.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. خبير واحد (Expert) — شبكة FeedForward قياسية
# ============================================================
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 2. طبقة MoE كاملة مع Top-K Routing وLoad Balancing
# ============================================================
class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        العائد: (output, aux_loss)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)   # (batch*seq_len, d_model) لتبسيط التوجيه

        # 1. حساب درجات الـ Router
        router_logits = self.router(x_flat)              # (N_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # 2. اختيار أفضل K خبراء لكل رمز
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات على الخبراء المُختارين فقط
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 3. حساب مخرج كل رمز بتجميع مخرجات الخبراء المُختارين (موزونة)
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # أي الرموز اختارت هذا الخبير ضمن أفضل K؟
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (N_tokens,)

            if expert_mask.sum() == 0:
                continue  # لا رموز لهذا الخبير في هذه الدفعة

            # الوزن المقابل لهذا الخبير لكل رمز اختاره
            expert_weight_positions = (top_k_indices == expert_idx)
            expert_weights = (top_k_probs * expert_weight_positions).sum(dim=-1)  # (N_tokens,)

            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            output[expert_mask] += expert_weights[expert_mask].unsqueeze(-1) * expert_output

        output = output.view(batch_size, seq_len, d_model)

        # 4. حساب Load Balancing Loss
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(self, router_probs, top_k_indices):
        """
        L_balance = N * Σ f_i * P_i
        f_i: نسبة الرموز الفعلية المُوجَّهة للخبير i
        P_i: متوسط احتمال الـ Router للخبير i عبر كل الرموز
        """
        num_tokens = router_probs.shape[0]

        # P_i: متوسط احتمال الـ router لكل خبير (عبر كل الرموز، وليس فقط المُختارة)
        P = router_probs.mean(dim=0)   # (num_experts,)

        # f_i: نسبة الرموز التي اختارت هذا الخبير ضمن Top-K فعليًا
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        # expert_mask: (num_tokens, top_k, num_experts) → نجمع على top_k
        f = expert_mask.sum(dim=1).mean(dim=0) / self.top_k  # (num_experts,)

        aux_loss = self.num_experts * torch.sum(f * P)
        return aux_loss

# ============================================================
# 3. طبقة Transformer كاملة تستخدم MoE بدلاً من FFN القياسية
# ============================================================
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)  # من الفصل 10
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)

    def forward(self, x, mask=None):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, mask)
        x = x + attn_out

        normed = self.ln2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

# ============================================================
# 4. مفهوم DeepSeek-V3 المتقدم: خبراء مشتركون + خبراء متخصصون
# ============================================================
class DeepSeekStyleMoELayer(nn.Module):
    """
    تطبيق مبسّط لفكرة DeepSeek-V3: خبراء مشتركون (shared experts)
    مُفعَّلون دائمًا لكل رمز (يلتقطون المعرفة العامة المشتركة)،
    بالإضافة لخبراء متخصصين (routed experts) يُختارون ديناميكيًا.
    """
    def __init__(self, d_model, d_ff, num_shared_experts=2,
                 num_routed_experts=64, top_k_routed=6, dropout=0.1):
        super().__init__()

        # خبراء مشتركون: نشطون دائمًا لكل رمز، بدون توجيه
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_shared_experts)
        ])

        # خبراء متخصصون: يُختار منهم عبر Top-K Routing
        self.routed_moe = MoELayer(
            d_model, d_ff, num_experts=num_routed_experts,
            top_k=top_k_routed, dropout=dropout
        )

    def forward(self, x):
        # مساهمة الخبراء المشتركين (دائمًا نشطة)
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # مساهمة الخبراء المتخصصين (عبر التوجيه الديناميكي)
        routed_output, aux_loss = self.routed_moe(x)

        return shared_output + routed_output, aux_loss

def compute_moe_efficiency_stats(num_experts, top_k, expert_params, router_params):
    """حساب نسبة السعة الكلية إلى التكلفة الفعلية في MoE."""
    total_params = num_experts * expert_params + router_params
    active_params = top_k * expert_params + router_params

    efficiency_ratio = total_params / active_params

    print(f"إجمالي المعاملات (السعة): {total_params:,}")
    print(f"المعاملات النشطة لكل رمز (التكلفة): {active_params:,}")
    print(f"نسبة السعة إلى التكلفة: {efficiency_ratio:.2f}x")

    return total_params, active_params, efficiency_ratio

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch15-mixture-of-experts.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. خبير واحد (Expert) — شبكة FeedForward قياسية
# ============================================================
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 2. طبقة MoE كاملة مع Top-K Routing وLoad Balancing
# ============================================================
class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        العائد: (output, aux_loss)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)   # (batch*seq_len, d_model) لتبسيط التوجيه

        # 1. حساب درجات الـ Router
        router_logits = self.router(x_flat)              # (N_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # 2. اختيار أفضل K خبراء لكل رمز
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # إعادة تطبيع الاحتمالات على الخبراء المُختارين فقط
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 3. حساب مخرج كل رمز بتجميع مخرجات الخبراء المُختارين (موزونة)
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # أي الرموز اختارت هذا الخبير ضمن أفضل K؟
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (N_tokens,)

            if expert_mask.sum() == 0:
                continue  # لا رموز لهذا الخبير في هذه الدفعة

            # الوزن المقابل لهذا الخبير لكل رمز اختاره
            expert_weight_positions = (top_k_indices == expert_idx)
            expert_weights = (top_k_probs * expert_weight_positions).sum(dim=-1)  # (N_tokens,)

            expert_input = x_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)

            output[expert_mask] += expert_weights[expert_mask].unsqueeze(-1) * expert_output

        output = output.view(batch_size, seq_len, d_model)

        # 4. حساب Load Balancing Loss
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss

    def _compute_load_balancing_loss(self, router_probs, top_k_indices):
        """
        L_balance = N * Σ f_i * P_i
        f_i: نسبة الرموز الفعلية المُوجَّهة للخبير i
        P_i: متوسط احتمال الـ Router للخبير i عبر كل الرموز
        """
        num_tokens = router_probs.shape[0]

        # P_i: متوسط احتمال الـ router لكل خبير (عبر كل الرموز، وليس فقط المُختارة)
        P = router_probs.mean(dim=0)   # (num_experts,)

        # f_i: نسبة الرموز التي اختارت هذا الخبير ضمن Top-K فعليًا
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        # expert_mask: (num_tokens, top_k, num_experts) → نجمع على top_k
        f = expert_mask.sum(dim=1).mean(dim=0) / self.top_k  # (num_experts,)

        aux_loss = self.num_experts * torch.sum(f * P)
        return aux_loss

# ============================================================
# 3. طبقة Transformer كاملة تستخدم MoE بدلاً من FFN القياسية
# ============================================================
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)  # من الفصل 10
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)

    def forward(self, x, mask=None):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, mask)
        x = x + attn_out

        normed = self.ln2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

# ============================================================
# 4. مفهوم DeepSeek-V3 المتقدم: خبراء مشتركون + خبراء متخصصون
# ============================================================
class DeepSeekStyleMoELayer(nn.Module):
    """
    تطبيق مبسّط لفكرة DeepSeek-V3: خبراء مشتركون (shared experts)
    مُفعَّلون دائمًا لكل رمز (يلتقطون المعرفة العامة المشتركة)،
    بالإضافة لخبراء متخصصين (routed experts) يُختارون ديناميكيًا.
    """
    def __init__(self, d_model, d_ff, num_shared_experts=2,
                 num_routed_experts=64, top_k_routed=6, dropout=0.1):
        super().__init__()

        # خبراء مشتركون: نشطون دائمًا لكل رمز، بدون توجيه
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_shared_experts)
        ])

        # خبراء متخصصون: يُختار منهم عبر Top-K Routing
        self.routed_moe = MoELayer(
            d_model, d_ff, num_experts=num_routed_experts,
            top_k=top_k_routed, dropout=dropout
        )

    def forward(self, x):
        # مساهمة الخبراء المشتركين (دائمًا نشطة)
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # مساهمة الخبراء المتخصصين (عبر التوجيه الديناميكي)
        routed_output, aux_loss = self.routed_moe(x)

        return shared_output + routed_output, aux_loss

def compute_moe_efficiency_stats(num_experts, top_k, expert_params, router_params):
    """حساب نسبة السعة الكلية إلى التكلفة الفعلية في MoE."""
    total_params = num_experts * expert_params + router_params
    active_params = top_k * expert_params + router_params

    efficiency_ratio = total_params / active_params

    print(f"إجمالي المعاملات (السعة): {total_params:,}")
    print(f"المعاملات النشطة لكل رمز (التكلفة): {active_params:,}")
    print(f"نسبة السعة إلى التكلفة: {efficiency_ratio:.2f}x")

    return total_params, active_params, efficiency_ratio

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-15-ch15-mixture-of-experts.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
