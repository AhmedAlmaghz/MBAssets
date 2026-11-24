#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الفضاء الحالة: SSM وMamba
Mastering Transformers and Large Language Models — State Space Models

Book slug : mastering-llms-transformers
Chapter   : ch-16-ch16-state-space-models (lesson 16)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-16-ch16-state-space-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الفضاء الحالة: SSM وMamba
Mastering Transformers and Large Language Models — State Space Models

Book slug : mastering-llms-transformers
Chapter   : ch-16-ch16-state-space-models (lesson 16)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-16-ch16-state-space-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الفضاء الحالة: SSM وMamba
Mastering Transformers and Large Language Models — State Space Models

Book slug : mastering-llms-transformers
Chapter   : ch-16-ch16-state-space-models (lesson 16)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch16-state-space-models.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch16-state-space-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. SimplifiedMambaBlock — تطبيق مبسّط تعليمي
#    (التطبيق الفعلي الرسمي يستخدم CUDA kernels مُحسَّنة جدًا
#     لخوارزمية Parallel Scan؛ هذا الكود لفهم الآلية حسابيًا)
# ============================================================
class SimplifiedMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model   # توسيع البعد الداخلي

        # إسقاط المدخل مع توسيع البعد (نمط "Gated" شائع)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # التفاف 1D محلي قصير المدى (يلتقط سياقًا محليًا قبل SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True
        )

        # إسقاطات لحساب B, C, Δ بشكل ديناميكي من المدخل (الانتقائية!)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # مصفوفة A (قابلة للتعلم لكن لا تعتمد مباشرة على x — إعداد S4/S6 القياسي)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """ x: (batch, seq_len, d_model) """
        batch_size, seq_len, _ = x.shape

        # 1. الإسقاط والتوسيع + بوابة (gate)
        x_and_gate = self.in_proj(x)   # (batch, seq_len, 2*d_inner)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)

        # 2. التفاف محلي قصير المدى
        x_conv = x_inner.transpose(1, 2)              # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]   # اقتصاص padding الزائد
        x_conv = x_conv.transpose(1, 2)                # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)   # تفعيل SiLU/Swish

        # 3. حساب B, C, Δ بشكل ديناميكي (الانتقائية الجوهرية لـ Mamba)
        x_dbl = self.x_proj(x_conv)   # (batch, seq_len, 2*d_state + 1)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))   # (batch, seq_len, d_inner)

        A = -torch.exp(self.A_log)   # (d_inner, d_state)، سالبة لضمان استقرار النظام

        # 4. المسح التكراري (Selective Scan) — هنا التطبيق الفعلي يستخدم
        #    CUDA kernel مُحسَّن جدًا؛ هنا حلقة تسلسلية توضيحية فقط
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)              # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))          # (batch, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1)                  # (batch, d_inner, 1)

            h = A_bar * h + delta_t * B_t * x_t                  # تحديث الحالة

            C_t = C[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)                          # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)
        y = y + x_conv * self.D   # اتصال متبقٍّ مباشر (skip connection) بوزن D

        # 5. البوابة النهائية (Gating) ثم الإسقاط للخارج
        y = y * F.silu(gate)
        output = self.out_proj(y)

        return output

# ============================================================
# 2. طبقة Mamba كاملة (مع LayerNorm والاتصال المتبقي)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimplifiedMambaBlock(d_model, d_state, d_conv, expand_factor)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

# ============================================================
# 3. نموذج Mamba كامل مُكدَّس (بديل كامل لـ GPT من الفصل 12)
# ============================================================
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # Weight Tying

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)   # ملاحظة: لا حاجة لأي قناع (mask)! السببية متأصلة في التصميم

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مفهوم Jamba — معمارية هجينة (Transformer + Mamba)
# ============================================================
class JambaStyleBlock(nn.Module):
    """
    كتلة هجينة تتناوب بين Mamba (للكفاءة والسياق الطويل)
    وMulti-Head Attention (للقدرة على الربط الدقيق بين عناصر بعيدة محددة).
    """
    def __init__(self, d_model, num_heads, d_ff, use_attention: bool,
                 d_state=16, num_experts=8, top_k=2):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = MultiHeadAttention(d_model, num_heads)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = SimplifiedMambaBlock(d_model, d_state)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)  # من الفصل 15

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        if self.use_attention:
            mix_out, _ = self.mixer(normed, normed, normed, mask)
        else:
            mix_out = self.mixer(normed)
        x = x + mix_out

        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

def build_jamba_style_model(num_layers=8, attention_ratio=0.25):
    """
    بناء تسلسل من الطبقات حيث نسبة صغيرة فقط (مثلًا 1 من كل 4، كما في Jamba)
    تستخدم Attention الكامل، والباقي يستخدم Mamba الأكفأ.
    """
    attention_positions = set(
        int(i / attention_ratio) for i in range(int(num_layers * attention_ratio))
    )
    layer_types = ["attention" if i in attention_positions else "mamba"
                   for i in range(num_layers)]
    return layer_types

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch16-state-space-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. SimplifiedMambaBlock — تطبيق مبسّط تعليمي
#    (التطبيق الفعلي الرسمي يستخدم CUDA kernels مُحسَّنة جدًا
#     لخوارزمية Parallel Scan؛ هذا الكود لفهم الآلية حسابيًا)
# ============================================================
class SimplifiedMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model   # توسيع البعد الداخلي

        # إسقاط المدخل مع توسيع البعد (نمط "Gated" شائع)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # التفاف 1D محلي قصير المدى (يلتقط سياقًا محليًا قبل SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True
        )

        # إسقاطات لحساب B, C, Δ بشكل ديناميكي من المدخل (الانتقائية!)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # مصفوفة A (قابلة للتعلم لكن لا تعتمد مباشرة على x — إعداد S4/S6 القياسي)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """ x: (batch, seq_len, d_model) """
        batch_size, seq_len, _ = x.shape

        # 1. الإسقاط والتوسيع + بوابة (gate)
        x_and_gate = self.in_proj(x)   # (batch, seq_len, 2*d_inner)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)

        # 2. التفاف محلي قصير المدى
        x_conv = x_inner.transpose(1, 2)              # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]   # اقتصاص padding الزائد
        x_conv = x_conv.transpose(1, 2)                # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)   # تفعيل SiLU/Swish

        # 3. حساب B, C, Δ بشكل ديناميكي (الانتقائية الجوهرية لـ Mamba)
        x_dbl = self.x_proj(x_conv)   # (batch, seq_len, 2*d_state + 1)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))   # (batch, seq_len, d_inner)

        A = -torch.exp(self.A_log)   # (d_inner, d_state)، سالبة لضمان استقرار النظام

        # 4. المسح التكراري (Selective Scan) — هنا التطبيق الفعلي يستخدم
        #    CUDA kernel مُحسَّن جدًا؛ هنا حلقة تسلسلية توضيحية فقط
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)              # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))          # (batch, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1)                  # (batch, d_inner, 1)

            h = A_bar * h + delta_t * B_t * x_t                  # تحديث الحالة

            C_t = C[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)                          # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)
        y = y + x_conv * self.D   # اتصال متبقٍّ مباشر (skip connection) بوزن D

        # 5. البوابة النهائية (Gating) ثم الإسقاط للخارج
        y = y * F.silu(gate)
        output = self.out_proj(y)

        return output

# ============================================================
# 2. طبقة Mamba كاملة (مع LayerNorm والاتصال المتبقي)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimplifiedMambaBlock(d_model, d_state, d_conv, expand_factor)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

# ============================================================
# 3. نموذج Mamba كامل مُكدَّس (بديل كامل لـ GPT من الفصل 12)
# ============================================================
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # Weight Tying

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)   # ملاحظة: لا حاجة لأي قناع (mask)! السببية متأصلة في التصميم

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مفهوم Jamba — معمارية هجينة (Transformer + Mamba)
# ============================================================
class JambaStyleBlock(nn.Module):
    """
    كتلة هجينة تتناوب بين Mamba (للكفاءة والسياق الطويل)
    وMulti-Head Attention (للقدرة على الربط الدقيق بين عناصر بعيدة محددة).
    """
    def __init__(self, d_model, num_heads, d_ff, use_attention: bool,
                 d_state=16, num_experts=8, top_k=2):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = MultiHeadAttention(d_model, num_heads)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = SimplifiedMambaBlock(d_model, d_state)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)  # من الفصل 15

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        if self.use_attention:
            mix_out, _ = self.mixer(normed, normed, normed, mask)
        else:
            mix_out = self.mixer(normed)
        x = x + mix_out

        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

def build_jamba_style_model(num_layers=8, attention_ratio=0.25):
    """
    بناء تسلسل من الطبقات حيث نسبة صغيرة فقط (مثلًا 1 من كل 4، كما في Jamba)
    تستخدم Attention الكامل، والباقي يستخدم Mamba الأكفأ.
    """
    attention_positions = set(
        int(i / attention_ratio) for i in range(int(num_layers * attention_ratio))
    )
    layer_types = ["attention" if i in attention_positions else "mamba"
                   for i in range(num_layers)]
    return layer_types

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الفضاء الحالة: SSM وMamba
Mastering Transformers and Large Language Models — State Space Models

Book slug : mastering-llms-transformers
Chapter   : ch-16-ch16-state-space-models (lesson 16)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch16-state-space-models.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch16-state-space-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. SimplifiedMambaBlock — تطبيق مبسّط تعليمي
#    (التطبيق الفعلي الرسمي يستخدم CUDA kernels مُحسَّنة جدًا
#     لخوارزمية Parallel Scan؛ هذا الكود لفهم الآلية حسابيًا)
# ============================================================
class SimplifiedMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model   # توسيع البعد الداخلي

        # إسقاط المدخل مع توسيع البعد (نمط "Gated" شائع)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # التفاف 1D محلي قصير المدى (يلتقط سياقًا محليًا قبل SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True
        )

        # إسقاطات لحساب B, C, Δ بشكل ديناميكي من المدخل (الانتقائية!)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # مصفوفة A (قابلة للتعلم لكن لا تعتمد مباشرة على x — إعداد S4/S6 القياسي)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """ x: (batch, seq_len, d_model) """
        batch_size, seq_len, _ = x.shape

        # 1. الإسقاط والتوسيع + بوابة (gate)
        x_and_gate = self.in_proj(x)   # (batch, seq_len, 2*d_inner)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)

        # 2. التفاف محلي قصير المدى
        x_conv = x_inner.transpose(1, 2)              # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]   # اقتصاص padding الزائد
        x_conv = x_conv.transpose(1, 2)                # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)   # تفعيل SiLU/Swish

        # 3. حساب B, C, Δ بشكل ديناميكي (الانتقائية الجوهرية لـ Mamba)
        x_dbl = self.x_proj(x_conv)   # (batch, seq_len, 2*d_state + 1)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))   # (batch, seq_len, d_inner)

        A = -torch.exp(self.A_log)   # (d_inner, d_state)، سالبة لضمان استقرار النظام

        # 4. المسح التكراري (Selective Scan) — هنا التطبيق الفعلي يستخدم
        #    CUDA kernel مُحسَّن جدًا؛ هنا حلقة تسلسلية توضيحية فقط
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)              # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))          # (batch, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1)                  # (batch, d_inner, 1)

            h = A_bar * h + delta_t * B_t * x_t                  # تحديث الحالة

            C_t = C[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)                          # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)
        y = y + x_conv * self.D   # اتصال متبقٍّ مباشر (skip connection) بوزن D

        # 5. البوابة النهائية (Gating) ثم الإسقاط للخارج
        y = y * F.silu(gate)
        output = self.out_proj(y)

        return output

# ============================================================
# 2. طبقة Mamba كاملة (مع LayerNorm والاتصال المتبقي)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimplifiedMambaBlock(d_model, d_state, d_conv, expand_factor)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

# ============================================================
# 3. نموذج Mamba كامل مُكدَّس (بديل كامل لـ GPT من الفصل 12)
# ============================================================
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # Weight Tying

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)   # ملاحظة: لا حاجة لأي قناع (mask)! السببية متأصلة في التصميم

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مفهوم Jamba — معمارية هجينة (Transformer + Mamba)
# ============================================================
class JambaStyleBlock(nn.Module):
    """
    كتلة هجينة تتناوب بين Mamba (للكفاءة والسياق الطويل)
    وMulti-Head Attention (للقدرة على الربط الدقيق بين عناصر بعيدة محددة).
    """
    def __init__(self, d_model, num_heads, d_ff, use_attention: bool,
                 d_state=16, num_experts=8, top_k=2):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = MultiHeadAttention(d_model, num_heads)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = SimplifiedMambaBlock(d_model, d_state)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)  # من الفصل 15

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        if self.use_attention:
            mix_out, _ = self.mixer(normed, normed, normed, mask)
        else:
            mix_out = self.mixer(normed)
        x = x + mix_out

        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

def build_jamba_style_model(num_layers=8, attention_ratio=0.25):
    """
    بناء تسلسل من الطبقات حيث نسبة صغيرة فقط (مثلًا 1 من كل 4، كما في Jamba)
    تستخدم Attention الكامل، والباقي يستخدم Mamba الأكفأ.
    """
    attention_positions = set(
        int(i / attention_ratio) for i in range(int(num_layers * attention_ratio))
    )
    layer_types = ["attention" if i in attention_positions else "mamba"
                   for i in range(num_layers)]
    return layer_types

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch16-state-space-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. SimplifiedMambaBlock — تطبيق مبسّط تعليمي
#    (التطبيق الفعلي الرسمي يستخدم CUDA kernels مُحسَّنة جدًا
#     لخوارزمية Parallel Scan؛ هذا الكود لفهم الآلية حسابيًا)
# ============================================================
class SimplifiedMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model   # توسيع البعد الداخلي

        # إسقاط المدخل مع توسيع البعد (نمط "Gated" شائع)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # التفاف 1D محلي قصير المدى (يلتقط سياقًا محليًا قبل SSM)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True
        )

        # إسقاطات لحساب B, C, Δ بشكل ديناميكي من المدخل (الانتقائية!)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # مصفوفة A (قابلة للتعلم لكن لا تعتمد مباشرة على x — إعداد S4/S6 القياسي)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """ x: (batch, seq_len, d_model) """
        batch_size, seq_len, _ = x.shape

        # 1. الإسقاط والتوسيع + بوابة (gate)
        x_and_gate = self.in_proj(x)   # (batch, seq_len, 2*d_inner)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)

        # 2. التفاف محلي قصير المدى
        x_conv = x_inner.transpose(1, 2)              # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]   # اقتصاص padding الزائد
        x_conv = x_conv.transpose(1, 2)                # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)   # تفعيل SiLU/Swish

        # 3. حساب B, C, Δ بشكل ديناميكي (الانتقائية الجوهرية لـ Mamba)
        x_dbl = self.x_proj(x_conv)   # (batch, seq_len, 2*d_state + 1)
        delta, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))   # (batch, seq_len, d_inner)

        A = -torch.exp(self.A_log)   # (d_inner, d_state)، سالبة لضمان استقرار النظام

        # 4. المسح التكراري (Selective Scan) — هنا التطبيق الفعلي يستخدم
        #    CUDA kernel مُحسَّن جدًا؛ هنا حلقة تسلسلية توضيحية فقط
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)              # (batch, d_inner, 1)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))          # (batch, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            x_t = x_conv[:, t, :].unsqueeze(-1)                  # (batch, d_inner, 1)

            h = A_bar * h + delta_t * B_t * x_t                  # تحديث الحالة

            C_t = C[:, t, :].unsqueeze(1)                        # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)                          # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)   # (batch, seq_len, d_inner)
        y = y + x_conv * self.D   # اتصال متبقٍّ مباشر (skip connection) بوزن D

        # 5. البوابة النهائية (Gating) ثم الإسقاط للخارج
        y = y * F.silu(gate)
        output = self.out_proj(y)

        return output

# ============================================================
# 2. طبقة Mamba كاملة (مع LayerNorm والاتصال المتبقي)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimplifiedMambaBlock(d_model, d_state, d_conv, expand_factor)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

# ============================================================
# 3. نموذج Mamba كامل مُكدَّس (بديل كامل لـ GPT من الفصل 12)
# ============================================================
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight   # Weight Tying

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)   # ملاحظة: لا حاجة لأي قناع (mask)! السببية متأصلة في التصميم

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مفهوم Jamba — معمارية هجينة (Transformer + Mamba)
# ============================================================
class JambaStyleBlock(nn.Module):
    """
    كتلة هجينة تتناوب بين Mamba (للكفاءة والسياق الطويل)
    وMulti-Head Attention (للقدرة على الربط الدقيق بين عناصر بعيدة محددة).
    """
    def __init__(self, d_model, num_heads, d_ff, use_attention: bool,
                 d_state=16, num_experts=8, top_k=2):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = MultiHeadAttention(d_model, num_heads)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.mixer = SimplifiedMambaBlock(d_model, d_state)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)  # من الفصل 15

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        if self.use_attention:
            mix_out, _ = self.mixer(normed, normed, normed, mask)
        else:
            mix_out = self.mixer(normed)
        x = x + mix_out

        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss

def build_jamba_style_model(num_layers=8, attention_ratio=0.25):
    """
    بناء تسلسل من الطبقات حيث نسبة صغيرة فقط (مثلًا 1 من كل 4، كما في Jamba)
    تستخدم Attention الكامل، والباقي يستخدم Mamba الأكفأ.
    """
    attention_positions = set(
        int(i / attention_ratio) for i in range(int(num_layers * attention_ratio))
    )
    layer_types = ["attention" if i in attention_positions else "mamba"
                   for i in range(num_layers)]
    return layer_types

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-16-ch16-state-space-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
