#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المعماريات التالية لما بعد Transformer
Mastering Transformers and Large Language Models — Post Transformer

Book slug : mastering-llms-transformers
Chapter   : ch-62-ch62-post-transformer (lesson 62)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-62-ch62-post-transformer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المعماريات التالية لما بعد Transformer
Mastering Transformers and Large Language Models — Post Transformer

Book slug : mastering-llms-transformers
Chapter   : ch-62-ch62-post-transformer (lesson 62)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-62-ch62-post-transformer.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المعماريات التالية لما بعد Transformer
Mastering Transformers and Large Language Models — Post Transformer

Book slug : mastering-llms-transformers
Chapter   : ch-62-ch62-post-transformer (lesson 62)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-1
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
# [002] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MixtureOfDepthsBlock(nn.Module):
    """
    تطبيق مبسّط لفكرة Mixture-of-Depths (DeepMind, 2024):
    كل طبقة تُقرر - عبر Router - أي الرموز "تستحق" المرور عبر الحساب
    الكامل (Attention + FFN) وأيها يمكنه "تخطي" الطبقة (Residual مباشر).
    النتيجة: توفير حسابي كبير مع الحفاظ على الأداء، لأن بعض الرموز
    (كعلامات الترقيم أو الكلمات المتكررة السهلة) لا تحتاج معالجة عميقة.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 capacity_fraction: float = 0.5):
        super().__init__()
        # capacity_fraction: نسبة الرموز التي "تمر" عبر الحساب الكامل
        # (مثال: 0.5 يعني فقط نصف الرموز في كل تسلسل تُعالَج بالكامل)
        self.capacity_fraction = capacity_fraction
        self.router = nn.Linear(d_model, 1)  # يُنتج درجة أهمية لكل رمز

        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        capacity = int(seq_len * self.capacity_fraction)

        # 1. الـ Router يحسب درجة أهمية لكل رمز
        router_scores = self.router(x).squeeze(-1)  # [batch, seq_len]

        # 2. اختيار أعلى `capacity` رمز من حيث الأهمية لكل عيّنة في الدفعة
        top_k_scores, top_k_indices = torch.topk(
            router_scores, k=capacity, dim=-1
        )
        # ترتيب الفهارس للحفاظ على الترتيب الأصلي للتسلسل (مهم للـ causal attention)
        top_k_indices, _ = torch.sort(top_k_indices, dim=-1)

        output = x.clone()

        for b in range(batch_size):
            selected_indices = top_k_indices[b]
            selected_tokens = x[b, selected_indices].unsqueeze(0)  # [1, capacity, d]

            # 3. الرموز المختارة فقط تمر عبر Attention + FFN الكاملين
            attn_out, _ = self.attention(
                selected_tokens, selected_tokens, selected_tokens,
                attn_mask=self._causal_mask(capacity, x.device)
            )
            selected_tokens = self.norm1(selected_tokens + attn_out)
            ffn_out = self.ffn(selected_tokens)
            processed_tokens = self.norm2(selected_tokens + ffn_out)

            # 4. دمج الرموز المُعالَجة مرة أخرى في مواضعها الأصلية
            #    (الرموز غير المختارة تمرّ عبر Residual فقط - بدون معالجة)
            output[b, selected_indices] = processed_tokens.squeeze(0)

        return output

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device), diagonal=1
        )

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DynamicBytePatching:
    """
    تطبيق مفاهيمي لفكرة BLT (Byte Latent Transformer):
    بدلاً من Tokenizer ثابت المفردات، تُجمَّع البايتات في "رقع" (Patches)
    ديناميكية الطول بناءً على التعقيد المحلي للنص (Entropy-based)،
    مقاسةً بواسطة نموذج لغوي صغير يعمل على مستوى البايت.
    """
    def __init__(self, byte_lm, entropy_threshold: float = 1.5):
        self.byte_lm = byte_lm  # نموذج صغير يتنبأ بالبايت التالي
        self.entropy_threshold = entropy_threshold

    def compute_byte_entropy(self, byte_sequence: bytes) -> list:
        """حساب إنتروبيا التنبؤ بكل بايت بناءً على ما سبقه (عدم اليقين)."""
        entropies = []
        for i in range(1, len(byte_sequence)):
            context = byte_sequence[:i]
            next_byte_probs = self.byte_lm.predict_next_byte_distribution(context)
            entropy = -sum(p * torch.log(p + 1e-10) for p in next_byte_probs)
            entropies.append(entropy.item())
        return entropies

    def create_dynamic_patches(self, byte_sequence: bytes) -> list:
        """
        إنشاء رقع ديناميكية: حد رقعة جديدة يُفتَح عند نقاط عدم اليقين
        العالي (إنتروبيا مرتفعة = بداية كلمة/مفهوم جديد محتمل)،
        بينما تُجمَّع البايتات شديدة القابلية للتنبؤ (مثل نهايات الكلمات
        الشائعة) في رقعة واحدة أطول. هذا يُنتج ضغطًا تكيفيًا: نص بسيط
        متكرر → رقع طويلة (حوسبة أقل)، نص معقد/نادر → رقع قصيرة (حوسبة أدق).
        """
        entropies = self.compute_byte_entropy(byte_sequence)
        patches = []
        current_patch_start = 0

        for i, entropy in enumerate(entropies):
            if entropy > self.entropy_threshold:
                patches.append(byte_sequence[current_patch_start:i+1])
                current_patch_start = i + 1

        if current_patch_start < len(byte_sequence):
            patches.append(byte_sequence[current_patch_start:])

        return patches

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class AdaptiveHybridBlock(nn.Module):
    """
    معمارية هجينة متقدمة (بأسلوب Hymba/Zamba 2024): بدلاً من تبديل طبقات
    Attention وMamba بالتناوب كما في Jamba البسيط، تُدمَج مساراتهما
    بالتوازي (Parallel Heads) داخل نفس الطبقة، مع دمج مُرجَّح يسمح
    للنموذج بتعلّم متى يعتمد على كل مسار (ذاكرة دقيقة قصيرة المدى عبر
    Attention مقابل ملخّص مُضغوط طويل المدى عبر SSM).
    """
    def __init__(self, d_model: int, n_attn_heads: int, ssm_state_dim: int):
        super().__init__()
        # مسار Attention الدقيق (يحتفظ بتفاصيل كاملة - محدود بطول نافذة محلية)
        self.local_attention = nn.MultiheadAttention(
            d_model, n_attn_heads, batch_first=True
        )
        self.local_window_size = 256  # نافذة محلية فقط - اقتصاد حسابي

        # مسار SSM (يحتفظ بملخص مُضغوط للسياق الكامل بلا حدود عملية)
        from_chapter_16_mamba_block = None  # راجع SimplifiedMambaBlock (الفصل 16)
        self.ssm_path = SimplifiedMambaBlock(d_model, state_dim=ssm_state_dim)

        # بوابة دمج مُتعلَّمة (Learned Fusion Gate)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # معالجة متوازية (وليست متسلسلة) للمسارين
        local_out = self._windowed_attention(x)
        ssm_out = self.ssm_path(x)

        # بوابة تُقرر نسبة المزج بين المسارين لكل رمز ديناميكيًا
        gate_input = torch.cat([local_out, ssm_out], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused = gate * local_out + (1 - gate) * ssm_out

        return self.norm(x + fused)

    def _windowed_attention(self, x):
        # تطبيق Attention ضمن نافذة محلية منزلقة فقط (اقتصاد O(n·w) بدل O(n²))
        # تفاصيل Sliding Window Attention غُطيت ضمنيًا في الفصل 14
        return x  # تبسيط توضيحي

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MixtureOfDepthsBlock(nn.Module):
    """
    تطبيق مبسّط لفكرة Mixture-of-Depths (DeepMind, 2024):
    كل طبقة تُقرر - عبر Router - أي الرموز "تستحق" المرور عبر الحساب
    الكامل (Attention + FFN) وأيها يمكنه "تخطي" الطبقة (Residual مباشر).
    النتيجة: توفير حسابي كبير مع الحفاظ على الأداء، لأن بعض الرموز
    (كعلامات الترقيم أو الكلمات المتكررة السهلة) لا تحتاج معالجة عميقة.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 capacity_fraction: float = 0.5):
        super().__init__()
        # capacity_fraction: نسبة الرموز التي "تمر" عبر الحساب الكامل
        # (مثال: 0.5 يعني فقط نصف الرموز في كل تسلسل تُعالَج بالكامل)
        self.capacity_fraction = capacity_fraction
        self.router = nn.Linear(d_model, 1)  # يُنتج درجة أهمية لكل رمز

        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        capacity = int(seq_len * self.capacity_fraction)

        # 1. الـ Router يحسب درجة أهمية لكل رمز
        router_scores = self.router(x).squeeze(-1)  # [batch, seq_len]

        # 2. اختيار أعلى `capacity` رمز من حيث الأهمية لكل عيّنة في الدفعة
        top_k_scores, top_k_indices = torch.topk(
            router_scores, k=capacity, dim=-1
        )
        # ترتيب الفهارس للحفاظ على الترتيب الأصلي للتسلسل (مهم للـ causal attention)
        top_k_indices, _ = torch.sort(top_k_indices, dim=-1)

        output = x.clone()

        for b in range(batch_size):
            selected_indices = top_k_indices[b]
            selected_tokens = x[b, selected_indices].unsqueeze(0)  # [1, capacity, d]

            # 3. الرموز المختارة فقط تمر عبر Attention + FFN الكاملين
            attn_out, _ = self.attention(
                selected_tokens, selected_tokens, selected_tokens,
                attn_mask=self._causal_mask(capacity, x.device)
            )
            selected_tokens = self.norm1(selected_tokens + attn_out)
            ffn_out = self.ffn(selected_tokens)
            processed_tokens = self.norm2(selected_tokens + ffn_out)

            # 4. دمج الرموز المُعالَجة مرة أخرى في مواضعها الأصلية
            #    (الرموز غير المختارة تمرّ عبر Residual فقط - بدون معالجة)
            output[b, selected_indices] = processed_tokens.squeeze(0)

        return output

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device), diagonal=1
        )

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DynamicBytePatching:
    """
    تطبيق مفاهيمي لفكرة BLT (Byte Latent Transformer):
    بدلاً من Tokenizer ثابت المفردات، تُجمَّع البايتات في "رقع" (Patches)
    ديناميكية الطول بناءً على التعقيد المحلي للنص (Entropy-based)،
    مقاسةً بواسطة نموذج لغوي صغير يعمل على مستوى البايت.
    """
    def __init__(self, byte_lm, entropy_threshold: float = 1.5):
        self.byte_lm = byte_lm  # نموذج صغير يتنبأ بالبايت التالي
        self.entropy_threshold = entropy_threshold

    def compute_byte_entropy(self, byte_sequence: bytes) -> list:
        """حساب إنتروبيا التنبؤ بكل بايت بناءً على ما سبقه (عدم اليقين)."""
        entropies = []
        for i in range(1, len(byte_sequence)):
            context = byte_sequence[:i]
            next_byte_probs = self.byte_lm.predict_next_byte_distribution(context)
            entropy = -sum(p * torch.log(p + 1e-10) for p in next_byte_probs)
            entropies.append(entropy.item())
        return entropies

    def create_dynamic_patches(self, byte_sequence: bytes) -> list:
        """
        إنشاء رقع ديناميكية: حد رقعة جديدة يُفتَح عند نقاط عدم اليقين
        العالي (إنتروبيا مرتفعة = بداية كلمة/مفهوم جديد محتمل)،
        بينما تُجمَّع البايتات شديدة القابلية للتنبؤ (مثل نهايات الكلمات
        الشائعة) في رقعة واحدة أطول. هذا يُنتج ضغطًا تكيفيًا: نص بسيط
        متكرر → رقع طويلة (حوسبة أقل)، نص معقد/نادر → رقع قصيرة (حوسبة أدق).
        """
        entropies = self.compute_byte_entropy(byte_sequence)
        patches = []
        current_patch_start = 0

        for i, entropy in enumerate(entropies):
            if entropy > self.entropy_threshold:
                patches.append(byte_sequence[current_patch_start:i+1])
                current_patch_start = i + 1

        if current_patch_start < len(byte_sequence):
            patches.append(byte_sequence[current_patch_start:])

        return patches

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class AdaptiveHybridBlock(nn.Module):
    """
    معمارية هجينة متقدمة (بأسلوب Hymba/Zamba 2024): بدلاً من تبديل طبقات
    Attention وMamba بالتناوب كما في Jamba البسيط، تُدمَج مساراتهما
    بالتوازي (Parallel Heads) داخل نفس الطبقة، مع دمج مُرجَّح يسمح
    للنموذج بتعلّم متى يعتمد على كل مسار (ذاكرة دقيقة قصيرة المدى عبر
    Attention مقابل ملخّص مُضغوط طويل المدى عبر SSM).
    """
    def __init__(self, d_model: int, n_attn_heads: int, ssm_state_dim: int):
        super().__init__()
        # مسار Attention الدقيق (يحتفظ بتفاصيل كاملة - محدود بطول نافذة محلية)
        self.local_attention = nn.MultiheadAttention(
            d_model, n_attn_heads, batch_first=True
        )
        self.local_window_size = 256  # نافذة محلية فقط - اقتصاد حسابي

        # مسار SSM (يحتفظ بملخص مُضغوط للسياق الكامل بلا حدود عملية)
        from_chapter_16_mamba_block = None  # راجع SimplifiedMambaBlock (الفصل 16)
        self.ssm_path = SimplifiedMambaBlock(d_model, state_dim=ssm_state_dim)

        # بوابة دمج مُتعلَّمة (Learned Fusion Gate)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # معالجة متوازية (وليست متسلسلة) للمسارين
        local_out = self._windowed_attention(x)
        ssm_out = self.ssm_path(x)

        # بوابة تُقرر نسبة المزج بين المسارين لكل رمز ديناميكيًا
        gate_input = torch.cat([local_out, ssm_out], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused = gate * local_out + (1 - gate) * ssm_out

        return self.norm(x + fused)

    def _windowed_attention(self, x):
        # تطبيق Attention ضمن نافذة محلية منزلقة فقط (اقتصاد O(n·w) بدل O(n²))
        # تفاصيل Sliding Window Attention غُطيت ضمنيًا في الفصل 14
        return x  # تبسيط توضيحي

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المعماريات التالية لما بعد Transformer
Mastering Transformers and Large Language Models — Post Transformer

Book slug : mastering-llms-transformers
Chapter   : ch-62-ch62-post-transformer (lesson 62)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-1
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
# [002] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MixtureOfDepthsBlock(nn.Module):
    """
    تطبيق مبسّط لفكرة Mixture-of-Depths (DeepMind, 2024):
    كل طبقة تُقرر - عبر Router - أي الرموز "تستحق" المرور عبر الحساب
    الكامل (Attention + FFN) وأيها يمكنه "تخطي" الطبقة (Residual مباشر).
    النتيجة: توفير حسابي كبير مع الحفاظ على الأداء، لأن بعض الرموز
    (كعلامات الترقيم أو الكلمات المتكررة السهلة) لا تحتاج معالجة عميقة.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 capacity_fraction: float = 0.5):
        super().__init__()
        # capacity_fraction: نسبة الرموز التي "تمر" عبر الحساب الكامل
        # (مثال: 0.5 يعني فقط نصف الرموز في كل تسلسل تُعالَج بالكامل)
        self.capacity_fraction = capacity_fraction
        self.router = nn.Linear(d_model, 1)  # يُنتج درجة أهمية لكل رمز

        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        capacity = int(seq_len * self.capacity_fraction)

        # 1. الـ Router يحسب درجة أهمية لكل رمز
        router_scores = self.router(x).squeeze(-1)  # [batch, seq_len]

        # 2. اختيار أعلى `capacity` رمز من حيث الأهمية لكل عيّنة في الدفعة
        top_k_scores, top_k_indices = torch.topk(
            router_scores, k=capacity, dim=-1
        )
        # ترتيب الفهارس للحفاظ على الترتيب الأصلي للتسلسل (مهم للـ causal attention)
        top_k_indices, _ = torch.sort(top_k_indices, dim=-1)

        output = x.clone()

        for b in range(batch_size):
            selected_indices = top_k_indices[b]
            selected_tokens = x[b, selected_indices].unsqueeze(0)  # [1, capacity, d]

            # 3. الرموز المختارة فقط تمر عبر Attention + FFN الكاملين
            attn_out, _ = self.attention(
                selected_tokens, selected_tokens, selected_tokens,
                attn_mask=self._causal_mask(capacity, x.device)
            )
            selected_tokens = self.norm1(selected_tokens + attn_out)
            ffn_out = self.ffn(selected_tokens)
            processed_tokens = self.norm2(selected_tokens + ffn_out)

            # 4. دمج الرموز المُعالَجة مرة أخرى في مواضعها الأصلية
            #    (الرموز غير المختارة تمرّ عبر Residual فقط - بدون معالجة)
            output[b, selected_indices] = processed_tokens.squeeze(0)

        return output

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device), diagonal=1
        )

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DynamicBytePatching:
    """
    تطبيق مفاهيمي لفكرة BLT (Byte Latent Transformer):
    بدلاً من Tokenizer ثابت المفردات، تُجمَّع البايتات في "رقع" (Patches)
    ديناميكية الطول بناءً على التعقيد المحلي للنص (Entropy-based)،
    مقاسةً بواسطة نموذج لغوي صغير يعمل على مستوى البايت.
    """
    def __init__(self, byte_lm, entropy_threshold: float = 1.5):
        self.byte_lm = byte_lm  # نموذج صغير يتنبأ بالبايت التالي
        self.entropy_threshold = entropy_threshold

    def compute_byte_entropy(self, byte_sequence: bytes) -> list:
        """حساب إنتروبيا التنبؤ بكل بايت بناءً على ما سبقه (عدم اليقين)."""
        entropies = []
        for i in range(1, len(byte_sequence)):
            context = byte_sequence[:i]
            next_byte_probs = self.byte_lm.predict_next_byte_distribution(context)
            entropy = -sum(p * torch.log(p + 1e-10) for p in next_byte_probs)
            entropies.append(entropy.item())
        return entropies

    def create_dynamic_patches(self, byte_sequence: bytes) -> list:
        """
        إنشاء رقع ديناميكية: حد رقعة جديدة يُفتَح عند نقاط عدم اليقين
        العالي (إنتروبيا مرتفعة = بداية كلمة/مفهوم جديد محتمل)،
        بينما تُجمَّع البايتات شديدة القابلية للتنبؤ (مثل نهايات الكلمات
        الشائعة) في رقعة واحدة أطول. هذا يُنتج ضغطًا تكيفيًا: نص بسيط
        متكرر → رقع طويلة (حوسبة أقل)، نص معقد/نادر → رقع قصيرة (حوسبة أدق).
        """
        entropies = self.compute_byte_entropy(byte_sequence)
        patches = []
        current_patch_start = 0

        for i, entropy in enumerate(entropies):
            if entropy > self.entropy_threshold:
                patches.append(byte_sequence[current_patch_start:i+1])
                current_patch_start = i + 1

        if current_patch_start < len(byte_sequence):
            patches.append(byte_sequence[current_patch_start:])

        return patches

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class AdaptiveHybridBlock(nn.Module):
    """
    معمارية هجينة متقدمة (بأسلوب Hymba/Zamba 2024): بدلاً من تبديل طبقات
    Attention وMamba بالتناوب كما في Jamba البسيط، تُدمَج مساراتهما
    بالتوازي (Parallel Heads) داخل نفس الطبقة، مع دمج مُرجَّح يسمح
    للنموذج بتعلّم متى يعتمد على كل مسار (ذاكرة دقيقة قصيرة المدى عبر
    Attention مقابل ملخّص مُضغوط طويل المدى عبر SSM).
    """
    def __init__(self, d_model: int, n_attn_heads: int, ssm_state_dim: int):
        super().__init__()
        # مسار Attention الدقيق (يحتفظ بتفاصيل كاملة - محدود بطول نافذة محلية)
        self.local_attention = nn.MultiheadAttention(
            d_model, n_attn_heads, batch_first=True
        )
        self.local_window_size = 256  # نافذة محلية فقط - اقتصاد حسابي

        # مسار SSM (يحتفظ بملخص مُضغوط للسياق الكامل بلا حدود عملية)
        from_chapter_16_mamba_block = None  # راجع SimplifiedMambaBlock (الفصل 16)
        self.ssm_path = SimplifiedMambaBlock(d_model, state_dim=ssm_state_dim)

        # بوابة دمج مُتعلَّمة (Learned Fusion Gate)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # معالجة متوازية (وليست متسلسلة) للمسارين
        local_out = self._windowed_attention(x)
        ssm_out = self.ssm_path(x)

        # بوابة تُقرر نسبة المزج بين المسارين لكل رمز ديناميكيًا
        gate_input = torch.cat([local_out, ssm_out], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused = gate * local_out + (1 - gate) * ssm_out

        return self.norm(x + fused)

    def _windowed_attention(self, x):
        # تطبيق Attention ضمن نافذة محلية منزلقة فقط (اقتصاد O(n·w) بدل O(n²))
        # تفاصيل Sliding Window Attention غُطيت ضمنيًا في الفصل 14
        return x  # تبسيط توضيحي

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-12-future/ch62-post-transformer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MixtureOfDepthsBlock(nn.Module):
    """
    تطبيق مبسّط لفكرة Mixture-of-Depths (DeepMind, 2024):
    كل طبقة تُقرر - عبر Router - أي الرموز "تستحق" المرور عبر الحساب
    الكامل (Attention + FFN) وأيها يمكنه "تخطي" الطبقة (Residual مباشر).
    النتيجة: توفير حسابي كبير مع الحفاظ على الأداء، لأن بعض الرموز
    (كعلامات الترقيم أو الكلمات المتكررة السهلة) لا تحتاج معالجة عميقة.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 capacity_fraction: float = 0.5):
        super().__init__()
        # capacity_fraction: نسبة الرموز التي "تمر" عبر الحساب الكامل
        # (مثال: 0.5 يعني فقط نصف الرموز في كل تسلسل تُعالَج بالكامل)
        self.capacity_fraction = capacity_fraction
        self.router = nn.Linear(d_model, 1)  # يُنتج درجة أهمية لكل رمز

        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        capacity = int(seq_len * self.capacity_fraction)

        # 1. الـ Router يحسب درجة أهمية لكل رمز
        router_scores = self.router(x).squeeze(-1)  # [batch, seq_len]

        # 2. اختيار أعلى `capacity` رمز من حيث الأهمية لكل عيّنة في الدفعة
        top_k_scores, top_k_indices = torch.topk(
            router_scores, k=capacity, dim=-1
        )
        # ترتيب الفهارس للحفاظ على الترتيب الأصلي للتسلسل (مهم للـ causal attention)
        top_k_indices, _ = torch.sort(top_k_indices, dim=-1)

        output = x.clone()

        for b in range(batch_size):
            selected_indices = top_k_indices[b]
            selected_tokens = x[b, selected_indices].unsqueeze(0)  # [1, capacity, d]

            # 3. الرموز المختارة فقط تمر عبر Attention + FFN الكاملين
            attn_out, _ = self.attention(
                selected_tokens, selected_tokens, selected_tokens,
                attn_mask=self._causal_mask(capacity, x.device)
            )
            selected_tokens = self.norm1(selected_tokens + attn_out)
            ffn_out = self.ffn(selected_tokens)
            processed_tokens = self.norm2(selected_tokens + ffn_out)

            # 4. دمج الرموز المُعالَجة مرة أخرى في مواضعها الأصلية
            #    (الرموز غير المختارة تمرّ عبر Residual فقط - بدون معالجة)
            output[b, selected_indices] = processed_tokens.squeeze(0)

        return output

    def _causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device), diagonal=1
        )

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class DynamicBytePatching:
    """
    تطبيق مفاهيمي لفكرة BLT (Byte Latent Transformer):
    بدلاً من Tokenizer ثابت المفردات، تُجمَّع البايتات في "رقع" (Patches)
    ديناميكية الطول بناءً على التعقيد المحلي للنص (Entropy-based)،
    مقاسةً بواسطة نموذج لغوي صغير يعمل على مستوى البايت.
    """
    def __init__(self, byte_lm, entropy_threshold: float = 1.5):
        self.byte_lm = byte_lm  # نموذج صغير يتنبأ بالبايت التالي
        self.entropy_threshold = entropy_threshold

    def compute_byte_entropy(self, byte_sequence: bytes) -> list:
        """حساب إنتروبيا التنبؤ بكل بايت بناءً على ما سبقه (عدم اليقين)."""
        entropies = []
        for i in range(1, len(byte_sequence)):
            context = byte_sequence[:i]
            next_byte_probs = self.byte_lm.predict_next_byte_distribution(context)
            entropy = -sum(p * torch.log(p + 1e-10) for p in next_byte_probs)
            entropies.append(entropy.item())
        return entropies

    def create_dynamic_patches(self, byte_sequence: bytes) -> list:
        """
        إنشاء رقع ديناميكية: حد رقعة جديدة يُفتَح عند نقاط عدم اليقين
        العالي (إنتروبيا مرتفعة = بداية كلمة/مفهوم جديد محتمل)،
        بينما تُجمَّع البايتات شديدة القابلية للتنبؤ (مثل نهايات الكلمات
        الشائعة) في رقعة واحدة أطول. هذا يُنتج ضغطًا تكيفيًا: نص بسيط
        متكرر → رقع طويلة (حوسبة أقل)، نص معقد/نادر → رقع قصيرة (حوسبة أدق).
        """
        entropies = self.compute_byte_entropy(byte_sequence)
        patches = []
        current_patch_start = 0

        for i, entropy in enumerate(entropies):
            if entropy > self.entropy_threshold:
                patches.append(byte_sequence[current_patch_start:i+1])
                current_patch_start = i + 1

        if current_patch_start < len(byte_sequence):
            patches.append(byte_sequence[current_patch_start:])

        return patches

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class AdaptiveHybridBlock(nn.Module):
    """
    معمارية هجينة متقدمة (بأسلوب Hymba/Zamba 2024): بدلاً من تبديل طبقات
    Attention وMamba بالتناوب كما في Jamba البسيط، تُدمَج مساراتهما
    بالتوازي (Parallel Heads) داخل نفس الطبقة، مع دمج مُرجَّح يسمح
    للنموذج بتعلّم متى يعتمد على كل مسار (ذاكرة دقيقة قصيرة المدى عبر
    Attention مقابل ملخّص مُضغوط طويل المدى عبر SSM).
    """
    def __init__(self, d_model: int, n_attn_heads: int, ssm_state_dim: int):
        super().__init__()
        # مسار Attention الدقيق (يحتفظ بتفاصيل كاملة - محدود بطول نافذة محلية)
        self.local_attention = nn.MultiheadAttention(
            d_model, n_attn_heads, batch_first=True
        )
        self.local_window_size = 256  # نافذة محلية فقط - اقتصاد حسابي

        # مسار SSM (يحتفظ بملخص مُضغوط للسياق الكامل بلا حدود عملية)
        from_chapter_16_mamba_block = None  # راجع SimplifiedMambaBlock (الفصل 16)
        self.ssm_path = SimplifiedMambaBlock(d_model, state_dim=ssm_state_dim)

        # بوابة دمج مُتعلَّمة (Learned Fusion Gate)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # معالجة متوازية (وليست متسلسلة) للمسارين
        local_out = self._windowed_attention(x)
        ssm_out = self.ssm_path(x)

        # بوابة تُقرر نسبة المزج بين المسارين لكل رمز ديناميكيًا
        gate_input = torch.cat([local_out, ssm_out], dim=-1)
        gate = self.fusion_gate(gate_input)
        fused = gate * local_out + (1 - gate) * ssm_out

        return self.norm(x + fused)

    def _windowed_attention(self, x):
        # تطبيق Attention ضمن نافذة محلية منزلقة فقط (اقتصاد O(n·w) بدل O(n²))
        # تفاصيل Sliding Window Attention غُطيت ضمنيًا في الفصل 14
        return x  # تبسيط توضيحي

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-62-ch62-post-transformer.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
