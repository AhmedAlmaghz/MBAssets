#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الرؤية واللغة: VLMs
Mastering Transformers and Large Language Models — Vision Language Models

Book slug : mastering-llms-transformers
Chapter   : ch-37-ch37-vision-language-models (lesson 37)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-37-ch37-vision-language-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الرؤية واللغة: VLMs
Mastering Transformers and Large Language Models — Vision Language Models

Book slug : mastering-llms-transformers
Chapter   : ch-37-ch37-vision-language-models (lesson 37)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-37-ch37-vision-language-models.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الرؤية واللغة: VLMs
Mastering Transformers and Large Language Models — Vision Language Models

Book slug : mastering-llms-transformers
Chapter   : ch-37-ch37-vision-language-models (lesson 37)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-1
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class VisionLanguageConnector(nn.Module):
    """
    طبقة الإسقاط (Projector) — القلب النابض لعائلة LLaVA.
    تحوّل visual features (من CLIP) إلى نفس فضاء embeddings النص.
    """
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()
        if projector_type == "linear":
            # أبسط تصميم (LLaVA-1.0)
            self.proj = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projector_type == "mlp2x_gelu":
            # تصميم LLaVA-1.5 — تحسّن ملحوظ بتكلفة زهيدة
            self.proj = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            raise ValueError(f"نوع غير مدعوم: {projector_type}")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: [batch, num_patches, vision_hidden_size]
        return self.proj(visual_features)  # -> [batch, num_patches, llm_hidden_size]

class SimpleLLaVA(nn.Module):
    """
    تطبيق تعليمي مبسّط لمعمارية LLaVA الكاملة.
    """
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14-336",
                 llm_model_name="lmsys/vicuna-7b-v1.5",
                 freeze_vision=True, use_lora_llm=True):
        super().__init__()

        # 1) مُرمِّز الرؤية — يبقى مجمّدًا في أغلب التصاميم
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # 2) طبقة الربط
        vision_hidden = self.vision_encoder.config.hidden_size   # 1024 لـ ViT-L

        # 3) النموذج اللغوي
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_hidden = self.llm.config.hidden_size

        self.connector = VisionLanguageConnector(vision_hidden, llm_hidden)

        # رمز خاص يمثّل مكان إدراج الصورة في النص
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        if use_lora_llm:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=128, lora_alpha=256,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05, task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """استخراج ملامح الصورة وإسقاطها إلى فضاء اللغة"""
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            vision_outputs = self.vision_encoder(pixel_values)
        # نتجاهل CLS token، نأخذ patch tokens فقط (مثل LLaVA)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        return self.connector(patch_features)   # [batch, 576, llm_hidden] لصورة 336x336

    def merge_text_and_image_embeds(self, input_ids, pixel_values, attention_mask):
        """
        الخطوة الأهم: استبدال موضع <image> في النص بـ 576 visual token
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)  # [B, 576, H]

        batch_size = input_ids.shape[0]
        merged_embeds_list = []
        merged_mask_list = []

        for b in range(batch_size):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged_embeds_list.append(text_embeds[b])
                merged_mask_list.append(attention_mask[b])
                continue
            pos = img_pos[0].item()
            merged = torch.cat([
                text_embeds[b, :pos],
                image_embeds[b],
                text_embeds[b, pos+1:]
            ], dim=0)
            new_mask = torch.cat([
                attention_mask[b, :pos],
                torch.ones(image_embeds.shape[1], device=attention_mask.device),
                attention_mask[b, pos+1:]
            ], dim=0)
            merged_embeds_list.append(merged)
            merged_mask_list.append(new_mask)

        # padding للدفعة لتوحيد الطول
        max_len = max(e.shape[0] for e in merged_embeds_list)
        padded_embeds = torch.stack([
            torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0]))
            for e in merged_embeds_list
        ])
        padded_mask = torch.stack([
            torch.nn.functional.pad(m, (0, max_len - m.shape[0]))
            for m in merged_mask_list
        ])
        return padded_embeds, padded_mask

    def forward(self, input_ids, pixel_values, attention_mask, labels=None):
        inputs_embeds, merged_mask = self.merge_text_and_image_embeds(
            input_ids, pixel_values, attention_mask
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            labels=labels   # يُحسب فقط على أجزاء النص، visual tokens تُقنَّع بـ -100
        )
        return outputs

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class GatedCrossAttentionBlock(nn.Module):
    """
    نمط Flamingo: يُدخَل بين طبقات LLM المجمَّدة بالكامل
    بوابة tanh تبدأ من الصفر لضمان عدم إفساد النموذج المُدرَّب مسبقًا في البداية
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_gate = nn.Parameter(torch.zeros(1))   # tanh-gating — البداية = صفر تأثير!

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, text_hidden, visual_features):
        attn_out, _ = self.cross_attn(
            query=self.norm1(text_hidden), key=visual_features, value=visual_features
        )
        text_hidden = text_hidden + torch.tanh(self.attn_gate) * attn_out
        ff_out = self.ff(self.norm2(text_hidden))
        text_hidden = text_hidden + torch.tanh(self.ff_gate) * ff_out
        return text_hidden

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-1
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class VisionLanguageConnector(nn.Module):
    """
    طبقة الإسقاط (Projector) — القلب النابض لعائلة LLaVA.
    تحوّل visual features (من CLIP) إلى نفس فضاء embeddings النص.
    """
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()
        if projector_type == "linear":
            # أبسط تصميم (LLaVA-1.0)
            self.proj = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projector_type == "mlp2x_gelu":
            # تصميم LLaVA-1.5 — تحسّن ملحوظ بتكلفة زهيدة
            self.proj = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            raise ValueError(f"نوع غير مدعوم: {projector_type}")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: [batch, num_patches, vision_hidden_size]
        return self.proj(visual_features)  # -> [batch, num_patches, llm_hidden_size]

class SimpleLLaVA(nn.Module):
    """
    تطبيق تعليمي مبسّط لمعمارية LLaVA الكاملة.
    """
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14-336",
                 llm_model_name="lmsys/vicuna-7b-v1.5",
                 freeze_vision=True, use_lora_llm=True):
        super().__init__()

        # 1) مُرمِّز الرؤية — يبقى مجمّدًا في أغلب التصاميم
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # 2) طبقة الربط
        vision_hidden = self.vision_encoder.config.hidden_size   # 1024 لـ ViT-L

        # 3) النموذج اللغوي
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_hidden = self.llm.config.hidden_size

        self.connector = VisionLanguageConnector(vision_hidden, llm_hidden)

        # رمز خاص يمثّل مكان إدراج الصورة في النص
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        if use_lora_llm:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=128, lora_alpha=256,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05, task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """استخراج ملامح الصورة وإسقاطها إلى فضاء اللغة"""
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            vision_outputs = self.vision_encoder(pixel_values)
        # نتجاهل CLS token، نأخذ patch tokens فقط (مثل LLaVA)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        return self.connector(patch_features)   # [batch, 576, llm_hidden] لصورة 336x336

    def merge_text_and_image_embeds(self, input_ids, pixel_values, attention_mask):
        """
        الخطوة الأهم: استبدال موضع <image> في النص بـ 576 visual token
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)  # [B, 576, H]

        batch_size = input_ids.shape[0]
        merged_embeds_list = []
        merged_mask_list = []

        for b in range(batch_size):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged_embeds_list.append(text_embeds[b])
                merged_mask_list.append(attention_mask[b])
                continue
            pos = img_pos[0].item()
            merged = torch.cat([
                text_embeds[b, :pos],
                image_embeds[b],
                text_embeds[b, pos+1:]
            ], dim=0)
            new_mask = torch.cat([
                attention_mask[b, :pos],
                torch.ones(image_embeds.shape[1], device=attention_mask.device),
                attention_mask[b, pos+1:]
            ], dim=0)
            merged_embeds_list.append(merged)
            merged_mask_list.append(new_mask)

        # padding للدفعة لتوحيد الطول
        max_len = max(e.shape[0] for e in merged_embeds_list)
        padded_embeds = torch.stack([
            torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0]))
            for e in merged_embeds_list
        ])
        padded_mask = torch.stack([
            torch.nn.functional.pad(m, (0, max_len - m.shape[0]))
            for m in merged_mask_list
        ])
        return padded_embeds, padded_mask

    def forward(self, input_ids, pixel_values, attention_mask, labels=None):
        inputs_embeds, merged_mask = self.merge_text_and_image_embeds(
            input_ids, pixel_values, attention_mask
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            labels=labels   # يُحسب فقط على أجزاء النص، visual tokens تُقنَّع بـ -100
        )
        return outputs

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class GatedCrossAttentionBlock(nn.Module):
    """
    نمط Flamingo: يُدخَل بين طبقات LLM المجمَّدة بالكامل
    بوابة tanh تبدأ من الصفر لضمان عدم إفساد النموذج المُدرَّب مسبقًا في البداية
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_gate = nn.Parameter(torch.zeros(1))   # tanh-gating — البداية = صفر تأثير!

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, text_hidden, visual_features):
        attn_out, _ = self.cross_attn(
            query=self.norm1(text_hidden), key=visual_features, value=visual_features
        )
        text_hidden = text_hidden + torch.tanh(self.attn_gate) * attn_out
        ff_out = self.ff(self.norm2(text_hidden))
        text_hidden = text_hidden + torch.tanh(self.ff_gate) * ff_out
        return text_hidden

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — نماذج الرؤية واللغة: VLMs
Mastering Transformers and Large Language Models — Vision Language Models

Book slug : mastering-llms-transformers
Chapter   : ch-37-ch37-vision-language-models (lesson 37)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-1
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class VisionLanguageConnector(nn.Module):
    """
    طبقة الإسقاط (Projector) — القلب النابض لعائلة LLaVA.
    تحوّل visual features (من CLIP) إلى نفس فضاء embeddings النص.
    """
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()
        if projector_type == "linear":
            # أبسط تصميم (LLaVA-1.0)
            self.proj = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projector_type == "mlp2x_gelu":
            # تصميم LLaVA-1.5 — تحسّن ملحوظ بتكلفة زهيدة
            self.proj = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            raise ValueError(f"نوع غير مدعوم: {projector_type}")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: [batch, num_patches, vision_hidden_size]
        return self.proj(visual_features)  # -> [batch, num_patches, llm_hidden_size]

class SimpleLLaVA(nn.Module):
    """
    تطبيق تعليمي مبسّط لمعمارية LLaVA الكاملة.
    """
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14-336",
                 llm_model_name="lmsys/vicuna-7b-v1.5",
                 freeze_vision=True, use_lora_llm=True):
        super().__init__()

        # 1) مُرمِّز الرؤية — يبقى مجمّدًا في أغلب التصاميم
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # 2) طبقة الربط
        vision_hidden = self.vision_encoder.config.hidden_size   # 1024 لـ ViT-L

        # 3) النموذج اللغوي
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_hidden = self.llm.config.hidden_size

        self.connector = VisionLanguageConnector(vision_hidden, llm_hidden)

        # رمز خاص يمثّل مكان إدراج الصورة في النص
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        if use_lora_llm:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=128, lora_alpha=256,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05, task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """استخراج ملامح الصورة وإسقاطها إلى فضاء اللغة"""
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            vision_outputs = self.vision_encoder(pixel_values)
        # نتجاهل CLS token، نأخذ patch tokens فقط (مثل LLaVA)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        return self.connector(patch_features)   # [batch, 576, llm_hidden] لصورة 336x336

    def merge_text_and_image_embeds(self, input_ids, pixel_values, attention_mask):
        """
        الخطوة الأهم: استبدال موضع <image> في النص بـ 576 visual token
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)  # [B, 576, H]

        batch_size = input_ids.shape[0]
        merged_embeds_list = []
        merged_mask_list = []

        for b in range(batch_size):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged_embeds_list.append(text_embeds[b])
                merged_mask_list.append(attention_mask[b])
                continue
            pos = img_pos[0].item()
            merged = torch.cat([
                text_embeds[b, :pos],
                image_embeds[b],
                text_embeds[b, pos+1:]
            ], dim=0)
            new_mask = torch.cat([
                attention_mask[b, :pos],
                torch.ones(image_embeds.shape[1], device=attention_mask.device),
                attention_mask[b, pos+1:]
            ], dim=0)
            merged_embeds_list.append(merged)
            merged_mask_list.append(new_mask)

        # padding للدفعة لتوحيد الطول
        max_len = max(e.shape[0] for e in merged_embeds_list)
        padded_embeds = torch.stack([
            torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0]))
            for e in merged_embeds_list
        ])
        padded_mask = torch.stack([
            torch.nn.functional.pad(m, (0, max_len - m.shape[0]))
            for m in merged_mask_list
        ])
        return padded_embeds, padded_mask

    def forward(self, input_ids, pixel_values, attention_mask, labels=None):
        inputs_embeds, merged_mask = self.merge_text_and_image_embeds(
            input_ids, pixel_values, attention_mask
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            labels=labels   # يُحسب فقط على أجزاء النص، visual tokens تُقنَّع بـ -100
        )
        return outputs

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class GatedCrossAttentionBlock(nn.Module):
    """
    نمط Flamingo: يُدخَل بين طبقات LLM المجمَّدة بالكامل
    بوابة tanh تبدأ من الصفر لضمان عدم إفساد النموذج المُدرَّب مسبقًا في البداية
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_gate = nn.Parameter(torch.zeros(1))   # tanh-gating — البداية = صفر تأثير!

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, text_hidden, visual_features):
        attn_out, _ = self.cross_attn(
            query=self.norm1(text_hidden), key=visual_features, value=visual_features
        )
        text_hidden = text_hidden + torch.tanh(self.attn_gate) * attn_out
        ff_out = self.ff(self.norm2(text_hidden))
        text_hidden = text_hidden + torch.tanh(self.ff_gate) * ff_out
        return text_hidden

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch37-vision-language-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-2
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
    ("peft", "peft"),
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class VisionLanguageConnector(nn.Module):
    """
    طبقة الإسقاط (Projector) — القلب النابض لعائلة LLaVA.
    تحوّل visual features (من CLIP) إلى نفس فضاء embeddings النص.
    """
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()
        if projector_type == "linear":
            # أبسط تصميم (LLaVA-1.0)
            self.proj = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projector_type == "mlp2x_gelu":
            # تصميم LLaVA-1.5 — تحسّن ملحوظ بتكلفة زهيدة
            self.proj = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            raise ValueError(f"نوع غير مدعوم: {projector_type}")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: [batch, num_patches, vision_hidden_size]
        return self.proj(visual_features)  # -> [batch, num_patches, llm_hidden_size]

class SimpleLLaVA(nn.Module):
    """
    تطبيق تعليمي مبسّط لمعمارية LLaVA الكاملة.
    """
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14-336",
                 llm_model_name="lmsys/vicuna-7b-v1.5",
                 freeze_vision=True, use_lora_llm=True):
        super().__init__()

        # 1) مُرمِّز الرؤية — يبقى مجمّدًا في أغلب التصاميم
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # 2) طبقة الربط
        vision_hidden = self.vision_encoder.config.hidden_size   # 1024 لـ ViT-L

        # 3) النموذج اللغوي
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_hidden = self.llm.config.hidden_size

        self.connector = VisionLanguageConnector(vision_hidden, llm_hidden)

        # رمز خاص يمثّل مكان إدراج الصورة في النص
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        if use_lora_llm:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=128, lora_alpha=256,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05, task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """استخراج ملامح الصورة وإسقاطها إلى فضاء اللغة"""
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            vision_outputs = self.vision_encoder(pixel_values)
        # نتجاهل CLS token، نأخذ patch tokens فقط (مثل LLaVA)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        return self.connector(patch_features)   # [batch, 576, llm_hidden] لصورة 336x336

    def merge_text_and_image_embeds(self, input_ids, pixel_values, attention_mask):
        """
        الخطوة الأهم: استبدال موضع <image> في النص بـ 576 visual token
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_embeds = self.encode_image(pixel_values)  # [B, 576, H]

        batch_size = input_ids.shape[0]
        merged_embeds_list = []
        merged_mask_list = []

        for b in range(batch_size):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged_embeds_list.append(text_embeds[b])
                merged_mask_list.append(attention_mask[b])
                continue
            pos = img_pos[0].item()
            merged = torch.cat([
                text_embeds[b, :pos],
                image_embeds[b],
                text_embeds[b, pos+1:]
            ], dim=0)
            new_mask = torch.cat([
                attention_mask[b, :pos],
                torch.ones(image_embeds.shape[1], device=attention_mask.device),
                attention_mask[b, pos+1:]
            ], dim=0)
            merged_embeds_list.append(merged)
            merged_mask_list.append(new_mask)

        # padding للدفعة لتوحيد الطول
        max_len = max(e.shape[0] for e in merged_embeds_list)
        padded_embeds = torch.stack([
            torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0]))
            for e in merged_embeds_list
        ])
        padded_mask = torch.stack([
            torch.nn.functional.pad(m, (0, max_len - m.shape[0]))
            for m in merged_mask_list
        ])
        return padded_embeds, padded_mask

    def forward(self, input_ids, pixel_values, attention_mask, labels=None):
        inputs_embeds, merged_mask = self.merge_text_and_image_embeds(
            input_ids, pixel_values, attention_mask
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            labels=labels   # يُحسب فقط على أجزاء النص، visual tokens تُقنَّع بـ -100
        )
        return outputs

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class GatedCrossAttentionBlock(nn.Module):
    """
    نمط Flamingo: يُدخَل بين طبقات LLM المجمَّدة بالكامل
    بوابة tanh تبدأ من الصفر لضمان عدم إفساد النموذج المُدرَّب مسبقًا في البداية
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_gate = nn.Parameter(torch.zeros(1))   # tanh-gating — البداية = صفر تأثير!

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, text_hidden, visual_features):
        attn_out, _ = self.cross_attn(
            query=self.norm1(text_hidden), key=visual_features, value=visual_features
        )
        text_hidden = text_hidden + torch.tanh(self.attn_gate) * attn_out
        ff_out = self.ff(self.norm2(text_hidden))
        text_hidden = text_hidden + torch.tanh(self.ff_gate) * ff_out
        return text_hidden

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-37-ch37-vision-language-models.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
