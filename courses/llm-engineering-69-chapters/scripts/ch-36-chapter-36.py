#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التحسين المباشر للتفضيلات (DPO)
Large Language Model Engineering — DPO

Book slug : llm-engineering-69-chapters
Chapter   : ch-36-chapter-36 (lesson 36)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-36-chapter-36.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التحسين المباشر للتفضيلات (DPO)
Large Language Model Engineering — DPO

Book slug : llm-engineering-69-chapters
Chapter   : ch-36-chapter-36 (lesson 36)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-36-chapter-36.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التحسين المباشر للتفضيلات (DPO)
Large Language Model Engineering — DPO

Book slug : llm-engineering-69-chapters
Chapter   : ch-36-chapter-36 (lesson 36)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_36.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 36: الفصل 36: التحسين المباشر للتفضيلات (DPO)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_dpo() -> None:
    chosen_margin, rejected_margin, beta = (1.2, -0.3, 0.1)
    logit = beta * (chosen_margin - rejected_margin)
    loss = math.log1p(math.exp(-logit))
    print('preference logit:', round(logit, 4))
    print('DPO logistic loss:', round(loss, 4))
    print('larger chosen-vs-rejected margin lowers loss.')

if __name__ == "__main__":
    print("Chapter 36 lab: Compute a stable DPO preference loss")
    run_dpo()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-36.md
# ────────────────────────────────────────────────────────────────────────
"""
بناء DPO من الصفر لفهم الآلية الداخلية بالكامل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """
    يحسب مجموع log-probabilities للتوكنز الفعلية في الاستجابة فقط
    (نتجاهل توكنز الـ prompt عبر labels_mask).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]           # إزاحة بمقدار واحد (توقّع التوكن التالي)
    targets = input_ids[:, 1:]
    mask = labels_mask[:, 1:]                     # نفس الإزاحة لقناع التسميات

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # مجموع log-probs فقط على توكنز الاستجابة (ليس الـ prompt)
    sequence_log_prob = (token_log_probs * mask).sum(dim=-1)
    return sequence_log_prob


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة DPO الكاملة كما اشتققناها رياضيًا:

    L = -log(sigmoid(β * [(log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))]))
    """
    # الفرق بين السياسة والـ Reference لكل من الاستجابة المفضَّلة والمرفوضة
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps

    # هذا هو المقياس الداخلي لمعادلة DPO: الفرق بين الفروق
    logits = beta * (policy_log_ratio - reference_log_ratio)

    loss = -F.logsigmoid(logits).mean()

    # إحصائيات مفيدة للمراقبة أثناء التدريب
    with torch.no_grad():
        implicit_reward_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
        implicit_reward_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
        reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

    stats = {
        "loss": loss.item(),
        "reward_accuracy": reward_accuracy.item(),
        "reward_margin": reward_margin.item(),
        "implicit_reward_chosen_mean": implicit_reward_chosen.mean().item(),
        "implicit_reward_rejected_mean": implicit_reward_rejected.mean().item(),
    }

    return loss, stats

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-36.md
# ────────────────────────────────────────────────────────────────────────
"""
حلقة تدريب DPO كاملة — نموذجان فقط، بلا PPO وبلا Reward Model منفصل
"""

import copy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset(Dataset):
    def __init__(self, examples: list[DPOExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _tokenize_with_mask(self, prompt: str, response: str) -> dict:
        """يُرجع input_ids كاملة + قناع يُفعَّل فقط على توكنز الاستجابة (لا الـ prompt)."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        input_ids = input_ids[: self.max_length]
        labels_mask = labels_mask[: self.max_length]

        padding_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        labels_mask = labels_mask + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels_mask": torch.tensor(labels_mask),
        }

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        chosen_enc = self._tokenize_with_mask(ex.prompt, ex.chosen)
        rejected_enc = self._tokenize_with_mask(ex.prompt, ex.rejected)
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels_mask": chosen_enc["labels_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels_mask": rejected_enc["labels_mask"],
        }


def train_dpo(
    model_name: str,
    train_examples: list[DPOExample],
    beta: float = 0.1,
    num_epochs: int = 3,
    learning_rate: float = 5e-7,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # السياسة القابلة للتدريب (نبدأ من نموذج SFT)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy_model.train()

    # Reference Model: نسخة مجمَّدة تمامًا — لاحظ: نموذجان فقط، لا أربعة!
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    dataset = DPODataset(train_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_stats = {"loss": [], "reward_accuracy": [], "reward_margin": []}

        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels_mask = batch["chosen_labels_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels_mask = batch["rejected_labels_mask"].to(device)

            # log-probs من السياسة الحالية (قابلة للتدريب)
            policy_chosen_logps = compute_sequence_log_probs(
                policy_model, chosen_ids, chosen_mask, chosen_labels_mask
            )
            policy_rejected_logps = compute_sequence_log_probs(
                policy_model, rejected_ids, rejected_mask, rejected_labels_mask
            )

            # log-probs من Reference Model (بدون تدرّجات)
            with torch.no_grad():
                reference_chosen_logps = compute_sequence_log_probs(
                    reference_model, chosen_ids, chosen_mask, chosen_labels_mask
                )
                reference_rejected_logps = compute_sequence_log_probs(
                    reference_model, rejected_ids, rejected_mask, rejected_labels_mask
                )

            loss, stats = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_stats:
                epoch_stats[k].append(stats[k])

        avg_loss = sum(epoch_stats["loss"]) / len(epoch_stats["loss"])
        avg_acc = sum(epoch_stats["reward_accuracy"]) / len(epoch_stats["reward_accuracy"])
        avg_margin = sum(epoch_stats["reward_margin"]) / len(epoch_stats["reward_margin"])
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Reward Accuracy={avg_acc:.2%}, Reward Margin={avg_margin:.4f}")

    return policy_model

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-36.md
# ────────────────────────────────────────────────────────────────────────
"""
استخدام DPOTrainer من مكتبة trl للتطبيق الإنتاجي المُحسَّن
pip install trl transformers datasets peft
"""

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig  # يمكن دمج DPO مع LoRA مباشرة لكفاءة أكبر!


MODEL_NAME = "./sft-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# ملاحظة: إذا لم تُمرَّر ref_model، تستنتج المكتبة تلقائيًا نسخة مجمَّدة من نفس النموذج
# أو تستخدم نفس النموذج مع تعطيل محولات LoRA (إذا استُخدم PEFT) لتوفير الذاكرة

# تنسيق البيانات المطلوب من DPOTrainer: أعمدة prompt, chosen, rejected
raw_data = [
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف لبناء أنظمة قادرة على أداء مهام تتطلب عادة ذكاءً بشريًا، مثل التعلّم واتخاذ القرار وفهم اللغة.",
        "rejected": "الذكاء الاصطناعي شيء معقد جدًا ولا أحد يفهمه بشكل كامل، إنه مستقبل كل شيء تقريبًا.",
    },
    {
        "prompt": "كيف أتعلم البرمجة من الصفر؟",
        "chosen": "ابدأ بلغة سهلة مثل Python، تعلّم الأساسيات (المتغيرات، الحلقات، الدوال)، ثم مارس عبر مشاريع صغيرة تدريجية الصعوبة.",
        "rejected": "البرمجة صعبة جدًا، تحتاج سنوات طويلة، ومعظم الناس يفشلون فيها، لذا فكّر جيدًا قبل البدء.",
    },
    # في مشروع حقيقي: آلاف الأمثلة من بيانات تفضيل حقيقية
]

train_dataset = Dataset.from_list(raw_data)

dpo_config = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,                        # β في معادلة DPO — يتحكّم بقوة الانحراف المسموح
    learning_rate=5e-7,              # DPO يحتاج معدلات تعلّم منخفضة جدًا، أقل حتى من RLHF
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    max_length=512,
    max_prompt_length=256,
    logging_steps=1,
    report_to="none",
)

# يمكن دمج LoRA مباشرة — "QDPO" فعليًا، كفاءة ذاكرة مذهلة
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model=policy_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,  # إضافة هذا السطر وحده تُحوّل DPO العادي إلى نسخة فعّالة بالذاكرة
)

dpo_trainer.train()
dpo_trainer.save_model("./dpo-model-final")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 36: الفصل 36: التحسين المباشر للتفضيلات (DPO)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_dpo() -> None:
    chosen_margin, rejected_margin, beta = (1.2, -0.3, 0.1)
    logit = beta * (chosen_margin - rejected_margin)
    loss = math.log1p(math.exp(-logit))
    print('preference logit:', round(logit, 4))
    print('DPO logistic loss:', round(loss, 4))
    print('larger chosen-vs-rejected margin lowers loss.')

if __name__ == "__main__":
    print("Chapter 36 lab: Compute a stable DPO preference loss")
    run_dpo()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
بناء DPO من الصفر لفهم الآلية الداخلية بالكامل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """
    يحسب مجموع log-probabilities للتوكنز الفعلية في الاستجابة فقط
    (نتجاهل توكنز الـ prompt عبر labels_mask).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]           # إزاحة بمقدار واحد (توقّع التوكن التالي)
    targets = input_ids[:, 1:]
    mask = labels_mask[:, 1:]                     # نفس الإزاحة لقناع التسميات

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # مجموع log-probs فقط على توكنز الاستجابة (ليس الـ prompt)
    sequence_log_prob = (token_log_probs * mask).sum(dim=-1)
    return sequence_log_prob


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة DPO الكاملة كما اشتققناها رياضيًا:

    L = -log(sigmoid(β * [(log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))]))
    """
    # الفرق بين السياسة والـ Reference لكل من الاستجابة المفضَّلة والمرفوضة
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps

    # هذا هو المقياس الداخلي لمعادلة DPO: الفرق بين الفروق
    logits = beta * (policy_log_ratio - reference_log_ratio)

    loss = -F.logsigmoid(logits).mean()

    # إحصائيات مفيدة للمراقبة أثناء التدريب
    with torch.no_grad():
        implicit_reward_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
        implicit_reward_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
        reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

    stats = {
        "loss": loss.item(),
        "reward_accuracy": reward_accuracy.item(),
        "reward_margin": reward_margin.item(),
        "implicit_reward_chosen_mean": implicit_reward_chosen.mean().item(),
        "implicit_reward_rejected_mean": implicit_reward_rejected.mean().item(),
    }

    return loss, stats

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
حلقة تدريب DPO كاملة — نموذجان فقط، بلا PPO وبلا Reward Model منفصل
"""

import copy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset(Dataset):
    def __init__(self, examples: list[DPOExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _tokenize_with_mask(self, prompt: str, response: str) -> dict:
        """يُرجع input_ids كاملة + قناع يُفعَّل فقط على توكنز الاستجابة (لا الـ prompt)."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        input_ids = input_ids[: self.max_length]
        labels_mask = labels_mask[: self.max_length]

        padding_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        labels_mask = labels_mask + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels_mask": torch.tensor(labels_mask),
        }

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        chosen_enc = self._tokenize_with_mask(ex.prompt, ex.chosen)
        rejected_enc = self._tokenize_with_mask(ex.prompt, ex.rejected)
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels_mask": chosen_enc["labels_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels_mask": rejected_enc["labels_mask"],
        }


def train_dpo(
    model_name: str,
    train_examples: list[DPOExample],
    beta: float = 0.1,
    num_epochs: int = 3,
    learning_rate: float = 5e-7,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # السياسة القابلة للتدريب (نبدأ من نموذج SFT)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy_model.train()

    # Reference Model: نسخة مجمَّدة تمامًا — لاحظ: نموذجان فقط، لا أربعة!
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    dataset = DPODataset(train_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_stats = {"loss": [], "reward_accuracy": [], "reward_margin": []}

        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels_mask = batch["chosen_labels_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels_mask = batch["rejected_labels_mask"].to(device)

            # log-probs من السياسة الحالية (قابلة للتدريب)
            policy_chosen_logps = compute_sequence_log_probs(
                policy_model, chosen_ids, chosen_mask, chosen_labels_mask
            )
            policy_rejected_logps = compute_sequence_log_probs(
                policy_model, rejected_ids, rejected_mask, rejected_labels_mask
            )

            # log-probs من Reference Model (بدون تدرّجات)
            with torch.no_grad():
                reference_chosen_logps = compute_sequence_log_probs(
                    reference_model, chosen_ids, chosen_mask, chosen_labels_mask
                )
                reference_rejected_logps = compute_sequence_log_probs(
                    reference_model, rejected_ids, rejected_mask, rejected_labels_mask
                )

            loss, stats = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_stats:
                epoch_stats[k].append(stats[k])

        avg_loss = sum(epoch_stats["loss"]) / len(epoch_stats["loss"])
        avg_acc = sum(epoch_stats["reward_accuracy"]) / len(epoch_stats["reward_accuracy"])
        avg_margin = sum(epoch_stats["reward_margin"]) / len(epoch_stats["reward_margin"])
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Reward Accuracy={avg_acc:.2%}, Reward Margin={avg_margin:.4f}")

    return policy_model

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
استخدام DPOTrainer من مكتبة trl للتطبيق الإنتاجي المُحسَّن
pip install trl transformers datasets peft
"""

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig  # يمكن دمج DPO مع LoRA مباشرة لكفاءة أكبر!


MODEL_NAME = "./sft-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# ملاحظة: إذا لم تُمرَّر ref_model، تستنتج المكتبة تلقائيًا نسخة مجمَّدة من نفس النموذج
# أو تستخدم نفس النموذج مع تعطيل محولات LoRA (إذا استُخدم PEFT) لتوفير الذاكرة

# تنسيق البيانات المطلوب من DPOTrainer: أعمدة prompt, chosen, rejected
raw_data = [
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف لبناء أنظمة قادرة على أداء مهام تتطلب عادة ذكاءً بشريًا، مثل التعلّم واتخاذ القرار وفهم اللغة.",
        "rejected": "الذكاء الاصطناعي شيء معقد جدًا ولا أحد يفهمه بشكل كامل، إنه مستقبل كل شيء تقريبًا.",
    },
    {
        "prompt": "كيف أتعلم البرمجة من الصفر؟",
        "chosen": "ابدأ بلغة سهلة مثل Python، تعلّم الأساسيات (المتغيرات، الحلقات، الدوال)، ثم مارس عبر مشاريع صغيرة تدريجية الصعوبة.",
        "rejected": "البرمجة صعبة جدًا، تحتاج سنوات طويلة، ومعظم الناس يفشلون فيها، لذا فكّر جيدًا قبل البدء.",
    },
    # في مشروع حقيقي: آلاف الأمثلة من بيانات تفضيل حقيقية
]

train_dataset = Dataset.from_list(raw_data)

dpo_config = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,                        # β في معادلة DPO — يتحكّم بقوة الانحراف المسموح
    learning_rate=5e-7,              # DPO يحتاج معدلات تعلّم منخفضة جدًا، أقل حتى من RLHF
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    max_length=512,
    max_prompt_length=256,
    logging_steps=1,
    report_to="none",
)

# يمكن دمج LoRA مباشرة — "QDPO" فعليًا، كفاءة ذاكرة مذهلة
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model=policy_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,  # إضافة هذا السطر وحده تُحوّل DPO العادي إلى نسخة فعّالة بالذاكرة
)

dpo_trainer.train()
dpo_trainer.save_model("./dpo-model-final")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التحسين المباشر للتفضيلات (DPO)
Large Language Model Engineering — DPO

Book slug : llm-engineering-69-chapters
Chapter   : ch-36-chapter-36 (lesson 36)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_36.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 36: الفصل 36: التحسين المباشر للتفضيلات (DPO)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_dpo() -> None:
    chosen_margin, rejected_margin, beta = (1.2, -0.3, 0.1)
    logit = beta * (chosen_margin - rejected_margin)
    loss = math.log1p(math.exp(-logit))
    print('preference logit:', round(logit, 4))
    print('DPO logistic loss:', round(loss, 4))
    print('larger chosen-vs-rejected margin lowers loss.')

if __name__ == "__main__":
    print("Chapter 36 lab: Compute a stable DPO preference loss")
    run_dpo()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-36.md
# ────────────────────────────────────────────────────────────────────────
"""
بناء DPO من الصفر لفهم الآلية الداخلية بالكامل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """
    يحسب مجموع log-probabilities للتوكنز الفعلية في الاستجابة فقط
    (نتجاهل توكنز الـ prompt عبر labels_mask).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]           # إزاحة بمقدار واحد (توقّع التوكن التالي)
    targets = input_ids[:, 1:]
    mask = labels_mask[:, 1:]                     # نفس الإزاحة لقناع التسميات

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # مجموع log-probs فقط على توكنز الاستجابة (ليس الـ prompt)
    sequence_log_prob = (token_log_probs * mask).sum(dim=-1)
    return sequence_log_prob


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة DPO الكاملة كما اشتققناها رياضيًا:

    L = -log(sigmoid(β * [(log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))]))
    """
    # الفرق بين السياسة والـ Reference لكل من الاستجابة المفضَّلة والمرفوضة
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps

    # هذا هو المقياس الداخلي لمعادلة DPO: الفرق بين الفروق
    logits = beta * (policy_log_ratio - reference_log_ratio)

    loss = -F.logsigmoid(logits).mean()

    # إحصائيات مفيدة للمراقبة أثناء التدريب
    with torch.no_grad():
        implicit_reward_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
        implicit_reward_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
        reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

    stats = {
        "loss": loss.item(),
        "reward_accuracy": reward_accuracy.item(),
        "reward_margin": reward_margin.item(),
        "implicit_reward_chosen_mean": implicit_reward_chosen.mean().item(),
        "implicit_reward_rejected_mean": implicit_reward_rejected.mean().item(),
    }

    return loss, stats

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-36.md
# ────────────────────────────────────────────────────────────────────────
"""
حلقة تدريب DPO كاملة — نموذجان فقط، بلا PPO وبلا Reward Model منفصل
"""

import copy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset(Dataset):
    def __init__(self, examples: list[DPOExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _tokenize_with_mask(self, prompt: str, response: str) -> dict:
        """يُرجع input_ids كاملة + قناع يُفعَّل فقط على توكنز الاستجابة (لا الـ prompt)."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        input_ids = input_ids[: self.max_length]
        labels_mask = labels_mask[: self.max_length]

        padding_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        labels_mask = labels_mask + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels_mask": torch.tensor(labels_mask),
        }

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        chosen_enc = self._tokenize_with_mask(ex.prompt, ex.chosen)
        rejected_enc = self._tokenize_with_mask(ex.prompt, ex.rejected)
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels_mask": chosen_enc["labels_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels_mask": rejected_enc["labels_mask"],
        }


def train_dpo(
    model_name: str,
    train_examples: list[DPOExample],
    beta: float = 0.1,
    num_epochs: int = 3,
    learning_rate: float = 5e-7,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # السياسة القابلة للتدريب (نبدأ من نموذج SFT)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy_model.train()

    # Reference Model: نسخة مجمَّدة تمامًا — لاحظ: نموذجان فقط، لا أربعة!
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    dataset = DPODataset(train_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_stats = {"loss": [], "reward_accuracy": [], "reward_margin": []}

        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels_mask = batch["chosen_labels_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels_mask = batch["rejected_labels_mask"].to(device)

            # log-probs من السياسة الحالية (قابلة للتدريب)
            policy_chosen_logps = compute_sequence_log_probs(
                policy_model, chosen_ids, chosen_mask, chosen_labels_mask
            )
            policy_rejected_logps = compute_sequence_log_probs(
                policy_model, rejected_ids, rejected_mask, rejected_labels_mask
            )

            # log-probs من Reference Model (بدون تدرّجات)
            with torch.no_grad():
                reference_chosen_logps = compute_sequence_log_probs(
                    reference_model, chosen_ids, chosen_mask, chosen_labels_mask
                )
                reference_rejected_logps = compute_sequence_log_probs(
                    reference_model, rejected_ids, rejected_mask, rejected_labels_mask
                )

            loss, stats = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_stats:
                epoch_stats[k].append(stats[k])

        avg_loss = sum(epoch_stats["loss"]) / len(epoch_stats["loss"])
        avg_acc = sum(epoch_stats["reward_accuracy"]) / len(epoch_stats["reward_accuracy"])
        avg_margin = sum(epoch_stats["reward_margin"]) / len(epoch_stats["reward_margin"])
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Reward Accuracy={avg_acc:.2%}, Reward Margin={avg_margin:.4f}")

    return policy_model

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-36.md
# ────────────────────────────────────────────────────────────────────────
"""
استخدام DPOTrainer من مكتبة trl للتطبيق الإنتاجي المُحسَّن
pip install trl transformers datasets peft
"""

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig  # يمكن دمج DPO مع LoRA مباشرة لكفاءة أكبر!


MODEL_NAME = "./sft-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# ملاحظة: إذا لم تُمرَّر ref_model، تستنتج المكتبة تلقائيًا نسخة مجمَّدة من نفس النموذج
# أو تستخدم نفس النموذج مع تعطيل محولات LoRA (إذا استُخدم PEFT) لتوفير الذاكرة

# تنسيق البيانات المطلوب من DPOTrainer: أعمدة prompt, chosen, rejected
raw_data = [
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف لبناء أنظمة قادرة على أداء مهام تتطلب عادة ذكاءً بشريًا، مثل التعلّم واتخاذ القرار وفهم اللغة.",
        "rejected": "الذكاء الاصطناعي شيء معقد جدًا ولا أحد يفهمه بشكل كامل، إنه مستقبل كل شيء تقريبًا.",
    },
    {
        "prompt": "كيف أتعلم البرمجة من الصفر؟",
        "chosen": "ابدأ بلغة سهلة مثل Python، تعلّم الأساسيات (المتغيرات، الحلقات، الدوال)، ثم مارس عبر مشاريع صغيرة تدريجية الصعوبة.",
        "rejected": "البرمجة صعبة جدًا، تحتاج سنوات طويلة، ومعظم الناس يفشلون فيها، لذا فكّر جيدًا قبل البدء.",
    },
    # في مشروع حقيقي: آلاف الأمثلة من بيانات تفضيل حقيقية
]

train_dataset = Dataset.from_list(raw_data)

dpo_config = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,                        # β في معادلة DPO — يتحكّم بقوة الانحراف المسموح
    learning_rate=5e-7,              # DPO يحتاج معدلات تعلّم منخفضة جدًا، أقل حتى من RLHF
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    max_length=512,
    max_prompt_length=256,
    logging_steps=1,
    report_to="none",
)

# يمكن دمج LoRA مباشرة — "QDPO" فعليًا، كفاءة ذاكرة مذهلة
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model=policy_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,  # إضافة هذا السطر وحده تُحوّل DPO العادي إلى نسخة فعّالة بالذاكرة
)

dpo_trainer.train()
dpo_trainer.save_model("./dpo-model-final")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 36: الفصل 36: التحسين المباشر للتفضيلات (DPO)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_dpo() -> None:
    chosen_margin, rejected_margin, beta = (1.2, -0.3, 0.1)
    logit = beta * (chosen_margin - rejected_margin)
    loss = math.log1p(math.exp(-logit))
    print('preference logit:', round(logit, 4))
    print('DPO logistic loss:', round(loss, 4))
    print('larger chosen-vs-rejected margin lowers loss.')

if __name__ == "__main__":
    print("Chapter 36 lab: Compute a stable DPO preference loss")
    run_dpo()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
بناء DPO من الصفر لفهم الآلية الداخلية بالكامل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """
    يحسب مجموع log-probabilities للتوكنز الفعلية في الاستجابة فقط
    (نتجاهل توكنز الـ prompt عبر labels_mask).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]           # إزاحة بمقدار واحد (توقّع التوكن التالي)
    targets = input_ids[:, 1:]
    mask = labels_mask[:, 1:]                     # نفس الإزاحة لقناع التسميات

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # مجموع log-probs فقط على توكنز الاستجابة (ليس الـ prompt)
    sequence_log_prob = (token_log_probs * mask).sum(dim=-1)
    return sequence_log_prob


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة DPO الكاملة كما اشتققناها رياضيًا:

    L = -log(sigmoid(β * [(log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))]))
    """
    # الفرق بين السياسة والـ Reference لكل من الاستجابة المفضَّلة والمرفوضة
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps

    # هذا هو المقياس الداخلي لمعادلة DPO: الفرق بين الفروق
    logits = beta * (policy_log_ratio - reference_log_ratio)

    loss = -F.logsigmoid(logits).mean()

    # إحصائيات مفيدة للمراقبة أثناء التدريب
    with torch.no_grad():
        implicit_reward_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
        implicit_reward_rejected = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()
        reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()

    stats = {
        "loss": loss.item(),
        "reward_accuracy": reward_accuracy.item(),
        "reward_margin": reward_margin.item(),
        "implicit_reward_chosen_mean": implicit_reward_chosen.mean().item(),
        "implicit_reward_rejected_mean": implicit_reward_rejected.mean().item(),
    }

    return loss, stats

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
حلقة تدريب DPO كاملة — نموذجان فقط، بلا PPO وبلا Reward Model منفصل
"""

import copy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset(Dataset):
    def __init__(self, examples: list[DPOExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _tokenize_with_mask(self, prompt: str, response: str) -> dict:
        """يُرجع input_ids كاملة + قناع يُفعَّل فقط على توكنز الاستجابة (لا الـ prompt)."""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        input_ids = input_ids[: self.max_length]
        labels_mask = labels_mask[: self.max_length]

        padding_len = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        labels_mask = labels_mask + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels_mask": torch.tensor(labels_mask),
        }

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        chosen_enc = self._tokenize_with_mask(ex.prompt, ex.chosen)
        rejected_enc = self._tokenize_with_mask(ex.prompt, ex.rejected)
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels_mask": chosen_enc["labels_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels_mask": rejected_enc["labels_mask"],
        }


def train_dpo(
    model_name: str,
    train_examples: list[DPOExample],
    beta: float = 0.1,
    num_epochs: int = 3,
    learning_rate: float = 5e-7,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # السياسة القابلة للتدريب (نبدأ من نموذج SFT)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy_model.train()

    # Reference Model: نسخة مجمَّدة تمامًا — لاحظ: نموذجان فقط، لا أربعة!
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    dataset = DPODataset(train_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_stats = {"loss": [], "reward_accuracy": [], "reward_margin": []}

        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels_mask = batch["chosen_labels_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels_mask = batch["rejected_labels_mask"].to(device)

            # log-probs من السياسة الحالية (قابلة للتدريب)
            policy_chosen_logps = compute_sequence_log_probs(
                policy_model, chosen_ids, chosen_mask, chosen_labels_mask
            )
            policy_rejected_logps = compute_sequence_log_probs(
                policy_model, rejected_ids, rejected_mask, rejected_labels_mask
            )

            # log-probs من Reference Model (بدون تدرّجات)
            with torch.no_grad():
                reference_chosen_logps = compute_sequence_log_probs(
                    reference_model, chosen_ids, chosen_mask, chosen_labels_mask
                )
                reference_rejected_logps = compute_sequence_log_probs(
                    reference_model, rejected_ids, rejected_mask, rejected_labels_mask
                )

            loss, stats = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_stats:
                epoch_stats[k].append(stats[k])

        avg_loss = sum(epoch_stats["loss"]) / len(epoch_stats["loss"])
        avg_acc = sum(epoch_stats["reward_accuracy"]) / len(epoch_stats["reward_accuracy"])
        avg_margin = sum(epoch_stats["reward_margin"]) / len(epoch_stats["reward_margin"])
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Reward Accuracy={avg_acc:.2%}, Reward Margin={avg_margin:.4f}")

    return policy_model

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-36-chapter-36.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
استخدام DPOTrainer من مكتبة trl للتطبيق الإنتاجي المُحسَّن
pip install trl transformers datasets peft
"""

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig  # يمكن دمج DPO مع LoRA مباشرة لكفاءة أكبر!


MODEL_NAME = "./sft-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# ملاحظة: إذا لم تُمرَّر ref_model، تستنتج المكتبة تلقائيًا نسخة مجمَّدة من نفس النموذج
# أو تستخدم نفس النموذج مع تعطيل محولات LoRA (إذا استُخدم PEFT) لتوفير الذاكرة

# تنسيق البيانات المطلوب من DPOTrainer: أعمدة prompt, chosen, rejected
raw_data = [
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف لبناء أنظمة قادرة على أداء مهام تتطلب عادة ذكاءً بشريًا، مثل التعلّم واتخاذ القرار وفهم اللغة.",
        "rejected": "الذكاء الاصطناعي شيء معقد جدًا ولا أحد يفهمه بشكل كامل، إنه مستقبل كل شيء تقريبًا.",
    },
    {
        "prompt": "كيف أتعلم البرمجة من الصفر؟",
        "chosen": "ابدأ بلغة سهلة مثل Python، تعلّم الأساسيات (المتغيرات، الحلقات، الدوال)، ثم مارس عبر مشاريع صغيرة تدريجية الصعوبة.",
        "rejected": "البرمجة صعبة جدًا، تحتاج سنوات طويلة، ومعظم الناس يفشلون فيها، لذا فكّر جيدًا قبل البدء.",
    },
    # في مشروع حقيقي: آلاف الأمثلة من بيانات تفضيل حقيقية
]

train_dataset = Dataset.from_list(raw_data)

dpo_config = DPOConfig(
    output_dir="./dpo-model",
    beta=0.1,                        # β في معادلة DPO — يتحكّم بقوة الانحراف المسموح
    learning_rate=5e-7,              # DPO يحتاج معدلات تعلّم منخفضة جدًا، أقل حتى من RLHF
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    max_length=512,
    max_prompt_length=256,
    logging_steps=1,
    report_to="none",
)

# يمكن دمج LoRA مباشرة — "QDPO" فعليًا، كفاءة ذاكرة مذهلة
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model=policy_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,  # إضافة هذا السطر وحده تُحوّل DPO العادي إلى نسخة فعّالة بالذاكرة
)

dpo_trainer.train()
dpo_trainer.save_model("./dpo-model-final")
