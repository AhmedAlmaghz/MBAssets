#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI
Large Language Model Engineering — KTO

Book slug : llm-engineering-69-chapters
Chapter   : ch-37-chapter-37 (lesson 37)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-37-chapter-37.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI
Large Language Model Engineering — KTO

Book slug : llm-engineering-69-chapters
Chapter   : ch-37-chapter-37 (lesson 37)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-37-chapter-37.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI
Large Language Model Engineering — KTO

Book slug : llm-engineering-69-chapters
Chapter   : ch-37-chapter-37 (lesson 37)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_37.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 37: الفصل 37: تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 37 lab: Compare simple KTO and ORPO objectives")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق KTO: محاذاة من بيانات فردية (مرغوبة/غير مرغوبة) بلا أزواج تفضيل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


class DesirabilityLabel(Enum):
    DESIRABLE = "desirable"
    UNDESIRABLE = "undesirable"


@dataclass
class KTOExample:
    """مثال فردي — لاحظ عدم وجود مفهوم 'زوج' هنا إطلاقًا."""
    prompt: str
    response: str
    label: DesirabilityLabel


def kto_loss(
    policy_logps: torch.Tensor,          # (batch,) log P_θ(y|x) لكل مثال في الدفعة
    reference_logps: torch.Tensor,       # (batch,) log P_ref(y|x) المقابلة
    labels: list[DesirabilityLabel],
    beta: float = 0.1,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.33,    # λ_U > λ_D لتطبيق كراهية الخسارة
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة KTO الكاملة. المرجعية z0 تُقدَّر من متوسط
    المكافأة الضمنية عبر الدفعة الحالية (تقريب عملي شائع).
    """
    implicit_rewards = beta * (policy_logps - reference_logps)  # r̂_θ(x,y) لكل مثال

    # تقدير نقطة المرجعية z0 كمتوسط الدفعة (KL تقريبي، كما في الورقة الأصلية)
    z0 = implicit_rewards.mean().detach()

    losses = []
    for i, label in enumerate(labels):
        r = implicit_rewards[i]
        if label == DesirabilityLabel.DESIRABLE:
            # نريد زيادة الاحتمال: v = sigmoid(β(r - z0)) يجب أن يقترب من 1
            v = torch.sigmoid(r - z0)
            loss_i = lambda_desirable * (1 - v)
        else:
            # نريد تقليل الاحتمال: v = sigmoid(β(z0 - r)) يجب أن يقترب من 1
            v = torch.sigmoid(z0 - r)
            loss_i = lambda_undesirable * (1 - v)
        losses.append(loss_i)

    loss = torch.stack(losses).mean()

    with torch.no_grad():
        desirable_mask = torch.tensor(
            [l == DesirabilityLabel.DESIRABLE for l in labels]
        )
        stats = {
            "loss": loss.item(),
            "mean_reward_desirable": implicit_rewards[desirable_mask].mean().item()
                if desirable_mask.any() else float("nan"),
            "mean_reward_undesirable": implicit_rewards[~desirable_mask].mean().item()
                if (~desirable_mask).any() else float("nan"),
            "reference_point_z0": z0.item(),
        }

    return loss, stats


# ------------------------------------------------------------------
# مثال بيانات واقعية: تقييمات 👍/👎 فردية من مستخدمين حقيقيين
# ------------------------------------------------------------------

kto_examples = [
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="عاصمة أستراليا هي كانبرا، وليست سيدني كما يعتقد كثيرون خطأً.",
        label=DesirabilityLabel.DESIRABLE,   # 👍 من المستخدم
    ),
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="أستراليا بلد كبير جدًا وفيه مدن كثيرة مشهورة مثل سيدني وملبورن.",
        label=DesirabilityLabel.UNDESIRABLE,  # 👎 — لم يُجب على السؤال فعليًا!
    ),
    KTOExample(
        prompt="اشرح الجاذبية ببساطة",
        response="الجاذبية قوة تجذب الأجسام نحو بعضها، وهي السبب في سقوط الأشياء نحو الأرض.",
        label=DesirabilityLabel.DESIRABLE,
    ),
    # لاحظ: لا يوجد شرط أن يكون لكل prompt مثالان (مفضَّل ومرفوض) — بيانات حرة تمامًا
]

print(f"عدد الأمثلة المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.DESIRABLE)}")
print(f"عدد الأمثلة غير المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.UNDESIRABLE)}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق ORPO: SFT + محاذاة في دالة خسارة واحدة، بلا Reference Model
"""

import torch
import torch.nn.functional as F


def compute_log_odds(sequence_log_prob: torch.Tensor, seq_length: torch.Tensor) -> torch.Tensor:
    """
    يحسب log(odds) = log(P / (1-P)) حيث P هو المتوسط الهندسي للاحتمال لكل توكن.
    """
    avg_log_prob = sequence_log_prob / seq_length  # log P الهندسي المتوسط
    prob = torch.exp(avg_log_prob).clamp(min=1e-7, max=1 - 1e-7)  # تجنّب log(0)
    log_odds = torch.log(prob) - torch.log(1 - prob)
    return log_odds


def orpo_loss(
    model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    chosen_labels_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    rejected_labels_mask: torch.Tensor,
    lambda_or: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة ORPO الكاملة: خسارة SFT على الاستجابة المفضَّلة
    + حدّ محاذاة يعتمد على نسبة الأرجحية، كل هذا بلا نموذج Reference!
    """
    # --- خسارة SFT القياسية (Cross-Entropy) على الاستجابة المفضَّلة فقط ---
    chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
    chosen_logits = chosen_outputs.logits[:, :-1, :]
    chosen_targets = chosen_input_ids[:, 1:]
    chosen_mask = chosen_labels_mask[:, 1:]

    ce_loss_per_token = F.cross_entropy(
        chosen_logits.reshape(-1, chosen_logits.size(-1)),
        chosen_targets.reshape(-1),
        reduction="none",
    ).reshape(chosen_targets.shape)
    sft_loss = (ce_loss_per_token * chosen_mask).sum() / chosen_mask.sum()

    # --- log-probabilities لحساب نسبة الأرجحية (لكل من المفضَّل والمرفوض) ---
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    chosen_token_lp = torch.gather(chosen_log_probs, 2, chosen_targets.unsqueeze(-1)).squeeze(-1)
    chosen_seq_lp = (chosen_token_lp * chosen_mask).sum(dim=-1)
    chosen_seq_len = chosen_mask.sum(dim=-1)

    rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
    rejected_logits = rejected_outputs.logits[:, :-1, :]
    rejected_targets = rejected_input_ids[:, 1:]
    rejected_mask = rejected_labels_mask[:, 1:]

    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    rejected_token_lp = torch.gather(rejected_log_probs, 2, rejected_targets.unsqueeze(-1)).squeeze(-1)
    rejected_seq_lp = (rejected_token_lp * rejected_mask).sum(dim=-1)
    rejected_seq_len = rejected_mask.sum(dim=-1)

    # --- نسبة الأرجحية (Odds Ratio) ---
    chosen_log_odds = compute_log_odds(chosen_seq_lp, chosen_seq_len)
    rejected_log_odds = compute_log_odds(rejected_seq_lp, rejected_seq_len)

    log_odds_ratio = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # --- الخسارة الكلية: SFT + λ × محاذاة، كل شيء في خطوة واحدة ---
    total_loss = sft_loss + lambda_or * or_loss

    stats = {
        "total_loss": total_loss.item(),
        "sft_loss": sft_loss.item(),
        "or_loss": or_loss.item(),
        "mean_log_odds_ratio": log_odds_ratio.mean().item(),
    }

    return total_loss, stats

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
الاستخدام الإنتاجي لـ KTOTrainer وORPOTrainer من مكتبة trl
"""

from trl import KTOConfig, KTOTrainer, ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# --- KTO: بيانات غير مُزاوجة، عمود "label" بولياني (True=مرغوب) ---
kto_data = Dataset.from_list([
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "عاصمتها كانبرا.", "label": True},
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "سيدني مدينة جميلة.", "label": False},
    {"prompt": "اشرح الجاذبية", "completion": "قوة تجذب الأجسام نحو بعضها.", "label": True},
])

kto_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
kto_config = KTOConfig(
    output_dir="./kto-model",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.33,   # كراهية الخسارة كما في نظرية التوقّع
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)
kto_trainer = KTOTrainer(
    model=kto_model, args=kto_config, train_dataset=kto_data, tokenizer=tokenizer,
)
kto_trainer.train()


# --- ORPO: نفس تنسيق بيانات DPO (prompt, chosen, rejected)، لكن بلا ref_model ---
orpo_data = Dataset.from_list([
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي فرع من علوم الحاسوب يبني أنظمة قادرة على محاكاة التفكير البشري.",
        "rejected": "لا أعرف بالضبط لكنه شيء متعلق بالحواسيب والروبوتات على ما أظن.",
    },
])

orpo_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
orpo_config = ORPOConfig(
    output_dir="./orpo-model",
    beta=0.1,                 # هنا β يلعب دور λ في معادلتنا أعلاه (تسمية المكتبة)
    learning_rate=8e-6,       # ORPO يتحمّل معدل تعلّم أعلى قليلًا من DPO
    per_device_train_batch_size=2,
    num_train_epochs=3,
)
orpo_trainer = ORPOTrainer(
    model=orpo_model, args=orpo_config, train_dataset=orpo_data, tokenizer=tokenizer,
    # لاحظ: لا وجود لمعامل ref_model على الإطلاق!
)
orpo_trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لمرحلة SL-CAI: نقد وتنقيح ذاتي موجَّه بدستور نصي
"""

from dataclasses import dataclass


CONSTITUTION_PRINCIPLES = [
    "اختر الاستجابة الأقل تحيّزًا والأكثر احترامًا لكل الفئات والثقافات.",
    "اختر الاستجابة التي تتجنّب تقديم معلومات قد تُستخدَم للإضرار بالآخرين.",
    "اختر الاستجابة الأكثر صدقًا وشفافية بشأن حدود معرفة النموذج.",
    "اختر الاستجابة التي تحترم خصوصية الأفراد ولا تُشجّع على انتهاكها.",
]


@dataclass
class ConstitutionalRevisionResult:
    original_response: str
    critique: str
    revised_response: str
    principle_used: str


def run_constitutional_revision(
    client,  # LLMClient من الفصل 33
    prompt: str,
    initial_response: str,
    principle: str,
) -> ConstitutionalRevisionResult:
    """تنفيذ خطوتَي النقد والتنقيح الذاتي وفق مبدأ دستوري واحد."""

    # الخطوة 1: النقد الذاتي
    critique_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة المُولَّدة: {initial_response}

المبدأ المطلوب تطبيقه: {principle}

مهمتك: انتقد الاستجابة أعلاه بصراحة تامة. هل تنتهك المبدأ المذكور؟
اشرح بالتفصيل أي جوانب مشكلة، أو اذكر صراحة أنها متوافقة تمامًا إن كانت كذلك."""

    critique = client.generate(critique_prompt, temperature=0.0)[0]

    # الخطوة 2: التنقيح بناءً على النقد
    revision_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة الأصلية: {initial_response}

النقد: {critique}

مهمتك: أعد كتابة الاستجابة لتكون متوافقة تمامًا مع المبدأ، مع معالجة كل نقطة
وردت في النقد أعلاه. إذا كان النقد يشير إلى أن الاستجابة جيدة بالفعل، أبقِها كما هي."""

    revised = client.generate(revision_prompt, temperature=0.0)[0]

    return ConstitutionalRevisionResult(
        original_response=initial_response,
        critique=critique,
        revised_response=revised,
        principle_used=principle,
    )


def build_cai_training_example(client, prompt: str) -> dict:
    """
    يُنفّذ الحلقة الكاملة: توليد أولي → نقد → تنقيح،
    عبر كل مبادئ الدستور بالتتابع (Chain of Revisions).
    """
    current_response = client.generate(prompt, temperature=0.7)[0]
    revision_history = []

    for principle in CONSTITUTION_PRINCIPLES:
        result = run_constitutional_revision(client, prompt, current_response, principle)
        revision_history.append(result)
        current_response = result.revised_response   # كل تنقيح يبني على سابقه

    return {
        "prompt": prompt,
        "initial_response": revision_history[0].original_response,
        "final_response": current_response,          # هذا ما يُستخدَم كبيانات SFT جديدة
        "revision_chain": revision_history,
    }


# استخدام فعلي (يتطلب HFLocalClient من الفصل 33)
# cai_example = build_cai_training_example(client, "كيف يمكنني اختراق حساب شخص آخر على فيسبوك؟")
# بعد عدة تنقيحات موجَّهة بالدستور، يُتوقَّع أن final_response يرفض بأدب تقديم هذه المعلومات
# مع شرح السبب، بدل رفض فظّ أو — أسوأ — تقديم المعلومات الضارة فعليًا

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 37: الفصل 37: تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 37 lab: Compare simple KTO and ORPO objectives")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق KTO: محاذاة من بيانات فردية (مرغوبة/غير مرغوبة) بلا أزواج تفضيل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


class DesirabilityLabel(Enum):
    DESIRABLE = "desirable"
    UNDESIRABLE = "undesirable"


@dataclass
class KTOExample:
    """مثال فردي — لاحظ عدم وجود مفهوم 'زوج' هنا إطلاقًا."""
    prompt: str
    response: str
    label: DesirabilityLabel


def kto_loss(
    policy_logps: torch.Tensor,          # (batch,) log P_θ(y|x) لكل مثال في الدفعة
    reference_logps: torch.Tensor,       # (batch,) log P_ref(y|x) المقابلة
    labels: list[DesirabilityLabel],
    beta: float = 0.1,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.33,    # λ_U > λ_D لتطبيق كراهية الخسارة
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة KTO الكاملة. المرجعية z0 تُقدَّر من متوسط
    المكافأة الضمنية عبر الدفعة الحالية (تقريب عملي شائع).
    """
    implicit_rewards = beta * (policy_logps - reference_logps)  # r̂_θ(x,y) لكل مثال

    # تقدير نقطة المرجعية z0 كمتوسط الدفعة (KL تقريبي، كما في الورقة الأصلية)
    z0 = implicit_rewards.mean().detach()

    losses = []
    for i, label in enumerate(labels):
        r = implicit_rewards[i]
        if label == DesirabilityLabel.DESIRABLE:
            # نريد زيادة الاحتمال: v = sigmoid(β(r - z0)) يجب أن يقترب من 1
            v = torch.sigmoid(r - z0)
            loss_i = lambda_desirable * (1 - v)
        else:
            # نريد تقليل الاحتمال: v = sigmoid(β(z0 - r)) يجب أن يقترب من 1
            v = torch.sigmoid(z0 - r)
            loss_i = lambda_undesirable * (1 - v)
        losses.append(loss_i)

    loss = torch.stack(losses).mean()

    with torch.no_grad():
        desirable_mask = torch.tensor(
            [l == DesirabilityLabel.DESIRABLE for l in labels]
        )
        stats = {
            "loss": loss.item(),
            "mean_reward_desirable": implicit_rewards[desirable_mask].mean().item()
                if desirable_mask.any() else float("nan"),
            "mean_reward_undesirable": implicit_rewards[~desirable_mask].mean().item()
                if (~desirable_mask).any() else float("nan"),
            "reference_point_z0": z0.item(),
        }

    return loss, stats


# ------------------------------------------------------------------
# مثال بيانات واقعية: تقييمات 👍/👎 فردية من مستخدمين حقيقيين
# ------------------------------------------------------------------

kto_examples = [
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="عاصمة أستراليا هي كانبرا، وليست سيدني كما يعتقد كثيرون خطأً.",
        label=DesirabilityLabel.DESIRABLE,   # 👍 من المستخدم
    ),
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="أستراليا بلد كبير جدًا وفيه مدن كثيرة مشهورة مثل سيدني وملبورن.",
        label=DesirabilityLabel.UNDESIRABLE,  # 👎 — لم يُجب على السؤال فعليًا!
    ),
    KTOExample(
        prompt="اشرح الجاذبية ببساطة",
        response="الجاذبية قوة تجذب الأجسام نحو بعضها، وهي السبب في سقوط الأشياء نحو الأرض.",
        label=DesirabilityLabel.DESIRABLE,
    ),
    # لاحظ: لا يوجد شرط أن يكون لكل prompt مثالان (مفضَّل ومرفوض) — بيانات حرة تمامًا
]

print(f"عدد الأمثلة المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.DESIRABLE)}")
print(f"عدد الأمثلة غير المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.UNDESIRABLE)}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق ORPO: SFT + محاذاة في دالة خسارة واحدة، بلا Reference Model
"""

import torch
import torch.nn.functional as F


def compute_log_odds(sequence_log_prob: torch.Tensor, seq_length: torch.Tensor) -> torch.Tensor:
    """
    يحسب log(odds) = log(P / (1-P)) حيث P هو المتوسط الهندسي للاحتمال لكل توكن.
    """
    avg_log_prob = sequence_log_prob / seq_length  # log P الهندسي المتوسط
    prob = torch.exp(avg_log_prob).clamp(min=1e-7, max=1 - 1e-7)  # تجنّب log(0)
    log_odds = torch.log(prob) - torch.log(1 - prob)
    return log_odds


def orpo_loss(
    model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    chosen_labels_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    rejected_labels_mask: torch.Tensor,
    lambda_or: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة ORPO الكاملة: خسارة SFT على الاستجابة المفضَّلة
    + حدّ محاذاة يعتمد على نسبة الأرجحية، كل هذا بلا نموذج Reference!
    """
    # --- خسارة SFT القياسية (Cross-Entropy) على الاستجابة المفضَّلة فقط ---
    chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
    chosen_logits = chosen_outputs.logits[:, :-1, :]
    chosen_targets = chosen_input_ids[:, 1:]
    chosen_mask = chosen_labels_mask[:, 1:]

    ce_loss_per_token = F.cross_entropy(
        chosen_logits.reshape(-1, chosen_logits.size(-1)),
        chosen_targets.reshape(-1),
        reduction="none",
    ).reshape(chosen_targets.shape)
    sft_loss = (ce_loss_per_token * chosen_mask).sum() / chosen_mask.sum()

    # --- log-probabilities لحساب نسبة الأرجحية (لكل من المفضَّل والمرفوض) ---
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    chosen_token_lp = torch.gather(chosen_log_probs, 2, chosen_targets.unsqueeze(-1)).squeeze(-1)
    chosen_seq_lp = (chosen_token_lp * chosen_mask).sum(dim=-1)
    chosen_seq_len = chosen_mask.sum(dim=-1)

    rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
    rejected_logits = rejected_outputs.logits[:, :-1, :]
    rejected_targets = rejected_input_ids[:, 1:]
    rejected_mask = rejected_labels_mask[:, 1:]

    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    rejected_token_lp = torch.gather(rejected_log_probs, 2, rejected_targets.unsqueeze(-1)).squeeze(-1)
    rejected_seq_lp = (rejected_token_lp * rejected_mask).sum(dim=-1)
    rejected_seq_len = rejected_mask.sum(dim=-1)

    # --- نسبة الأرجحية (Odds Ratio) ---
    chosen_log_odds = compute_log_odds(chosen_seq_lp, chosen_seq_len)
    rejected_log_odds = compute_log_odds(rejected_seq_lp, rejected_seq_len)

    log_odds_ratio = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # --- الخسارة الكلية: SFT + λ × محاذاة، كل شيء في خطوة واحدة ---
    total_loss = sft_loss + lambda_or * or_loss

    stats = {
        "total_loss": total_loss.item(),
        "sft_loss": sft_loss.item(),
        "or_loss": or_loss.item(),
        "mean_log_odds_ratio": log_odds_ratio.mean().item(),
    }

    return total_loss, stats

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
الاستخدام الإنتاجي لـ KTOTrainer وORPOTrainer من مكتبة trl
"""

from trl import KTOConfig, KTOTrainer, ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# --- KTO: بيانات غير مُزاوجة، عمود "label" بولياني (True=مرغوب) ---
kto_data = Dataset.from_list([
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "عاصمتها كانبرا.", "label": True},
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "سيدني مدينة جميلة.", "label": False},
    {"prompt": "اشرح الجاذبية", "completion": "قوة تجذب الأجسام نحو بعضها.", "label": True},
])

kto_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
kto_config = KTOConfig(
    output_dir="./kto-model",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.33,   # كراهية الخسارة كما في نظرية التوقّع
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)
kto_trainer = KTOTrainer(
    model=kto_model, args=kto_config, train_dataset=kto_data, tokenizer=tokenizer,
)
kto_trainer.train()


# --- ORPO: نفس تنسيق بيانات DPO (prompt, chosen, rejected)، لكن بلا ref_model ---
orpo_data = Dataset.from_list([
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي فرع من علوم الحاسوب يبني أنظمة قادرة على محاكاة التفكير البشري.",
        "rejected": "لا أعرف بالضبط لكنه شيء متعلق بالحواسيب والروبوتات على ما أظن.",
    },
])

orpo_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
orpo_config = ORPOConfig(
    output_dir="./orpo-model",
    beta=0.1,                 # هنا β يلعب دور λ في معادلتنا أعلاه (تسمية المكتبة)
    learning_rate=8e-6,       # ORPO يتحمّل معدل تعلّم أعلى قليلًا من DPO
    per_device_train_batch_size=2,
    num_train_epochs=3,
)
orpo_trainer = ORPOTrainer(
    model=orpo_model, args=orpo_config, train_dataset=orpo_data, tokenizer=tokenizer,
    # لاحظ: لا وجود لمعامل ref_model على الإطلاق!
)
orpo_trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لمرحلة SL-CAI: نقد وتنقيح ذاتي موجَّه بدستور نصي
"""

from dataclasses import dataclass


CONSTITUTION_PRINCIPLES = [
    "اختر الاستجابة الأقل تحيّزًا والأكثر احترامًا لكل الفئات والثقافات.",
    "اختر الاستجابة التي تتجنّب تقديم معلومات قد تُستخدَم للإضرار بالآخرين.",
    "اختر الاستجابة الأكثر صدقًا وشفافية بشأن حدود معرفة النموذج.",
    "اختر الاستجابة التي تحترم خصوصية الأفراد ولا تُشجّع على انتهاكها.",
]


@dataclass
class ConstitutionalRevisionResult:
    original_response: str
    critique: str
    revised_response: str
    principle_used: str


def run_constitutional_revision(
    client,  # LLMClient من الفصل 33
    prompt: str,
    initial_response: str,
    principle: str,
) -> ConstitutionalRevisionResult:
    """تنفيذ خطوتَي النقد والتنقيح الذاتي وفق مبدأ دستوري واحد."""

    # الخطوة 1: النقد الذاتي
    critique_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة المُولَّدة: {initial_response}

المبدأ المطلوب تطبيقه: {principle}

مهمتك: انتقد الاستجابة أعلاه بصراحة تامة. هل تنتهك المبدأ المذكور؟
اشرح بالتفصيل أي جوانب مشكلة، أو اذكر صراحة أنها متوافقة تمامًا إن كانت كذلك."""

    critique = client.generate(critique_prompt, temperature=0.0)[0]

    # الخطوة 2: التنقيح بناءً على النقد
    revision_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة الأصلية: {initial_response}

النقد: {critique}

مهمتك: أعد كتابة الاستجابة لتكون متوافقة تمامًا مع المبدأ، مع معالجة كل نقطة
وردت في النقد أعلاه. إذا كان النقد يشير إلى أن الاستجابة جيدة بالفعل، أبقِها كما هي."""

    revised = client.generate(revision_prompt, temperature=0.0)[0]

    return ConstitutionalRevisionResult(
        original_response=initial_response,
        critique=critique,
        revised_response=revised,
        principle_used=principle,
    )


def build_cai_training_example(client, prompt: str) -> dict:
    """
    يُنفّذ الحلقة الكاملة: توليد أولي → نقد → تنقيح،
    عبر كل مبادئ الدستور بالتتابع (Chain of Revisions).
    """
    current_response = client.generate(prompt, temperature=0.7)[0]
    revision_history = []

    for principle in CONSTITUTION_PRINCIPLES:
        result = run_constitutional_revision(client, prompt, current_response, principle)
        revision_history.append(result)
        current_response = result.revised_response   # كل تنقيح يبني على سابقه

    return {
        "prompt": prompt,
        "initial_response": revision_history[0].original_response,
        "final_response": current_response,          # هذا ما يُستخدَم كبيانات SFT جديدة
        "revision_chain": revision_history,
    }


# استخدام فعلي (يتطلب HFLocalClient من الفصل 33)
# cai_example = build_cai_training_example(client, "كيف يمكنني اختراق حساب شخص آخر على فيسبوك؟")
# بعد عدة تنقيحات موجَّهة بالدستور، يُتوقَّع أن final_response يرفض بأدب تقديم هذه المعلومات
# مع شرح السبب، بدل رفض فظّ أو — أسوأ — تقديم المعلومات الضارة فعليًا

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI
Large Language Model Engineering — KTO

Book slug : llm-engineering-69-chapters
Chapter   : ch-37-chapter-37 (lesson 37)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_37.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 37: الفصل 37: تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 37 lab: Compare simple KTO and ORPO objectives")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق KTO: محاذاة من بيانات فردية (مرغوبة/غير مرغوبة) بلا أزواج تفضيل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


class DesirabilityLabel(Enum):
    DESIRABLE = "desirable"
    UNDESIRABLE = "undesirable"


@dataclass
class KTOExample:
    """مثال فردي — لاحظ عدم وجود مفهوم 'زوج' هنا إطلاقًا."""
    prompt: str
    response: str
    label: DesirabilityLabel


def kto_loss(
    policy_logps: torch.Tensor,          # (batch,) log P_θ(y|x) لكل مثال في الدفعة
    reference_logps: torch.Tensor,       # (batch,) log P_ref(y|x) المقابلة
    labels: list[DesirabilityLabel],
    beta: float = 0.1,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.33,    # λ_U > λ_D لتطبيق كراهية الخسارة
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة KTO الكاملة. المرجعية z0 تُقدَّر من متوسط
    المكافأة الضمنية عبر الدفعة الحالية (تقريب عملي شائع).
    """
    implicit_rewards = beta * (policy_logps - reference_logps)  # r̂_θ(x,y) لكل مثال

    # تقدير نقطة المرجعية z0 كمتوسط الدفعة (KL تقريبي، كما في الورقة الأصلية)
    z0 = implicit_rewards.mean().detach()

    losses = []
    for i, label in enumerate(labels):
        r = implicit_rewards[i]
        if label == DesirabilityLabel.DESIRABLE:
            # نريد زيادة الاحتمال: v = sigmoid(β(r - z0)) يجب أن يقترب من 1
            v = torch.sigmoid(r - z0)
            loss_i = lambda_desirable * (1 - v)
        else:
            # نريد تقليل الاحتمال: v = sigmoid(β(z0 - r)) يجب أن يقترب من 1
            v = torch.sigmoid(z0 - r)
            loss_i = lambda_undesirable * (1 - v)
        losses.append(loss_i)

    loss = torch.stack(losses).mean()

    with torch.no_grad():
        desirable_mask = torch.tensor(
            [l == DesirabilityLabel.DESIRABLE for l in labels]
        )
        stats = {
            "loss": loss.item(),
            "mean_reward_desirable": implicit_rewards[desirable_mask].mean().item()
                if desirable_mask.any() else float("nan"),
            "mean_reward_undesirable": implicit_rewards[~desirable_mask].mean().item()
                if (~desirable_mask).any() else float("nan"),
            "reference_point_z0": z0.item(),
        }

    return loss, stats


# ------------------------------------------------------------------
# مثال بيانات واقعية: تقييمات 👍/👎 فردية من مستخدمين حقيقيين
# ------------------------------------------------------------------

kto_examples = [
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="عاصمة أستراليا هي كانبرا، وليست سيدني كما يعتقد كثيرون خطأً.",
        label=DesirabilityLabel.DESIRABLE,   # 👍 من المستخدم
    ),
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="أستراليا بلد كبير جدًا وفيه مدن كثيرة مشهورة مثل سيدني وملبورن.",
        label=DesirabilityLabel.UNDESIRABLE,  # 👎 — لم يُجب على السؤال فعليًا!
    ),
    KTOExample(
        prompt="اشرح الجاذبية ببساطة",
        response="الجاذبية قوة تجذب الأجسام نحو بعضها، وهي السبب في سقوط الأشياء نحو الأرض.",
        label=DesirabilityLabel.DESIRABLE,
    ),
    # لاحظ: لا يوجد شرط أن يكون لكل prompt مثالان (مفضَّل ومرفوض) — بيانات حرة تمامًا
]

print(f"عدد الأمثلة المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.DESIRABLE)}")
print(f"عدد الأمثلة غير المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.UNDESIRABLE)}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق ORPO: SFT + محاذاة في دالة خسارة واحدة، بلا Reference Model
"""

import torch
import torch.nn.functional as F


def compute_log_odds(sequence_log_prob: torch.Tensor, seq_length: torch.Tensor) -> torch.Tensor:
    """
    يحسب log(odds) = log(P / (1-P)) حيث P هو المتوسط الهندسي للاحتمال لكل توكن.
    """
    avg_log_prob = sequence_log_prob / seq_length  # log P الهندسي المتوسط
    prob = torch.exp(avg_log_prob).clamp(min=1e-7, max=1 - 1e-7)  # تجنّب log(0)
    log_odds = torch.log(prob) - torch.log(1 - prob)
    return log_odds


def orpo_loss(
    model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    chosen_labels_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    rejected_labels_mask: torch.Tensor,
    lambda_or: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة ORPO الكاملة: خسارة SFT على الاستجابة المفضَّلة
    + حدّ محاذاة يعتمد على نسبة الأرجحية، كل هذا بلا نموذج Reference!
    """
    # --- خسارة SFT القياسية (Cross-Entropy) على الاستجابة المفضَّلة فقط ---
    chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
    chosen_logits = chosen_outputs.logits[:, :-1, :]
    chosen_targets = chosen_input_ids[:, 1:]
    chosen_mask = chosen_labels_mask[:, 1:]

    ce_loss_per_token = F.cross_entropy(
        chosen_logits.reshape(-1, chosen_logits.size(-1)),
        chosen_targets.reshape(-1),
        reduction="none",
    ).reshape(chosen_targets.shape)
    sft_loss = (ce_loss_per_token * chosen_mask).sum() / chosen_mask.sum()

    # --- log-probabilities لحساب نسبة الأرجحية (لكل من المفضَّل والمرفوض) ---
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    chosen_token_lp = torch.gather(chosen_log_probs, 2, chosen_targets.unsqueeze(-1)).squeeze(-1)
    chosen_seq_lp = (chosen_token_lp * chosen_mask).sum(dim=-1)
    chosen_seq_len = chosen_mask.sum(dim=-1)

    rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
    rejected_logits = rejected_outputs.logits[:, :-1, :]
    rejected_targets = rejected_input_ids[:, 1:]
    rejected_mask = rejected_labels_mask[:, 1:]

    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    rejected_token_lp = torch.gather(rejected_log_probs, 2, rejected_targets.unsqueeze(-1)).squeeze(-1)
    rejected_seq_lp = (rejected_token_lp * rejected_mask).sum(dim=-1)
    rejected_seq_len = rejected_mask.sum(dim=-1)

    # --- نسبة الأرجحية (Odds Ratio) ---
    chosen_log_odds = compute_log_odds(chosen_seq_lp, chosen_seq_len)
    rejected_log_odds = compute_log_odds(rejected_seq_lp, rejected_seq_len)

    log_odds_ratio = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # --- الخسارة الكلية: SFT + λ × محاذاة، كل شيء في خطوة واحدة ---
    total_loss = sft_loss + lambda_or * or_loss

    stats = {
        "total_loss": total_loss.item(),
        "sft_loss": sft_loss.item(),
        "or_loss": or_loss.item(),
        "mean_log_odds_ratio": log_odds_ratio.mean().item(),
    }

    return total_loss, stats

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
الاستخدام الإنتاجي لـ KTOTrainer وORPOTrainer من مكتبة trl
"""

from trl import KTOConfig, KTOTrainer, ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# --- KTO: بيانات غير مُزاوجة، عمود "label" بولياني (True=مرغوب) ---
kto_data = Dataset.from_list([
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "عاصمتها كانبرا.", "label": True},
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "سيدني مدينة جميلة.", "label": False},
    {"prompt": "اشرح الجاذبية", "completion": "قوة تجذب الأجسام نحو بعضها.", "label": True},
])

kto_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
kto_config = KTOConfig(
    output_dir="./kto-model",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.33,   # كراهية الخسارة كما في نظرية التوقّع
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)
kto_trainer = KTOTrainer(
    model=kto_model, args=kto_config, train_dataset=kto_data, tokenizer=tokenizer,
)
kto_trainer.train()


# --- ORPO: نفس تنسيق بيانات DPO (prompt, chosen, rejected)، لكن بلا ref_model ---
orpo_data = Dataset.from_list([
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي فرع من علوم الحاسوب يبني أنظمة قادرة على محاكاة التفكير البشري.",
        "rejected": "لا أعرف بالضبط لكنه شيء متعلق بالحواسيب والروبوتات على ما أظن.",
    },
])

orpo_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
orpo_config = ORPOConfig(
    output_dir="./orpo-model",
    beta=0.1,                 # هنا β يلعب دور λ في معادلتنا أعلاه (تسمية المكتبة)
    learning_rate=8e-6,       # ORPO يتحمّل معدل تعلّم أعلى قليلًا من DPO
    per_device_train_batch_size=2,
    num_train_epochs=3,
)
orpo_trainer = ORPOTrainer(
    model=orpo_model, args=orpo_config, train_dataset=orpo_data, tokenizer=tokenizer,
    # لاحظ: لا وجود لمعامل ref_model على الإطلاق!
)
orpo_trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-07-fine-tuning-alignment/chapter-37.md
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لمرحلة SL-CAI: نقد وتنقيح ذاتي موجَّه بدستور نصي
"""

from dataclasses import dataclass


CONSTITUTION_PRINCIPLES = [
    "اختر الاستجابة الأقل تحيّزًا والأكثر احترامًا لكل الفئات والثقافات.",
    "اختر الاستجابة التي تتجنّب تقديم معلومات قد تُستخدَم للإضرار بالآخرين.",
    "اختر الاستجابة الأكثر صدقًا وشفافية بشأن حدود معرفة النموذج.",
    "اختر الاستجابة التي تحترم خصوصية الأفراد ولا تُشجّع على انتهاكها.",
]


@dataclass
class ConstitutionalRevisionResult:
    original_response: str
    critique: str
    revised_response: str
    principle_used: str


def run_constitutional_revision(
    client,  # LLMClient من الفصل 33
    prompt: str,
    initial_response: str,
    principle: str,
) -> ConstitutionalRevisionResult:
    """تنفيذ خطوتَي النقد والتنقيح الذاتي وفق مبدأ دستوري واحد."""

    # الخطوة 1: النقد الذاتي
    critique_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة المُولَّدة: {initial_response}

المبدأ المطلوب تطبيقه: {principle}

مهمتك: انتقد الاستجابة أعلاه بصراحة تامة. هل تنتهك المبدأ المذكور؟
اشرح بالتفصيل أي جوانب مشكلة، أو اذكر صراحة أنها متوافقة تمامًا إن كانت كذلك."""

    critique = client.generate(critique_prompt, temperature=0.0)[0]

    # الخطوة 2: التنقيح بناءً على النقد
    revision_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة الأصلية: {initial_response}

النقد: {critique}

مهمتك: أعد كتابة الاستجابة لتكون متوافقة تمامًا مع المبدأ، مع معالجة كل نقطة
وردت في النقد أعلاه. إذا كان النقد يشير إلى أن الاستجابة جيدة بالفعل، أبقِها كما هي."""

    revised = client.generate(revision_prompt, temperature=0.0)[0]

    return ConstitutionalRevisionResult(
        original_response=initial_response,
        critique=critique,
        revised_response=revised,
        principle_used=principle,
    )


def build_cai_training_example(client, prompt: str) -> dict:
    """
    يُنفّذ الحلقة الكاملة: توليد أولي → نقد → تنقيح،
    عبر كل مبادئ الدستور بالتتابع (Chain of Revisions).
    """
    current_response = client.generate(prompt, temperature=0.7)[0]
    revision_history = []

    for principle in CONSTITUTION_PRINCIPLES:
        result = run_constitutional_revision(client, prompt, current_response, principle)
        revision_history.append(result)
        current_response = result.revised_response   # كل تنقيح يبني على سابقه

    return {
        "prompt": prompt,
        "initial_response": revision_history[0].original_response,
        "final_response": current_response,          # هذا ما يُستخدَم كبيانات SFT جديدة
        "revision_chain": revision_history,
    }


# استخدام فعلي (يتطلب HFLocalClient من الفصل 33)
# cai_example = build_cai_training_example(client, "كيف يمكنني اختراق حساب شخص آخر على فيسبوك؟")
# بعد عدة تنقيحات موجَّهة بالدستور، يُتوقَّع أن final_response يرفض بأدب تقديم هذه المعلومات
# مع شرح السبب، بدل رفض فظّ أو — أسوأ — تقديم المعلومات الضارة فعليًا

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 37: الفصل 37: تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 37 lab: Compare simple KTO and ORPO objectives")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق KTO: محاذاة من بيانات فردية (مرغوبة/غير مرغوبة) بلا أزواج تفضيل
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


class DesirabilityLabel(Enum):
    DESIRABLE = "desirable"
    UNDESIRABLE = "undesirable"


@dataclass
class KTOExample:
    """مثال فردي — لاحظ عدم وجود مفهوم 'زوج' هنا إطلاقًا."""
    prompt: str
    response: str
    label: DesirabilityLabel


def kto_loss(
    policy_logps: torch.Tensor,          # (batch,) log P_θ(y|x) لكل مثال في الدفعة
    reference_logps: torch.Tensor,       # (batch,) log P_ref(y|x) المقابلة
    labels: list[DesirabilityLabel],
    beta: float = 0.1,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.33,    # λ_U > λ_D لتطبيق كراهية الخسارة
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة KTO الكاملة. المرجعية z0 تُقدَّر من متوسط
    المكافأة الضمنية عبر الدفعة الحالية (تقريب عملي شائع).
    """
    implicit_rewards = beta * (policy_logps - reference_logps)  # r̂_θ(x,y) لكل مثال

    # تقدير نقطة المرجعية z0 كمتوسط الدفعة (KL تقريبي، كما في الورقة الأصلية)
    z0 = implicit_rewards.mean().detach()

    losses = []
    for i, label in enumerate(labels):
        r = implicit_rewards[i]
        if label == DesirabilityLabel.DESIRABLE:
            # نريد زيادة الاحتمال: v = sigmoid(β(r - z0)) يجب أن يقترب من 1
            v = torch.sigmoid(r - z0)
            loss_i = lambda_desirable * (1 - v)
        else:
            # نريد تقليل الاحتمال: v = sigmoid(β(z0 - r)) يجب أن يقترب من 1
            v = torch.sigmoid(z0 - r)
            loss_i = lambda_undesirable * (1 - v)
        losses.append(loss_i)

    loss = torch.stack(losses).mean()

    with torch.no_grad():
        desirable_mask = torch.tensor(
            [l == DesirabilityLabel.DESIRABLE for l in labels]
        )
        stats = {
            "loss": loss.item(),
            "mean_reward_desirable": implicit_rewards[desirable_mask].mean().item()
                if desirable_mask.any() else float("nan"),
            "mean_reward_undesirable": implicit_rewards[~desirable_mask].mean().item()
                if (~desirable_mask).any() else float("nan"),
            "reference_point_z0": z0.item(),
        }

    return loss, stats


# ------------------------------------------------------------------
# مثال بيانات واقعية: تقييمات 👍/👎 فردية من مستخدمين حقيقيين
# ------------------------------------------------------------------

kto_examples = [
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="عاصمة أستراليا هي كانبرا، وليست سيدني كما يعتقد كثيرون خطأً.",
        label=DesirabilityLabel.DESIRABLE,   # 👍 من المستخدم
    ),
    KTOExample(
        prompt="ما هي عاصمة أستراليا؟",
        response="أستراليا بلد كبير جدًا وفيه مدن كثيرة مشهورة مثل سيدني وملبورن.",
        label=DesirabilityLabel.UNDESIRABLE,  # 👎 — لم يُجب على السؤال فعليًا!
    ),
    KTOExample(
        prompt="اشرح الجاذبية ببساطة",
        response="الجاذبية قوة تجذب الأجسام نحو بعضها، وهي السبب في سقوط الأشياء نحو الأرض.",
        label=DesirabilityLabel.DESIRABLE,
    ),
    # لاحظ: لا يوجد شرط أن يكون لكل prompt مثالان (مفضَّل ومرفوض) — بيانات حرة تمامًا
]

print(f"عدد الأمثلة المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.DESIRABLE)}")
print(f"عدد الأمثلة غير المرغوبة: {sum(1 for e in kto_examples if e.label == DesirabilityLabel.UNDESIRABLE)}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق ORPO: SFT + محاذاة في دالة خسارة واحدة، بلا Reference Model
"""

import torch
import torch.nn.functional as F


def compute_log_odds(sequence_log_prob: torch.Tensor, seq_length: torch.Tensor) -> torch.Tensor:
    """
    يحسب log(odds) = log(P / (1-P)) حيث P هو المتوسط الهندسي للاحتمال لكل توكن.
    """
    avg_log_prob = sequence_log_prob / seq_length  # log P الهندسي المتوسط
    prob = torch.exp(avg_log_prob).clamp(min=1e-7, max=1 - 1e-7)  # تجنّب log(0)
    log_odds = torch.log(prob) - torch.log(1 - prob)
    return log_odds


def orpo_loss(
    model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    chosen_labels_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor,
    rejected_labels_mask: torch.Tensor,
    lambda_or: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    دالة خسارة ORPO الكاملة: خسارة SFT على الاستجابة المفضَّلة
    + حدّ محاذاة يعتمد على نسبة الأرجحية، كل هذا بلا نموذج Reference!
    """
    # --- خسارة SFT القياسية (Cross-Entropy) على الاستجابة المفضَّلة فقط ---
    chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
    chosen_logits = chosen_outputs.logits[:, :-1, :]
    chosen_targets = chosen_input_ids[:, 1:]
    chosen_mask = chosen_labels_mask[:, 1:]

    ce_loss_per_token = F.cross_entropy(
        chosen_logits.reshape(-1, chosen_logits.size(-1)),
        chosen_targets.reshape(-1),
        reduction="none",
    ).reshape(chosen_targets.shape)
    sft_loss = (ce_loss_per_token * chosen_mask).sum() / chosen_mask.sum()

    # --- log-probabilities لحساب نسبة الأرجحية (لكل من المفضَّل والمرفوض) ---
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    chosen_token_lp = torch.gather(chosen_log_probs, 2, chosen_targets.unsqueeze(-1)).squeeze(-1)
    chosen_seq_lp = (chosen_token_lp * chosen_mask).sum(dim=-1)
    chosen_seq_len = chosen_mask.sum(dim=-1)

    rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
    rejected_logits = rejected_outputs.logits[:, :-1, :]
    rejected_targets = rejected_input_ids[:, 1:]
    rejected_mask = rejected_labels_mask[:, 1:]

    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    rejected_token_lp = torch.gather(rejected_log_probs, 2, rejected_targets.unsqueeze(-1)).squeeze(-1)
    rejected_seq_lp = (rejected_token_lp * rejected_mask).sum(dim=-1)
    rejected_seq_len = rejected_mask.sum(dim=-1)

    # --- نسبة الأرجحية (Odds Ratio) ---
    chosen_log_odds = compute_log_odds(chosen_seq_lp, chosen_seq_len)
    rejected_log_odds = compute_log_odds(rejected_seq_lp, rejected_seq_len)

    log_odds_ratio = chosen_log_odds - rejected_log_odds
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # --- الخسارة الكلية: SFT + λ × محاذاة، كل شيء في خطوة واحدة ---
    total_loss = sft_loss + lambda_or * or_loss

    stats = {
        "total_loss": total_loss.item(),
        "sft_loss": sft_loss.item(),
        "or_loss": or_loss.item(),
        "mean_log_odds_ratio": log_odds_ratio.mean().item(),
    }

    return total_loss, stats

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
الاستخدام الإنتاجي لـ KTOTrainer وORPOTrainer من مكتبة trl
"""

from trl import KTOConfig, KTOTrainer, ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# --- KTO: بيانات غير مُزاوجة، عمود "label" بولياني (True=مرغوب) ---
kto_data = Dataset.from_list([
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "عاصمتها كانبرا.", "label": True},
    {"prompt": "ما هي عاصمة أستراليا؟", "completion": "سيدني مدينة جميلة.", "label": False},
    {"prompt": "اشرح الجاذبية", "completion": "قوة تجذب الأجسام نحو بعضها.", "label": True},
])

kto_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
kto_config = KTOConfig(
    output_dir="./kto-model",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.33,   # كراهية الخسارة كما في نظرية التوقّع
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)
kto_trainer = KTOTrainer(
    model=kto_model, args=kto_config, train_dataset=kto_data, tokenizer=tokenizer,
)
kto_trainer.train()


# --- ORPO: نفس تنسيق بيانات DPO (prompt, chosen, rejected)، لكن بلا ref_model ---
orpo_data = Dataset.from_list([
    {
        "prompt": "اشرح لي ما هو الذكاء الاصطناعي",
        "chosen": "الذكاء الاصطناعي فرع من علوم الحاسوب يبني أنظمة قادرة على محاكاة التفكير البشري.",
        "rejected": "لا أعرف بالضبط لكنه شيء متعلق بالحواسيب والروبوتات على ما أظن.",
    },
])

orpo_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
orpo_config = ORPOConfig(
    output_dir="./orpo-model",
    beta=0.1,                 # هنا β يلعب دور λ في معادلتنا أعلاه (تسمية المكتبة)
    learning_rate=8e-6,       # ORPO يتحمّل معدل تعلّم أعلى قليلًا من DPO
    per_device_train_batch_size=2,
    num_train_epochs=3,
)
orpo_trainer = ORPOTrainer(
    model=orpo_model, args=orpo_config, train_dataset=orpo_data, tokenizer=tokenizer,
    # لاحظ: لا وجود لمعامل ref_model على الإطلاق!
)
orpo_trainer.train()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-37-chapter-37.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
تطبيق مبسّط لمرحلة SL-CAI: نقد وتنقيح ذاتي موجَّه بدستور نصي
"""

from dataclasses import dataclass


CONSTITUTION_PRINCIPLES = [
    "اختر الاستجابة الأقل تحيّزًا والأكثر احترامًا لكل الفئات والثقافات.",
    "اختر الاستجابة التي تتجنّب تقديم معلومات قد تُستخدَم للإضرار بالآخرين.",
    "اختر الاستجابة الأكثر صدقًا وشفافية بشأن حدود معرفة النموذج.",
    "اختر الاستجابة التي تحترم خصوصية الأفراد ولا تُشجّع على انتهاكها.",
]


@dataclass
class ConstitutionalRevisionResult:
    original_response: str
    critique: str
    revised_response: str
    principle_used: str


def run_constitutional_revision(
    client,  # LLMClient من الفصل 33
    prompt: str,
    initial_response: str,
    principle: str,
) -> ConstitutionalRevisionResult:
    """تنفيذ خطوتَي النقد والتنقيح الذاتي وفق مبدأ دستوري واحد."""

    # الخطوة 1: النقد الذاتي
    critique_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة المُولَّدة: {initial_response}

المبدأ المطلوب تطبيقه: {principle}

مهمتك: انتقد الاستجابة أعلاه بصراحة تامة. هل تنتهك المبدأ المذكور؟
اشرح بالتفصيل أي جوانب مشكلة، أو اذكر صراحة أنها متوافقة تمامًا إن كانت كذلك."""

    critique = client.generate(critique_prompt, temperature=0.0)[0]

    # الخطوة 2: التنقيح بناءً على النقد
    revision_prompt = f"""السؤال الأصلي: {prompt}

الاستجابة الأصلية: {initial_response}

النقد: {critique}

مهمتك: أعد كتابة الاستجابة لتكون متوافقة تمامًا مع المبدأ، مع معالجة كل نقطة
وردت في النقد أعلاه. إذا كان النقد يشير إلى أن الاستجابة جيدة بالفعل، أبقِها كما هي."""

    revised = client.generate(revision_prompt, temperature=0.0)[0]

    return ConstitutionalRevisionResult(
        original_response=initial_response,
        critique=critique,
        revised_response=revised,
        principle_used=principle,
    )


def build_cai_training_example(client, prompt: str) -> dict:
    """
    يُنفّذ الحلقة الكاملة: توليد أولي → نقد → تنقيح،
    عبر كل مبادئ الدستور بالتتابع (Chain of Revisions).
    """
    current_response = client.generate(prompt, temperature=0.7)[0]
    revision_history = []

    for principle in CONSTITUTION_PRINCIPLES:
        result = run_constitutional_revision(client, prompt, current_response, principle)
        revision_history.append(result)
        current_response = result.revised_response   # كل تنقيح يبني على سابقه

    return {
        "prompt": prompt,
        "initial_response": revision_history[0].original_response,
        "final_response": current_response,          # هذا ما يُستخدَم كبيانات SFT جديدة
        "revision_chain": revision_history,
    }


# استخدام فعلي (يتطلب HFLocalClient من الفصل 33)
# cai_example = build_cai_training_example(client, "كيف يمكنني اختراق حساب شخص آخر على فيسبوك؟")
# بعد عدة تنقيحات موجَّهة بالدستور، يُتوقَّع أن final_response يرفض بأدب تقديم هذه المعلومات
# مع شرح السبب، بدل رفض فظّ أو — أسوأ — تقديم المعلومات الضارة فعليًا
